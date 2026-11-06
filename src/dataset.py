import logging
import os
import json
import hashlib

from pathlib import Path
from glob import glob
from multiprocessing import Pool, cpu_count
import multiprocessing.util

import torch
from torch.utils.data import Dataset

import uproot
import awkward as ak
import numpy as np

from tqdm import tqdm

from utils import get_feature_names

TRACK_FEATURES = get_feature_names()

multiprocessing.util.log_to_stderr(logging.DEBUG)

# Extra per-track quantities cached next to the features (not model inputs).
# Used to build per-sample training weights.
AUX_BRANCHES = ["trk_pt", "trk_dxy"]

def compute_upweights(aux, pt_min=5.0, dxy_min=0.5, factor=5.0):
    """Per-track weights: `factor` for tracks with pT > pt_min OR |dxy| > dxy_min, 1 otherwise.
    aux: tensor [n_tracks, len(AUX_BRANCHES)]."""
    pt = aux[:, AUX_BRANCHES.index("trk_pt")]
    dxy = aux[:, AUX_BRANCHES.index("trk_dxy")]
    selected = (pt > pt_min) | (dxy.abs() > dxy_min)
    weights = torch.ones(len(aux), dtype=torch.float32)
    weights[selected] = float(factor)
    return weights

def track_labels(trk_sim_trk_idx):
    """Per-track label from trk_simTrkIdx ([event][track][match]): True if the track has
    at least one matched sim particle (real), False otherwise (fake). Flat over events."""
    return ak.fill_none(ak.firsts(ak.flatten(trk_sim_trk_idx, axis=1)), -1) != -1

def process_root_file(file_path):
    pid = os.getpid()
    try:
        logging.info(f"[PID {pid}] Starting {file_path}")
        with uproot.open(file_path + ":trackingNtuple/tree") as tree:
            branches = list(dict.fromkeys(TRACK_FEATURES + AUX_BRANCHES + ["trk_simTrkIdx"]))
            arrays = tree.arrays(branches, library="ak", entry_stop=50)
            
            features = {col: arrays[col] for col in TRACK_FEATURES}
            aux = {col: arrays[col] for col in AUX_BRANCHES}
            labels = track_labels(arrays["trk_simTrkIdx"])

            n_tracks = len(ak.flatten(arrays[TRACK_FEATURES[0]]))
            logging.info(
                f"[PID {pid}] Returning {file_path}: "
                f"{n_tracks} tracks"
            )
            return features, labels, aux
    except Exception as e:
        print(f"[PID {pid}]: Error processing {file_path}: {e}")
        return None, None, None

def expand_input_files(input_files):
    """Glob pattern, or list of patterns/paths, -> sorted list of unique files."""
    patterns = [input_files] if isinstance(input_files, str) else list(input_files)
    return sorted({f for pattern in patterns for f in glob(str(pattern))})

def compute_data_key(file_list):
    file_info = '|'.join([f"{f}:{os.path.getmtime(f)}" for f in file_list if os.path.exists(f)])
    feature_info = '|'.join(TRACK_FEATURES)
    return hashlib.md5((file_info + feature_info).encode()).hexdigest()[:16]

def find_dataset_metadata(input_files, data_dir=None):
    """Return the cached .meta.json path for a set of input files without loading the data, or None."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    file_list = expand_input_files(input_files)
    if not file_list:
        return None
    meta_path = Path(data_dir) / f'dataset_{compute_data_key(file_list)}.meta.json'
    return meta_path if meta_path.exists() else None

class TrackDataset(Dataset):
    def __init__(self, input_files, transform=None, data_dir=None, feature_min=None, feature_max=None,
                 upweight=None, **kwargs):
        """feature_min/feature_max: if given, the scaler uses these (e.g. training-set stats)
        instead of this dataset's own min/max.
        upweight: optional dict {pt_min, dxy_min, factor}. If given, samples are returned as
        (data, label, weight), with weight = factor for tracks with pT > pt_min and |dxy| > dxy_min."""
        self.input_files_pattern = input_files
        self.transform_obj = transform

        self._data = None
        self._labels = None
        self._weights = None
        self._metadata = None
        
        if data_dir is None:
            logging.warning("data_dir not specified, using default './data'")
            data_dir = Path(__file__).parent.parent / 'data'
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir

        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_list = expand_input_files(input_files)
        if len(file_list) == 0:
            raise ValueError(f"No files found matching pattern: {input_files}")
        
        logging.info(f"Found {len(file_list)} ROOT files")
        
        data_key = self._generate_data_key(file_list)
        self.data_key = data_key
        self.data_prefix = data_dir / f'dataset_{data_key}'

        self.data_file = self.data_prefix.with_suffix('.data.pt')
        self.labels_file = self.data_prefix.with_suffix('.labels.pt')
        self.aux_file = self.data_prefix.with_suffix('.aux.pt')
        self.metadata_file = self.data_prefix.with_suffix('.meta.json')

        if self.data_file.exists() and self.labels_file.exists() and self.metadata_file.exists():
            logging.info(f"Loading from data: {self.data_prefix.name}")
            self._load_data()
        else:
            logging.info(f"Creating data: {self.data_prefix.name}")
            self._create_data(file_list)
            self._load_data()

        if upweight is not None:
            self._setup_weights(upweight)
        
        self._feature_stats = None
        if transform is not None and isinstance(transform, type):
            stats = self.get_dataset_statistics()
            if feature_min is not None and feature_max is not None:
                logging.info("Scaling with externally provided feature min/max")
                stats['feature_min'] = torch.as_tensor(feature_min, dtype=torch.float32)
                stats['feature_max'] = torch.as_tensor(feature_max, dtype=torch.float32)
            self._feature_stats = {
                'feature_min': stats['feature_min'],
                'feature_max': stats['feature_max']
            }
            logging.info(f"Using transform: {transform.__name__}")
            self.transform_obj = transform(
                data=None,
                feature_min=stats['feature_min'],
                feature_max=stats['feature_max']
            )
        
    def _generate_data_key(self, file_list):
        hash_key = compute_data_key(file_list)
        logging.info(f"Generated data key: {hash_key}")
        return hash_key
    
    def _load_data(self):
        if self._data is None:
            logging.info(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, 'r') as f:
                self._metadata = json.load(f)
            logging.info(f"Loaded metadata from {self.metadata_file}")

            logging.info(f"Loading data from {self.data_file} and {self.labels_file}")
            self._data = torch.load(self.data_file, weights_only=False)
            self._labels = torch.load(self.labels_file, weights_only=False)
            logging.info(f"Loaded data from {self.data_file} and {self.labels_file}")
    
    def _setup_weights(self, upweight):
        if not self.aux_file.exists():
            raise FileNotFoundError(
                f"Upweighting needs per-track pT/dxy from {self.aux_file}, which does not exist. "
                f"This cache was built before pT/dxy were saved. Move or delete "
                f"{self.data_prefix}.* so the dataset is rebuilt from the ROOT files."
            )
        aux = torch.load(self.aux_file, weights_only=False)
        assert len(aux) == len(self._labels), "aux cache does not match labels cache"
        pt_min = float(upweight.get('pt_min', 5.0))
        dxy_min = float(upweight.get('dxy_min', 0.5))
        factor = float(upweight.get('factor', 5.0))
        self._weights = compute_upweights(aux, pt_min=pt_min, dxy_min=dxy_min, factor=factor)
        n_up = int((self._weights != 1).sum())
        logging.info(
            f"Upweighting {n_up:,} / {len(self._weights):,} tracks with pT > {pt_min} GeV "
            f"and |dxy| > {dxy_min} cm by a factor {factor}"
        )

    def _create_data(self, file_list):
        # logging.info(f"Processing ROOT files in parallel using {cpu_count() // 16} workers...")
        logging.info(f"Processing ROOT files in parallel using 4 workers...")
        all_data_chunks = []
        all_labels_chunks = []
        chunk_lengths = []
        chunk_dir = Path(f"{self.data_dir}/dataset_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_idx = 0 
        n_real = 0
        n_fake = 0
        n_features = len(TRACK_FEATURES)
        feature_min = None
        feature_max = None
        
        with Pool(processes=4) as pool:
            for features, labels, aux in tqdm(pool.imap_unordered(process_root_file, file_list, chunksize=1), 
                                        total=len(file_list), desc="Processing ROOT files"):
                if features is None:
                    continue
                
                data_numpy = np.column_stack([
                    ak.to_numpy(ak.flatten(features[col])) 
                    for col in TRACK_FEATURES
                ]).astype('float32')
                labels_numpy = labels.to_numpy().astype('float32')
                aux_numpy = np.column_stack([
                    ak.to_numpy(ak.flatten(aux[col]))
                    for col in AUX_BRANCHES
                ]).astype('float32')
                
                if len(data_numpy) == 0:
                    continue
                chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.npz"

                # data_numpy = np.column_stack([
                #     ak.to_numpy(ak.flatten(features[col])) 
                #     for col in TRACK_FEATURES
                # ]).astype('float32')
                # labels_numpy = labels.to_numpy().astype('float32')

                # Remove tracks with NaN or inf in any feature
                valid_mask = np.isfinite(data_numpy).all(axis=1)

                data_numpy = data_numpy[valid_mask]
                labels_numpy = labels_numpy[valid_mask]
                aux_numpy = aux_numpy[valid_mask]

                n_removed = (~valid_mask).sum()
                if n_removed > 0:
                    logging.info(f"Removed {n_removed} tracks with NaN/inf features")

                if len(data_numpy) == 0:
                    continue
                
                n_real += int((labels_numpy == 1).sum())
                n_fake += int((labels_numpy == 0).sum())
                
                if feature_min is None:
                    feature_min = data_numpy.min(axis=0)
                    feature_max = data_numpy.max(axis=0)
                else:
                    feature_min = np.minimum(feature_min, data_numpy.min(axis=0))
                    feature_max = np.maximum(feature_max, data_numpy.max(axis=0))
                
                # all_data_chunks.append(data_numpy)
                # all_labels_chunks.append(labels_numpy)
                chunk_lengths.append(len(data_numpy))
                np.savez(
                    chunk_path,
                    data=data_numpy,
                    labels=labels_numpy,
                    aux=aux_numpy,
                )
                logging.info(
                    f"Saved chunk {chunk_idx} to {chunk_path}: "
                    f"{len(data_numpy):,} samples, "
                    f"{data_numpy.nbytes / 1024**2:.1f} MB"
                )
                np.save(chunk_dir / "chunk_sizes.npy", np.array(chunk_lengths))

                chunk_idx += 1
        logging.info("Finished loop")
        # if not all_data_chunks:
        #     raise ValueError("No valid data found in any file!")
        # torch.save(all_data_chunks, "/ceph/cms/store/user/jachisma/trackingNtuples_offline_17_0_0_pre2/data/all_data_chunks.pt")
        # logging.info("saved data chunks")
        # torch.save(all_labels_chunks, "/ceph/cms/store/user/jachisma/trackingNtuples_offline_17_0_0_pre2/data/all_labels_chunks.pt")
        # logging.info("saved label chunks")
        chunk_sizes = np.load(chunk_dir / "chunk_sizes.npy")
        total_samples = sum(chunk_lengths)
        n_features = len(TRACK_FEATURES)
        logging.info(
            f"Creating memmap: "
            f"{total_samples:,} samples × {n_features} features"
        )

        data_memmap = np.memmap(
            chunk_dir / "data.dat",
            dtype=np.float32,
            mode="w+",
            shape=(total_samples, n_features),
        )

        labels_memmap = np.memmap(
            chunk_dir / "labels.dat",
            dtype=np.float32,
            mode="w+",
            shape=(total_samples,),
        )

        aux_memmap = np.memmap(
            chunk_dir / "aux.dat",
            dtype=np.float32,
            mode="w+",
            shape=(total_samples, len(AUX_BRANCHES)),
        )

        # logging.info("Concatenating data...")
        # # data = np.concatenate(all_data_chunks, axis=0)
        # # logging.info("Concatenated data") 
        # # labels = np.concatenate(all_labels_chunks, axis=0)
        # # logging.info("Concatenated labels")
        # # total_samples = len(data)
        # data = np.empty((total_samples, n_features), dtype=np.float32)
        # labels = np.empty(total_samples, dtype=np.float32)
        # offset = 0
        # for data_chunk, label_chunk in zip(all_data_chunks, all_labels_chunks):
        #     n = len(data_chunk)

        #     data[offset:offset+n] = data_chunk
        #     labels[offset:offset+n] = label_chunk

        #     offset += n
        offset = 0
        logging.info("Starting writing chunks")
        for i in range(len(chunk_sizes)):
            chunk_path = chunk_dir / f"chunk_{i:06d}.npz"

            with np.load(chunk_path) as chunk:
                data_chunk = chunk["data"]
                labels_chunk = chunk["labels"]
                aux_chunk = chunk["aux"]

                n = len(data_chunk)

                logging.info(
                    f"Writing chunk {i}: "
                    f"{n:,} samples, "
                    f"{data_chunk.nbytes / 1024**2:.1f} MB "
                    f"at offset {offset:,}"
                )

                data_memmap[offset:offset+n] = data_chunk
                labels_memmap[offset:offset+n] = labels_chunk
                aux_memmap[offset:offset+n] = aux_chunk

                offset += n
        data_memmap.flush()
        labels_memmap.flush()
        aux_memmap.flush()

        del data_memmap
        del labels_memmap
        del aux_memmap

        logging.info(f"Total samples: {total_samples:,} ({n_real:,} real, {n_fake:,} fake)")
        data = np.memmap(
            chunk_dir / "data.dat",
            dtype=np.float32,
            mode="r",
            shape=(total_samples, n_features),
        )

        labels = np.memmap(
            chunk_dir / "labels.dat",
            dtype=np.float32,
            mode="r",
            shape=(total_samples,),
        )
        aux = np.memmap(
            chunk_dir / "aux.dat",
            dtype=np.float32,
            mode="r",
            shape=(total_samples, len(AUX_BRANCHES)),
        )
        logging.info("Convert to tensors")
        data_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).float()
        aux_tensor = torch.from_numpy(aux).float()
        
        torch.save(data_tensor, self.data_file)
        torch.save(labels_tensor, self.labels_file)
        torch.save(aux_tensor, self.aux_file)
        logging.info(f"Saved data to {self.data_file}, {self.labels_file} and {self.aux_file}")
        
        metadata = {
            'total_samples': int(total_samples),
            'n_features': int(n_features),
            'aux_branches': AUX_BRANCHES,
            'n_real': int(n_real),
            'n_fake': int(n_fake),
            'pos_weight': float(n_fake / n_real) if n_real > 0 else 1.0,
            'feature_min': feature_min.tolist(),
            'feature_max': feature_max.tolist()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def get_dataset_statistics(self):
        if isinstance(self._metadata, dict):
            meta = self._metadata
        else:
            meta = self._metadata
        
        return {
            'total_samples': int(meta['total_samples']),
            'n_real': int(meta['n_real']),
            'n_fake': int(meta['n_fake']),
            'pos_weight': float(meta['pos_weight']),
            'feature_min': torch.tensor(meta['feature_min'], dtype=torch.float32),
            'feature_max': torch.tensor(meta['feature_max'], dtype=torch.float32)
        }
    
    def __len__(self):
        return int(self._metadata['total_samples'])
    
    def __getitem__(self, idx):
        data = self._data[idx]
        label = self._labels[idx].unsqueeze(0)
        
        sample = (data, label)
        if self._weights is not None:
            sample = (data, label, self._weights[idx].unsqueeze(0))
        if self.transform_obj:
            sample = self.transform_obj(sample)
        return sample
    
    def __del__(self):
        if self._data is not None:
            del self._data
        if self._labels is not None:
            del self._labels
        if self._weights is not None:
            del self._weights