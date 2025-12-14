import logging
import os
import json
import hashlib

from pathlib import Path
from glob import glob
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import Dataset

import uproot
import awkward as ak
import numpy as np

from tqdm import tqdm

from utils import get_feature_names

TRACK_FEATURES = get_feature_names()

def process_root_file(file_path):
    try:
        with uproot.open(file_path + ":trackingNtuple/tree") as tree:
            arrays = tree.arrays(TRACK_FEATURES + ["trk_simTrkIdx"], library="ak")
            
            features = {col: arrays[col] for col in TRACK_FEATURES}
            labels = ak.fill_none(ak.firsts(ak.flatten(arrays["trk_simTrkIdx"], axis=1)), -1) != -1
            
            return features, labels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

class TrackDataset(Dataset):
    def __init__(self, input_files, transform=None, data_dir=None, **kwargs):
        self.input_files_pattern = input_files
        self.transform_obj = transform

        self._data = None
        self._labels = None
        self._metadata = None
        
        if data_dir is None:
            logging.warning("data_dir not specified, using default './data'")
            data_dir = Path(__file__).parent.parent / 'data'
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir

        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_list = sorted(glob(input_files) if isinstance(input_files, str) else input_files)
        if len(file_list) == 0:
            raise ValueError(f"No files found matching pattern: {input_files}")
        
        logging.info(f"Found {len(file_list)} ROOT files")
        
        data_key = self._generate_data_key(file_list)
        self.data_key = data_key
        self.data_prefix = data_dir / f'dataset_{data_key}'

        self.data_file = self.data_prefix.with_suffix('.data.pt')
        self.labels_file = self.data_prefix.with_suffix('.labels.pt')
        self.metadata_file = self.data_prefix.with_suffix('.meta.json')

        if self.data_file.exists() and self.labels_file.exists() and self.metadata_file.exists():
            logging.info(f"Loading from data: {self.data_prefix.name}")
            self._load_data()
        else:
            logging.info(f"Creating data: {self.data_prefix.name}")
            self._create_data(file_list)
            self._load_data()
        
        self._feature_stats = None
        if transform is not None and isinstance(transform, type):
            stats = self.get_dataset_statistics()
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
        file_info = '|'.join([f"{f}:{os.path.getmtime(f)}" for f in file_list if os.path.exists(f)])
        feature_info = '|'.join(TRACK_FEATURES)
        hash_key = hashlib.md5((file_info + feature_info).encode()).hexdigest()[:16]
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
    
    def _create_data(self, file_list):
        logging.info(f"Processing ROOT files in parallel using {cpu_count() // 4} workers...")
        all_data_chunks = []
        all_labels_chunks = []
        
        n_real = 0
        n_fake = 0
        n_features = len(TRACK_FEATURES)
        feature_min = None
        feature_max = None
        
        with Pool(processes=cpu_count()//4) as pool:
            for features, labels in tqdm(pool.imap_unordered(process_root_file, file_list), 
                                        total=len(file_list), desc="Processing ROOT files"):
                if features is None:
                    continue
                
                data_numpy = np.column_stack([
                    ak.to_numpy(ak.flatten(features[col])) 
                    for col in TRACK_FEATURES
                ]).astype('float32')
                labels_numpy = labels.to_numpy().astype('float32')
                
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
                
                all_data_chunks.append(data_numpy)
                all_labels_chunks.append(labels_numpy)
        
        if not all_data_chunks:
            raise ValueError("No valid data found in any file!")
        
        logging.info("Concatenating data...")
        data = np.concatenate(all_data_chunks, axis=0)
        labels = np.concatenate(all_labels_chunks, axis=0)
        total_samples = len(data)
        
        logging.info(f"Total samples: {total_samples:,} ({n_real:,} real, {n_fake:,} fake)")
        
        data_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        torch.save(data_tensor, self.data_file)
        torch.save(labels_tensor, self.labels_file)
        logging.info(f"Saved data to {self.data_file} and {self.labels_file}")
        
        metadata = {
            'total_samples': int(total_samples),
            'n_features': int(n_features),
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
        if self.transform_obj:
            sample = self.transform_obj(sample)
        return sample
    
    def __del__(self):
        if self._data is not None:
            del self._data
        if self._labels is not None:
            del self._labels