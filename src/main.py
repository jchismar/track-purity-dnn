import json
import logging
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from model import TrackPurityDNN
from dataset import TrackDataset, find_dataset_metadata
from trainer import LightningModel, create_trainer
from utils import MinMaxScaler, stratified_split, get_feature_names, save_jit_inference_model
from analysis.performance import (
    compute_metrics, plot_roc_curve, plot_precision_recall_curve,
    plot_confusion_matrix, plot_prediction_distribution,
    plot_calibration, plot_loss
)
from analysis.correlation import correlation_heatmap
from analysis.pca import pca_analysis
from analysis.robustness import robustness_testing
from analysis.shap_analysis import shap_analysis

FEATURE_NAMES = get_feature_names()

def load_config(config_path='../config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_data(config):
    data_config = config['data']
    
    logging.info("Preparing the dataset")
    dataset = TrackDataset(
        data_config['input_files'],
        transform=MinMaxScaler,
        data_dir=data_config.get('data_dir', None),
    )


def setup_model_and_data(config):
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']

    # Optional per-sample upweighting of high-pT, displaced tracks (training.upweight in config)
    upweight_config = training_config.get('upweight') or {}
    upweight = upweight_config if upweight_config.get('enabled', False) else None
    
    logging.info("Preparing the dataset")
    dataset = TrackDataset(
        data_config['input_files'],
        transform=MinMaxScaler,
        data_dir=data_config.get('data_dir', None),
        upweight=upweight,
    )

    stats = dataset.get_dataset_statistics()
    
    logging.info("Initializing the model")
    model = TrackPurityDNN(
        input_dim=len(FEATURE_NAMES),
        hidden_dims=model_config['hidden_dims'],
        residual_dim=model_config['residual_dim'],
        dropout=model_config['dropout'],
        n_res_blocks=model_config['n_res_blocks']
    )

    total_size = len(dataset)
    train_size = int(total_size * data_config['train_split'])
    val_size = int(total_size * data_config['val_split'])
    test_size = total_size - train_size - val_size

    logging.info(f"Dataset split into train: {train_size}, val: {val_size}, test: {test_size}")
    
    train_dataset, val_dataset, test_dataset = stratified_split(
        dataset, 
        data_config['train_split'],
        data_config['val_split'],
        data_config['test_split'],
        random_seed=data_config.get('random_seed', 42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        num_workers=training_config.get('num_workers', 4),
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )
    
    return {
        'model': model,
        'dataset': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'stats': stats,
        'sizes': {'train': train_size, 'val': val_size, 'test': test_size}
    }

def train(config_path='config.yaml', checkpoint_path=None):
    logging.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    model_config = config['model']
    training_config = config['training']
    callback_config = config['callbacks']
    logger_config = config['logger']
    trainer_config = config['trainer']
    
    setup = setup_model_and_data(config)
    model = setup['model']
    train_loader = setup['train_loader']
    val_loader = setup['val_loader']
    test_loader = setup['test_loader']
    stats = setup['stats']
    sizes = setup['sizes']
    
    n_real, n_fake, pos_weight = stats['n_real'], stats['n_fake'], stats['pos_weight']
    
    logging.info(f"Number of real samples: {n_real}, Number of fake samples: {n_fake}, Pos weight: {pos_weight}")
    
    if checkpoint_path is not None:
        logging.info(f"Loading model weights from checkpoint: {checkpoint_path}")
        lightning_model = LightningModel.load_from_checkpoint(
            checkpoint_path,
            model=model,
            learning_rate=training_config['learning_rate'],
            scheduler_config=training_config.get('scheduler'),
            loss_fn=training_config.get('loss_fn'),
            pos_weight=pos_weight
        )
    else:
        lightning_model = LightningModel(
            model, 
            learning_rate=training_config['learning_rate'],
            scheduler_config=training_config.get('scheduler'),
            loss_fn=training_config.get('loss_fn'),
            pos_weight=pos_weight,
        )
    
    trainer = create_trainer(
        training_config=training_config,
        callback_config=callback_config,
        logger_config=logger_config,
        trainer_config=trainer_config
    )

    trainer.logger.log_hyperparams({
        'input_dim': len(FEATURE_NAMES),
        'hidden_dims': model_config['hidden_dims'],
        'dropout': model_config['dropout'],
        'learning_rate': training_config['learning_rate'],
        'batch_size': training_config['batch_size'],
        'max_epochs': training_config['max_epochs'],
        'train_split': config['data']['train_split'],
        'val_split': config['data']['val_split'],
        'random_seed': config['data']['random_seed'],
        'train_size': sizes['train'],
        'val_size': sizes['val'],
        'test_size': sizes['test'],
        'scheduler_type': training_config.get('scheduler', {}).get('type', 'None'),
        'optimizer': 'Adam',
    })

    logging.info("Starting training")
    trainer.fit(lightning_model, train_loader, val_loader)
    
    return trainer.checkpoint_callback.best_model_path

def run_inference(lightning_model, loader, device):
    """Run the model over a loader. Returns (labels, logits, scaled features) as numpy arrays."""
    all_predictions = []
    all_labels = []
    all_features = []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            y_pred = lightning_model(x)
            all_predictions.append(y_pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_features.append(x.cpu().numpy())
    return (np.concatenate(all_labels).flatten(),
            np.concatenate(all_predictions).flatten(),
            np.concatenate(all_features))

def _unscale(X, idx, feature_min, feature_max):
    return X[:, idx] * (feature_max[idx] - feature_min[idx] + 1e-8) + feature_min[idx]

def _performance_plots(y_true, y_pred_logits, out_dir, target_recall):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, threshold = compute_metrics(y_true, y_pred_logits, out_dir, threshold=None, target_recall=target_recall)
    plot_roc_curve(y_true, y_pred_logits, out_dir, highlight_threshold=threshold, target_recall=target_recall)
    plot_precision_recall_curve(y_true, y_pred_logits, out_dir, highlight_threshold=threshold, target_recall=target_recall)
    plot_confusion_matrix(y_true, y_pred_logits, out_dir, threshold=threshold, target_recall=target_recall)
    plot_prediction_distribution(y_true, y_pred_logits, out_dir, threshold=threshold, target_recall=target_recall)
    plot_calibration(y_true, y_pred_logits, out_dir)
    return metrics, threshold

def run_performance_suite(y_true, y_pred_logits, X, feature_min, feature_max, out_root, target_recall):
    """Metrics + plots for all tracks, prompt (|dxy|<=0.5), displaced (|dxy|>0.5) and nPixel>2,
    written to results_*_{target_recall} folders under out_root.
    feature_min/max must be the ones used to scale X."""
    out_root = Path(out_root)

    logging.info("Filtering by dxy")
    disp_mask = np.abs(_unscale(X, FEATURE_NAMES.index("trk_dxy"), feature_min, feature_max)) > 0.5
    logging.info("Filtering by n pixel")
    npix_mask = np.abs(_unscale(X, FEATURE_NAMES.index("trk_nPixel"), feature_min, feature_max)) > 2

    logging.info("Starting performance analysis")
    logging.info("  >> all tracks")
    _performance_plots(y_true, y_pred_logits, out_root / f"results_{target_recall}", target_recall)
    logging.info("  >> prompt tracks")
    _performance_plots(y_true[~disp_mask], y_pred_logits[~disp_mask], out_root / f"results_filtered_prompt_{target_recall}", target_recall)
    logging.info("  >> displaced tracks")
    _performance_plots(y_true[disp_mask], y_pred_logits[disp_mask], out_root / f"results_filtered_disp_{target_recall}", target_recall)
    logging.info("  >> nPixels > 2")
    _performance_plots(y_true[npix_mask], y_pred_logits[npix_mask], out_root / f"results_nPix_gt1_{target_recall}", target_recall)

def resolve_checkpoint(config, checkpoint_path):
    if checkpoint_path is not None:
        return checkpoint_path
    log_dir = Path(config['logger'].get('save_dir', 'output/lightning_logs/'))
    checkpoint_files = sorted(log_dir.glob('*/*/checkpoints/best*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")
    return str(checkpoint_files[-1])

def analyze(config_path='config.yaml', checkpoint_path=None, saveModel=True):
    logging.info("Starting analysis")
    config = load_config(config_path)
    training_config = config['training']
    logger_config = config['logger']
    analysis_config = config['analysis']
    # Setup model and data
    setup = setup_model_and_data(config)
    model = setup['model']
    test_loader = setup['test_loader']
    stats = setup['stats']
    pos_weight = stats['pos_weight']
    
    checkpoint_path = resolve_checkpoint(config, checkpoint_path)
    
    logging.info(f"Loading best model from {checkpoint_path} for evaluation")
    
    best_model = LightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        learning_rate=training_config['learning_rate'],
        scheduler_config=training_config.get('scheduler'),
        loss_fn=training_config.get('loss_fn'),
        pos_weight=pos_weight
    )
    best_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model.to(device)

    target_recall = analysis_config.get('target_recall', 0.999) 
    checkpoint_dir = Path(checkpoint_path).parent.parent
    results_dir = checkpoint_dir / f"results_{target_recall}"
    results_dir.mkdir(exist_ok=True)
    
    logging.info(f"Results will be saved to: {results_dir}")
    logging.info(f"Evaluating model on test set using device: {device}")
    
    y_true, y_pred_logits, X_test = run_inference(best_model, test_loader, device)
    run_performance_suite(y_true, y_pred_logits, X_test,
                          stats['feature_min'].numpy(), stats['feature_max'].numpy(),
                          checkpoint_dir, target_recall)

    logging.info("Plotting loss curve")
    log_dir = Path(checkpoint_path).parent.parent
    plot_loss(log_dir, results_dir)
    
    class ModelWrapper:
        def __init__(self, lightning_model):
            self.model = lightning_model
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        def predict(self, X, batch_size=32768):
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                    pred = self.model(batch).cpu().numpy()
                    predictions.append(pred)
            return np.concatenate(predictions).flatten()
    
    model_wrapper = ModelWrapper(best_model)
    # robustness_testing(model_wrapper, X_test, y_true, results_dir, 
    #                     noise_levels=analysis_config.get('robustness_noise_levels', [0.01, 0.05, 0.1]), 
    #                     n_samples=analysis_config.get('robustness_n_samples', 5000), 
    #                     batch_size=analysis_config.get('robustness_batch_size', 32768))
    
    # shap_analysis(model_wrapper, X_test, FEATURE_NAMES, results_dir, 
    #                     max_samples=analysis_config.get('shap_max_samples', 1000), 
    #                     batch_size=analysis_config.get('shap_batch_size', 32768))
    if saveModel:
        logging.info("Saving model")
        best_model.to('cpu')
        jit_inference_path = Path(checkpoint_path).parent.parent / "best_model_inference_jit.pt"
        save_jit_inference_model(best_model.model, feature_min=stats['feature_min'], feature_max=stats['feature_max'], save_path=jit_inference_path, input_dim=len(FEATURE_NAMES))

        logging.info(f"Saved JIT inference model (with built-in normalization) to {jit_inference_path}")
    logging.info("Done!")

def load_training_feature_stats(config, checkpoint_path, train_meta=None):
    """Find the feature min/max the model was trained with.
    Order: explicit --trainMeta file, the training dataset's cached .meta.json, then the
    min/max stored in best_model_inference_jit.pt next to the checkpoint."""
    data_config = config['data']
    if train_meta is None:
        train_meta = find_dataset_metadata(data_config['input_files'], data_config.get('data_dir'))
    if train_meta is not None:
        logging.info(f"Using training feature min/max from {train_meta}")
        with open(train_meta) as f:
            meta = json.load(f)
        return (torch.tensor(meta['feature_min'], dtype=torch.float32),
                torch.tensor(meta['feature_max'], dtype=torch.float32), str(train_meta))

    jit_path = Path(checkpoint_path).parent.parent / "best_model_inference_jit.pt"
    if jit_path.exists():
        logging.warning(f"Training dataset metadata not found; using feature min/max stored in {jit_path}")
        jit_model = torch.jit.load(str(jit_path), map_location='cpu')
        return (jit_model.feature_min.clone().float(),
                jit_model.feature_max.clone().float(), str(jit_path))

    raise FileNotFoundError(
        "Could not find the training feature min/max: no cached .meta.json for data.input_files "
        f"in {data_config.get('data_dir')} and no {jit_path}. Pass it with --trainMeta <file.meta.json>."
    )

def evaluate(config_path='config.yaml', checkpoint_path=None, eval_files=None, eval_name=None, train_meta=None):
    """Evaluate a trained model on samples it was not trained on.
    Uses every track in eval_files (no split), scales with the training min/max,
    writes to <version_dir>/eval_<eval_name>/ and never re-saves the JIT model."""
    config = load_config(config_path)
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    analysis_config = config['analysis']

    eval_files = eval_files or data_config.get('eval_input_files')
    if not eval_files:
        raise ValueError("No evaluation files given: use --evalFiles or set data.eval_input_files in the config")
    if isinstance(eval_files, (list, tuple)) and len(eval_files) == 1:
        eval_files = eval_files[0]
    if eval_name is None:
        if isinstance(eval_files, str):
            eval_name = Path(eval_files).parent.name
        else:
            eval_name = 'custom'
            logging.warning("Several eval inputs given; writing to eval_custom (set --evalName to choose a name)")

    checkpoint_path = resolve_checkpoint(config, checkpoint_path)
    checkpoint_dir = Path(checkpoint_path).parent.parent
    out_root = checkpoint_dir / f"eval_{eval_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluating {checkpoint_path} on {eval_files}")
    logging.info(f"Results will be saved to: {out_root}")

    feature_min, feature_max, stats_source = load_training_feature_stats(config, checkpoint_path, train_meta)

    dataset = TrackDataset(
        eval_files,
        transform=MinMaxScaler,
        data_dir=data_config.get('eval_data_dir', data_config.get('data_dir')),
        feature_min=feature_min,
        feature_max=feature_max,
    )
    eval_stats = dataset.get_dataset_statistics()
    logging.info(f"Eval sample: {eval_stats['total_samples']:,} tracks "
                 f"({eval_stats['n_real']:,} real, {eval_stats['n_fake']:,} fake)")

    # Tracks outside the training range get scaled outside [0, 1]; report how many.
    out_of_range = {}
    for i, name in enumerate(FEATURE_NAMES):
        if eval_stats['feature_min'][i] < feature_min[i] or eval_stats['feature_max'][i] > feature_max[i]:
            out_of_range[name] = {
                'eval_min': float(eval_stats['feature_min'][i]), 'eval_max': float(eval_stats['feature_max'][i]),
                'train_min': float(feature_min[i]), 'train_max': float(feature_max[i]),
            }
    if out_of_range:
        logging.warning(f"{len(out_of_range)} feature(s) have values outside the training range: "
                        f"{', '.join(out_of_range)} (details in eval_info.json)")

    loader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True,
    )

    model = TrackPurityDNN(
        input_dim=len(FEATURE_NAMES),
        hidden_dims=model_config['hidden_dims'],
        residual_dim=model_config['residual_dim'],
        dropout=model_config['dropout'],
        n_res_blocks=model_config['n_res_blocks']
    )
    # Other hyperparameters (pos_weight etc.) come from the checkpoint; they only affect the loss.
    best_model = LightningModel.load_from_checkpoint(checkpoint_path, model=model)
    best_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model.to(device)

    logging.info(f"Running inference on all {len(dataset):,} tracks using device: {device}")
    y_true, y_pred_logits, X = run_inference(best_model, loader, device)

    target_recall = analysis_config.get('target_recall', 0.999)
    run_performance_suite(y_true, y_pred_logits, X, feature_min.numpy(), feature_max.numpy(),
                          out_root, target_recall)

    with open(out_root / "eval_info.json", 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'eval_files': eval_files,
            'training_feature_stats_from': stats_source,
            'n_tracks': int(eval_stats['total_samples']),
            'n_real': int(eval_stats['n_real']),
            'n_fake': int(eval_stats['n_fake']),
            'target_recall': target_recall,
            'features_outside_training_range': out_of_range,
        }, f, indent=2)
    logging.info("Done!")

if __name__ == '__main__':    
    parser = ArgumentParser(description='Train track purity DNN')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--analyze', action='store_true',
                        help='Only run analysis on the latest trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint file for analysis or to continue training')
    parser.add_argument('--noSaveModel', action='store_false', default=True,
                        help='Do not save jit model (ie. if rerunning analysis)')
    # parser.add_argument('--filterDxy', action='store_true',
    #                     help='Get WPs for high dxy tracks (|dxy|>0.5)')
    parser.add_argument('--setupData', action='store_true',
                        help='only run setup_data')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model on other samples (all tracks, training scaling, no JIT re-save)')
    parser.add_argument('--evalFiles', type=str, nargs='+', default=None,
                        help='One or more ROOT files or quoted globs to evaluate on (overrides data.eval_input_files)')
    parser.add_argument('--evalName', type=str, default=None,
                        help='Name for the output folder eval_<name> (default: parent dir name of --evalFiles)')
    parser.add_argument('--trainMeta', type=str, default=None,
                        help='Path to the training dataset .meta.json (if it cannot be found automatically)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.evaluate:
        logging.info("Evaluate mode selected. Evaluating on new samples.")
        evaluate(config_path=args.config, checkpoint_path=args.checkpoint, eval_files=args.evalFiles,
                 eval_name=args.evalName, train_meta=args.trainMeta)
    elif args.analyze:
        logging.info("Analysis mode selected. Running analysis only.")
        if not args.noSaveModel:
            logging.info("No model will be saved")
        analyze(config_path=args.config, checkpoint_path=args.checkpoint, saveModel=args.noSaveModel)
    elif args.setupData:
        logging.info("Setup mode selected. Setting up dataset only")
        config = load_config(args.config)
        setup_data(config)
    else:
        best_model_path = train(config_path=args.config, checkpoint_path=args.checkpoint)
        analyze(config_path=args.config, checkpoint_path=best_model_path, saveModel=True)
