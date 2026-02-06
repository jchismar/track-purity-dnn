import logging
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from model import TrackPurityDNN
from dataset import TrackDataset
from trainer import LightningModel, create_trainer
from utils import MinMaxScaler, stratified_split, get_feature_names, save_jit_inference_model
from analysis.performance import (
    compute_metrics, plot_roc_curve, plot_precision_recall_curve,
    plot_confusion_matrix, plot_prediction_distribution,
    plot_calibration
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

def setup_model_and_data(config):
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    logging.info("Preparing the dataset")
    dataset = TrackDataset(
        data_config['input_files'],
        transform=MinMaxScaler,
        data_dir=data_config.get('data_dir', None),
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

def analyze(config_path='config.yaml', checkpoint_path=None):
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
    
    if checkpoint_path is None:
        log_dir = Path(logger_config.get('save_dir', 'output/lightning_logs/'))
        checkpoint_files = sorted(log_dir.glob('*/*/checkpoints/best*.ckpt'))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in {log_dir}")
        checkpoint_path = str(checkpoint_files[-1])
    
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
    
    checkpoint_dir = Path(checkpoint_path).parent.parent
    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    logging.info(f"Results will be saved to: {results_dir}")
    logging.info(f"Evaluating model on test set using device: {device}")
    
    all_predictions = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, w = batch
            x = x.to(device)
            y_pred = best_model(x)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_features.append(x.cpu().numpy())
    
    y_pred_logits = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    X_test = np.concatenate(all_features)
    
    logging.info("Starting performance analysis")
    target_recall = analysis_config.get('target_recall', 0.999)
    metrics, threshold = compute_metrics(y_true, y_pred_logits, results_dir, threshold=None, target_recall=target_recall)
    
    plot_roc_curve(y_true, y_pred_logits, results_dir, highlight_threshold=threshold, target_recall=target_recall)
    plot_precision_recall_curve(y_true, y_pred_logits, results_dir, highlight_threshold=threshold, target_recall=target_recall)
    plot_confusion_matrix(y_true, y_pred_logits, results_dir, threshold=threshold, target_recall=target_recall)
    plot_prediction_distribution(y_true, y_pred_logits, results_dir, threshold=threshold, target_recall=target_recall)
    plot_calibration(y_true, y_pred_logits, results_dir)
    
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
    robustness_testing(model_wrapper, X_test, y_true, results_dir, 
                        noise_levels=analysis_config.get('robustness_noise_levels', [0.01, 0.05, 0.1]), 
                        n_samples=analysis_config.get('robustness_n_samples', 5000), 
                        batch_size=analysis_config.get('robustness_batch_size', 32768))
    
    shap_analysis(model_wrapper, X_test, FEATURE_NAMES, results_dir, 
                    max_samples=analysis_config.get('shap_max_samples', 1000), 
                    batch_size=analysis_config.get('shap_batch_size', 32768))
    
    best_model.to('cpu')
    jit_inference_path = Path(checkpoint_path).parent.parent / "best_model_inference_jit.pt"
    save_jit_inference_model(best_model.model, feature_min=stats['feature_min'], feature_max=stats['feature_max'], save_path=jit_inference_path, input_dim=len(FEATURE_NAMES))

    logging.info(f"Saved JIT inference model (with built-in normalization) to {jit_inference_path}")
    logging.info("Done!")

if __name__ == '__main__':    
    parser = ArgumentParser(description='Train track purity DNN')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--analyze', action='store_true',
                        help='Only run analysis on the latest trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint file for analysis or to continue training')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.analyze:
        logging.info("Analysis mode selected. Running analysis only.")
        analyze(config_path=args.config, checkpoint_path=args.checkpoint)
    else:
        best_model_path = train(config_path=args.config, checkpoint_path=args.checkpoint)
        analyze(config_path=args.config, checkpoint_path=best_model_path)
