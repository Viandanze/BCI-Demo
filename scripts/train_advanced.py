#!/usr/bin/env python3
"""
Advanced BCI Models Training Script
Train ShallowConvNet, EEG-Conformer, or TCN on PhysioNet Motor Imagery dataset

Usage:
    python train_advanced.py --model shallowconvnet --subjects 1 2 3
    python train_advanced.py --model conformer --epochs 150 --batch_size 32
    python train_advanced.py --model tcn --lr 0.0005 --weight_decay 0.001

Models:
    - shallowconvnet: Wide shallow CNN (Schirrmeister 2017)
    - conformer: EEG-Conformer with attention (Song 2023)
    - tcn: Temporal Convolutional Network (Bai 2018)
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.advanced import ShallowConvNet, EEGConformer, TCN, create_model
from src.training.trainer import Trainer, TrainingConfig, set_seed, prepare_data_loaders
from src.training.augment import EEGAugmentor, AugmentationConfig
from src.data.loader import load_physionet_data, get_subject_data
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.evaluation.metrics import compute_all_metrics, print_results
from src.visualization.visualizer import BCIVisualizer
from src.utils.config import load_config, DictConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Advanced BCI Models on Motor Imagery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['shallowconvnet', 'conformer', 'tcn'],
        help='Model architecture to train',
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/',
        help='Path to PhysioNet dataset',
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=[1],
        help='Subject IDs to train on',
    )
    parser.add_argument(
        '--runs',
        type=int,
        nargs='+',
        default=[4, 5, 6],
        help='PhysioNet run numbers (4,5,6 = motor imagery)',
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--n_filters',
        type=int,
        default=None,
        help='Number of filters/channels (model-specific default)',
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=64,
        help='Embedding dimension for Conformer',
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads for Conformer',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=None,
        help='Number of layers (Conformer/TCN)',
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=None,
        help='Convolution kernel size',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout rate',
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay (L2 regularization)',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience',
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help='Label smoothing factor',
    )
    
    # Augmentation arguments
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation',
    )
    parser.add_argument(
        '--aug_prob',
        type=float,
        default=0.5,
        help='Base augmentation probability',
    )
    
    # Experiment arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/',
        help='Output directory',
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for saving',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use',
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Cross-validation folds (1 = no CV)',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots after training',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='YAML config file to load',
    )
    
    return parser.parse_args()


def get_model_defaults(model_type: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for each model type.
    
    Args:
        model_type: Model name
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'shallowconvnet': {
            'n_filters': 40,
            'filter_time_length': 25,
            'pool_time_length': 75,
            'pool_time_stride': 15,
            'dropout': 0.5,
        },
        'conformer': {
            'embed_dim': 64,
            'num_heads': 8,
            'num_layers': 3,
            'patch_size': 25,
            'kernel_size': 3,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        'tcn': {
            'n_filters': 32,
            'kernel_size': 7,
            'n_layers': 4,
            'dropout': 0.5,
        },
    }
    return defaults.get(model_type, {})


def load_data(
    data_path: str,
    subjects: List[int],
    runs: List[int],
) -> Dict[int, tuple]:
    """
    Load and preprocess data for multiple subjects.
    
    Returns:
        Dictionary mapping subject_id -> (X, y) tuple
    """
    logger.info(f"Loading data for subjects: {subjects}")
    
    try:
        raw_dict = load_physionet_data(
            data_path=data_path,
            subjects=subjects,
            runs=runs,
        )
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Generating synthetic data for testing")
        from src.data.loader import _create_sample_data
        raw_dict = {s: _create_sample_data(s, runs) for s in subjects}
    
    # Preprocessing config
    preproc_config = PreprocessingConfig(
        bandpass_low=4,
        bandpass_high=38,
        tmin=-1.0,
        tmax=4.0,
        baseline=(-1.0, 0.0),
        normalize=True,
        resample_freq=128,
    )
    
    pipeline = PreprocessingPipeline(preproc_config)
    
    data_dict = {}
    for subject_id, raw in raw_dict.items():
        logger.info(f"Processing subject {subject_id}...")
        
        try:
            epochs = pipeline.process_raw(raw)
            X = epochs.get_data()
            y = epochs.events[:, -1]
            
            # Convert to 0-indexed labels
            unique_labels = sorted(np.unique(y))
            label_map = {l: i for i, l in enumerate(unique_labels)}
            y = np.array([label_map[l] for l in y])
            
            data_dict[subject_id] = (X, y)
            logger.info(f"  Loaded {len(X)} epochs, shape {X.shape}, "
                       f"classes: {len(unique_labels)}")
            
        except Exception as e:
            logger.error(f"  Failed to process subject {subject_id}: {e}")
            continue
    
    return data_dict


def create_model_instance(
    model_type: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    args,
) -> nn.Module:
    """
    Create model instance with appropriate hyperparameters.
    
    Args:
        model_type: Model type string
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes
        args: Command line arguments
        
    Returns:
        Initialized model
    """
    defaults = get_model_defaults(model_type)
    
    # Build kwargs from defaults and args
    kwargs = {}
    
    if model_type == 'shallowconvnet':
        kwargs = {
            'n_filters': args.n_filters or defaults.get('n_filters', 40),
            'filter_time_length': defaults.get('filter_time_length', 25),
            'pool_time_length': defaults.get('pool_time_length', 75),
            'pool_time_stride': defaults.get('pool_time_stride', 15),
            'dropout_rate': args.dropout if args.dropout is not None else defaults.get('dropout', 0.5),
        }
        model = ShallowConvNet(n_channels, n_times, n_classes, **kwargs)
        
    elif model_type in ('conformer', 'eegconformer'):
        kwargs = {
            'embed_dim': args.embed_dim or defaults.get('embed_dim', 64),
            'num_heads': args.num_heads or defaults.get('num_heads', 8),
            'num_layers': args.num_layers or defaults.get('num_layers', 3),
            'patch_size': defaults.get('patch_size', 25),
            'kernel_size': defaults.get('kernel_size', 3),
            'mlp_ratio': defaults.get('mlp_ratio', 4.0),
            'dropout': args.dropout if args.dropout is not None else defaults.get('dropout', 0.1),
        }
        model = EEGConformer(n_channels, n_times, n_classes, **kwargs)
        
    elif model_type == 'tcn':
        kwargs = {
            'n_filters': args.n_filters or defaults.get('n_filters', 32),
            'kernel_size': args.kernel_size or defaults.get('kernel_size', 7),
            'n_layers': args.num_layers or defaults.get('n_layers', 4),
            'dropout': args.dropout if args.dropout is not None else defaults.get('dropout', 0.5),
        }
        model = TCN(n_channels, n_times, n_classes, **kwargs)
    
    logger.info(f"Created {model_type} with params: {kwargs}")
    return model


def train_single_subject(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    args,
    augmentor: Optional[EEGAugmentor] = None,
) -> Dict[str, Any]:
    """
    Train model on single subject data.
    
    Returns:
        Results dictionary with metrics and history
    """
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    
    logger.info(f"Training {model_type} on data: {X.shape}, {n_classes} classes")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=True
    )
    
    # Create model
    model = create_model_instance(
        model_type, n_channels, n_times, n_classes, args
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Training config
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        device=args.device,
        save_best_only=True,
    )
    
    # Create trainer
    trainer = Trainer(model, train_config)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        X_train, y_train,
        X_test, y_test,
        batch_size=args.batch_size,
        augmentor=augmentor,
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    trainer.model.eval()
    y_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), args.batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+args.batch_size]).unsqueeze(1)
            batch_X = batch_X.to(trainer.device)
            outputs = trainer.model(batch_X)
            _, preds = outputs.max(1)
            y_pred_list.extend(preds.cpu().numpy())
    
    y_pred = np.array(y_pred_list)
    metrics = compute_all_metrics(y_test, y_pred)
    
    return {
        'test_metrics': metrics.to_dict(),
        'history': history,
        'model_config': model.get_config(),
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
    }


def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    args,
    augmentor: Optional[EEGAugmentor] = None,
) -> Dict[str, Any]:
    """
    Train with k-fold cross-validation.
    
    Returns:
        Results dictionary with per-fold and mean metrics
    """
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    
    kfold = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model = create_model_instance(
            model_type, n_channels, n_times, n_classes, args
        )
        
        # Training config
        train_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.patience,
            label_smoothing=args.label_smoothing,
            device=args.device,
        )
        
        trainer = Trainer(model, train_config)
        
        # Data loaders
        train_loader, val_loader = prepare_data_loaders(
            X_train, y_train,
            X_val, y_val,
            batch_size=args.batch_size,
            augmentor=augmentor,
        )
        
        # Train
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate
        trainer.model.eval()
        y_pred_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                batch_X = torch.FloatTensor(X_val[i:i+args.batch_size]).unsqueeze(1)
                batch_X = batch_X.to(trainer.device)
                outputs = trainer.model(batch_X)
                _, preds = outputs.max(1)
                y_pred_list.extend(preds.cpu().numpy())
        
        y_pred = np.array(y_pred_list)
        fold_acc = accuracy_score(y_val, y_pred)
        fold_results.append(fold_acc)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
    
    # Aggregate results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    logger.info(f"\nCross-validation Results:")
    logger.info(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Compute overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    metrics = compute_all_metrics(all_y_true, all_y_pred)
    
    return {
        'fold_accuracies': fold_results,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'overall_metrics': metrics.to_dict(),
        'model_config': model.get_config(),
        'y_true': all_y_true.tolist(),
        'y_pred': all_y_pred.tolist(),
    }


def generate_visualizations(
    results: Dict[str, Any],
    output_dir: Path,
    model_type: str,
) -> None:
    """
    Generate visualization plots after training.
    
    Args:
        results: Training results dictionary
        output_dir: Output directory for figures
        model_type: Model type string
    """
    logger.info("Generating visualizations...")
    
    viz = BCIVisualizer(output_dir=str(output_dir / 'figures'))
    
    # Training curves
    if 'history' in results:
        history = results['history']
        val_history = {
            'val_loss': history.get('val_loss', []),
            'val_acc': history.get('val_acc', []),
        }
        viz.plot_training_curves(
            history, val_history,
            title=f'{model_type} Training Progress',
            save_name=f'{model_type}_training_curves',
        )
    
    # Confusion matrix
    if 'y_true' in results and 'y_pred' in results:
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        labels = ['Left Hand', 'Right Hand', 'Feet', 'Rest']
        
        viz.plot_confusion_matrix(
            y_true, y_pred, labels=labels[:len(np.unique(y_true))],
            title=f'{model_type} Confusion Matrix',
            save_name=f'{model_type}_confusion_matrix',
        )
    
    logger.info(f"Visualizations saved to {output_dir / 'figures'}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_s{'-'.join(map(str, args.subjects))}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create augmentor if enabled
    augmentor = None
    if args.augment:
        aug_config = AugmentationConfig(
            enabled=True,
            probability=args.aug_prob,
            temporal_mask={'enabled': True, 'prob': 0.3},
            channel_mask={'enabled': True, 'prob': 0.2},
            gaussian_noise={'enabled': True, 'prob': 0.3, 'snr_db': 10},
            time_shift={'enabled': True, 'prob': 0.2, 'max_shift_samples': 20},
        )
        augmentor = EEGAugmentor(aug_config, sfreq=128, random_state=args.seed)
        logger.info(f"Data augmentation enabled (prob={args.aug_prob})")
    
    # Load data
    data_dict = load_data(args.data_path, args.subjects, args.runs)
    
    if len(data_dict) == 0:
        logger.error("No data loaded. Exiting.")
        return
    
    # Train on each subject
    all_results = {}
    
    for subject_id, (X, y) in data_dict.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training Subject {subject_id}")
        logger.info(f"{'=' * 50}")
        
        if args.cv_folds > 1:
            results = train_with_cross_validation(X, y, args.model, args, augmentor)
        else:
            results = train_single_subject(X, y, args.model, args, augmentor)
        
        all_results[f"subject_{subject_id}"] = results
        
        # Print results
        if 'overall_metrics' in results:
            print_results(results['overall_metrics'], title=f"Subject {subject_id} (CV)")
        elif 'test_metrics' in results:
            print_results(results['test_metrics'], title=f"Subject {subject_id}")
        
        # Generate visualizations
        if args.visualize:
            subject_dir = output_dir / f"subject_{subject_id}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            generate_visualizations(results, subject_dir, args.model)
    
    # Save all results
    save_path = output_dir / "results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {save_path}")
    
    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("Training Complete")
    logger.info(f"{'=' * 50}")
    
    if args.cv_folds > 1:
        for subject_id, results in all_results.items():
            logger.info(
                f"{subject_id}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
            )
    else:
        for subject_id, results in all_results.items():
            logger.info(
                f"{subject_id}: {results['test_metrics']['accuracy']:.4f}"
            )


if __name__ == '__main__':
    main()
