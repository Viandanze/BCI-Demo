#!/usr/bin/env python3
"""
EEGNet Training Script
Train EEGNet model on PhysioNet Motor Imagery dataset

P1-③ Improvements:
  - Class weighting (balanced) to fix prediction collapse
  - CosineAnnealingLR scheduler by default
  - Data augmentation enabled by default
  - Label smoothing (0.1) to prevent overconfident predictions
  - Model checkpoint saved to output directory

Usage:
    python train_eegnet.py --subjects 1 2 3 4 5
    python train_eegnet.py --subjects 1 2 3 --epochs 100 --no_augment
    python train_eegnet.py --config configs/default.yaml --epochs 100
"""

import os
import sys
import argparse
import logging
import json
from typing import Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eegnet import EEGNetClassifier
from src.training.trainer import (
    Trainer, TrainingConfig, set_seed, prepare_data_loaders,
    create_optimizer,
)
from src.training.augment import EEGAugmentor, AugmentationConfig
from src.data.loader import load_physionet_data, get_subject_data
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.evaluation.metrics import compute_all_metrics, print_results, save_results
from src.utils.config import load_config, merge_configs, DictConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train EEGNet on Motor Imagery BCI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Model arguments
    parser.add_argument(
        '--F1',
        type=int,
        default=8,
        help='Number of temporal filters',
    )
    parser.add_argument(
        '--D',
        type=int,
        default=2,
        help='Depth multiplier for spatial filters',
    )
    parser.add_argument(
        '--kernel_length',
        type=int,
        default=64,
        help='Temporal convolution kernel length',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate',
    )
    parser.add_argument(
        '--use_batchnorm',
        action='store_true',
        help='Use batch normalization',
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
        default=0.1,
        help='Label smoothing factor (prevents overconfident predictions)',
    )

    # P1-③ Anti-collapse arguments
    parser.add_argument(
        '--no_class_weighting',
        action='store_true',
        help='Disable class weighting (enabled by default to fix prediction collapse)',
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['none', 'cosine', 'plateau'],
        help='Learning rate scheduler (cosine annealing by default)',
    )

    # Augmentation arguments
    parser.add_argument(
        '--no_augment',
        action='store_true',
        help='Disable data augmentation (enabled by default)',
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
        '--config',
        type=str,
        help='YAML config file to load',
    )

    return parser.parse_args()


def load_data(
    data_path: str,
    subjects: list,
    runs: list,
    config: Optional[DictConfig] = None,
) -> dict:
    """
    Load and preprocess data for multiple subjects.

    Returns:
        Dictionary mapping subject_id -> (X, y) tuple
    """
    if config is None:
        config = DictConfig()

    logger.info(f"Loading data for subjects: {subjects}")

    # Load raw data
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
        bandpass_low=config.get('preprocessing.bandpass_low', 4),
        bandpass_high=config.get('preprocessing.bandpass_high', 38),
        tmin=config.get('data.tmin', -1.0),
        tmax=config.get('data.tmax', 4.0),
        baseline=(-1.0, 0.0) if config.get('data.baseline_correction', True) else None,
        normalize=config.get('preprocessing.normalize', True),
        resample_freq=config.get('data.resample_freq', 128),
    )

    pipeline = PreprocessingPipeline(preproc_config)

    data_dict = {}
    for subject_id, raw in raw_dict.items():
        logger.info(f"Processing subject {subject_id}...")

        try:
            epochs = pipeline.process_raw(raw)
            X = epochs.get_data()
            y = epochs.events[:, -1]

            # Convert to 0-indexed
            unique_labels = sorted(np.unique(y))
            label_map = {l: i for i, l in enumerate(unique_labels)}
            y = np.array([label_map[l] for l in y])

            data_dict[subject_id] = (X, y)
            logger.info(f"  Loaded {len(X)} epochs, shape {X.shape}")

            # Log class distribution
            for cls in sorted(np.unique(y)):
                count = (y == cls).sum()
                logger.info(f"    Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")

        except Exception as e:
            logger.error(f"  Failed to process subject {subject_id}: {e}")
            continue

    return data_dict


def _compute_class_weights(y_train: np.ndarray, device: str) -> Optional[torch.Tensor]:
    """Compute balanced class weights for CrossEntropyLoss."""
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None

    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.FloatTensor(weights).to(device)

    logger.info(f"Class weights: {dict(zip(classes.tolist(), [f'{w:.3f}' for w in weights]))}")
    return class_weights


def _create_scheduler(optimizer, scheduler_type: str, epochs: int):
    """Create learning rate scheduler."""
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    return None


def train_single_subject(
    X: np.ndarray,
    y: np.ndarray,
    args,
    augmentor: Optional[EEGAugmentor] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Train EEGNet on single subject data.

    Returns:
        Results dictionary
    """
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))

    logger.info(f"Training on data: {X.shape}, {n_classes} classes")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=True
    )

    # Create model
    model = EEGNetClassifier(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=args.F1,
        D=args.D,
        kernel_length=args.kernel_length,
        dropout_rate=args.dropout,
        use_batchnorm=args.use_batchnorm,
        apply_softmax=False,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # Create optimizer first (needed for scheduler)
    optimizer = create_optimizer(
        model,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # Create scheduler
    scheduler = _create_scheduler(optimizer, args.scheduler, args.epochs)

    # Create criterion with optional class weighting
    device = train_config.device if train_config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.no_class_weighting:
        class_weights = _compute_class_weights(y_train, device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Create trainer with optimizer, scheduler, and criterion
    trainer = Trainer(
        model,
        train_config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

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

    # Save model to output directory
    if output_dir:
        model_path = output_dir / 'best_model.pt'
        trainer.save_checkpoint(str(model_path))
        logger.info(f"Model saved to {model_path}")

    # Evaluate on test set
    trainer.model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1).to(trainer.device)
        outputs = trainer.model(X_test_t)
        _, y_pred = outputs.max(1)
        y_pred = y_pred.cpu().numpy()

    metrics = compute_all_metrics(y_test, y_pred)

    # Log per-class accuracy
    logger.info("Per-class accuracy:")
    for cls in sorted(np.unique(y_test)):
        mask = y_test == cls
        cls_acc = (y_pred[mask] == cls).mean() if mask.sum() > 0 else 0.0
        logger.info(f"  Class {cls}: {cls_acc:.2%} ({mask.sum()} samples)")

    return {
        'test_metrics': metrics,
        'history': history,
        'model_config': model.get_config(),
    }


def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    args,
    augmentor: Optional[EEGAugmentor] = None,
    output_dir: Optional[Path] = None,
) -> dict:
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
    best_fold_acc = 0.0
    best_fold_trainer = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create model
        model = EEGNetClassifier(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            F1=args.F1,
            D=args.D,
            kernel_length=args.kernel_length,
            dropout_rate=args.dropout,
            use_batchnorm=args.use_batchnorm,
            apply_softmax=False,
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

        # Create optimizer
        optimizer = create_optimizer(
            model,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Create scheduler
        scheduler = _create_scheduler(optimizer, args.scheduler, args.epochs)

        # Create criterion with optional class weighting
        device = train_config.device if train_config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        if not args.no_class_weighting:
            class_weights = _compute_class_weights(y_train, device)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Create trainer
        trainer = Trainer(
            model,
            train_config,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )

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
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(trainer.device)
            outputs = trainer.model(X_val_t)
            _, y_pred = outputs.max(1)
            y_pred = y_pred.cpu().numpy()

        fold_acc = accuracy_score(y_val, y_pred)
        fold_results.append(fold_acc)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")

        # Track best fold for model saving
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            best_fold_trainer = trainer

    # Save best fold model
    if output_dir and best_fold_trainer:
        model_path = output_dir / 'best_model.pt'
        best_fold_trainer.save_checkpoint(str(model_path))
        logger.info(f"Best fold model (acc={best_fold_acc:.4f}) saved to {model_path}")

    # Aggregate results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    logger.info(f"\nCross-validation Results:")
    logger.info(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Compute overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    metrics = compute_all_metrics(all_y_true, all_y_pred)

    # Log per-class accuracy
    logger.info("Overall per-class accuracy:")
    for cls in sorted(np.unique(all_y_true)):
        mask = all_y_true == cls
        cls_acc = (all_y_pred[mask] == cls).mean() if mask.sum() > 0 else 0.0
        logger.info(f"  Class {cls}: {cls_acc:.2%} ({mask.sum()} samples)")

    return {
        'fold_accuracies': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'overall_metrics': metrics,
    }


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"eegnet_s{'-'.join(map(str, args.subjects))}_{timestamp}"

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Anti-collapse measures:")
    logger.info(f"  Class weighting: {'OFF' if args.no_class_weighting else 'ON (balanced)'}")
    logger.info(f"  LR scheduler: {args.scheduler}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Data augmentation: {'OFF' if args.no_augment else 'ON'}")

    # Create augmentor (enabled by default)
    augmentor = None
    if not args.no_augment:
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
            results = train_with_cross_validation(X, y, args, augmentor, output_dir)
        else:
            results = train_single_subject(X, y, args, augmentor, output_dir)

        all_results[f"subject_{subject_id}"] = results

        # Print results
        if 'overall_metrics' in results:
            print_results(
                results['overall_metrics'],
                title=f"Subject {subject_id} (CV)"
            )
        elif 'test_metrics' in results:
            print_results(
                results['test_metrics'],
                title=f"Subject {subject_id}"
            )

    # Save results
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
                f"{subject_id}: {results['test_metrics'].accuracy:.4f}"
            )

    logger.info(f"\nModel checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
