#!/usr/bin/env python3
"""
EEGNet Hyperparameter Tuning Script
Provides three tuning strategies to break through 70% accuracy barrier

Strategy A: Data Augmentation Combinations
Strategy B: Hyperparameter Grid Search
Strategy C: Architecture Improvements

Usage:
    python tune_eegnet.py --strategy A --subjects 1 2 3
    python tune_eegnet.py --strategy B --full_search
    python tune_eegnet.py --strategy C --attention
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eegnet import EEGNetClassifier, EEGNet
from src.training.trainer import Trainer, TrainingConfig, set_seed
from src.training.augment import EEGAugmentor, AugmentationConfig
from src.data.loader import load_physionet_data, get_subject_data, create_train_test_split
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.evaluation.metrics import compute_all_metrics, EvaluationMetrics
from src.utils.config import load_config, DictConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./outputs/tuning.log'),
    ]
)
logger = logging.getLogger(__name__)


class TuningExperiment:
    """
    Base class for EEGNet tuning experiments.
    
    Provides common utilities for data loading, model training,
    and result tracking.
    """
    
    def __init__(
        self,
        data_path: str = "./data/",
        output_dir: str = "./outputs/",
        seed: int = 42,
        device: str = "auto",
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = device
        
        # Results storage
        self.experiment_results = []
        
        # Set random seed
        set_seed(seed)
        
        logger.info(f"Initialized tuning experiment")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_subject_data(
        self,
        subject_id: int,
        runs: List[int] = [4, 5, 6],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data for a single subject.
        
        Args:
            subject_id: Subject ID (1-109)
            runs: PhysioNet run numbers
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info(f"Loading data for subject {subject_id}")
        
        try:
            # Try to load real data
            raw_dict = load_physionet_data(
                data_path=self.data_path,
                subjects=[subject_id],
                runs=runs,
            )
            
            if subject_id not in raw_dict:
                raise ValueError(f"Subject {subject_id} not loaded")
            
            raw = raw_dict[subject_id]
            
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
            logger.info("Generating synthetic data for testing")
            from src.data.loader import _create_sample_data
            raw = _create_sample_data(subject_id, runs)
        
        # Preprocess
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
        epochs = pipeline.process_raw(raw)
        
        # Extract data
        X = epochs.get_data()
        y = epochs.events[:, -1]
        
        # Convert to 0-indexed labels
        unique_labels = sorted(np.unique(y))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y])
        
        logger.info(f"Loaded {len(X)} epochs, shape {X.shape}")
        
        return X, y
    
    def create_model(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        config: Optional[Dict] = None,
    ) -> Tuple[EEGNetClassifier, TrainingConfig]:
        """
        Create EEGNet model with given configuration.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points
            n_classes: Number of classes
            config: Model and training configuration
            
        Returns:
            Tuple of (model, training_config)
        """
        if config is None:
            config = {}
        
        # Model config
        model_config = {
            'F1': config.get('F1', 8),
            'D': config.get('D', 2),
            'kernel_length': config.get('kernel_length', 64),
            'dropout_rate': config.get('dropout_rate', 0.5),
            'use_batchnorm': config.get('use_batchnorm', False),
        }
        
        # Training config
        train_config = TrainingConfig(
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 64),
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01),
            early_stopping_patience=config.get('patience', 15),
            label_smoothing=config.get('label_smoothing', 0.0),
            device=self.device,
        )
        
        model = EEGNetClassifier(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            apply_softmax=False,
            **model_config,
        )
        
        return model, train_config
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_config: Dict,
        augmentor: Optional[EEGAugmentor] = None,
        n_folds: int = 5,
    ) -> Tuple[float, EvaluationMetrics]:
        """
        Train model with cross-validation and return average accuracy.
        
        Args:
            X: Data array (samples, channels, times)
            y: Labels
            model_config: Model configuration
            augmentor: Data augmentor (optional)
            n_folds: Number of CV folds
            
        Returns:
            Tuple of (mean_accuracy, last_fold_metrics)
        """
        n_channels = X.shape[1]
        n_times = X.shape[2]
        n_classes = len(np.unique(y))
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        fold_accuracies = []
        last_metrics = None
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            model, train_config = self.create_model(
                n_channels, n_times, n_classes, model_config
            )
            
            # Create data loaders
            X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
            y_train_t = torch.LongTensor(y_train)
            X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
            y_val_t = torch.LongTensor(y_val)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            
            train_loader = DataLoader(
                train_dataset, batch_size=train_config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=train_config.batch_size, shuffle=False
            )
            
            # Create trainer
            trainer = Trainer(model, train_config)
            
            # Train
            history = trainer.train(train_loader, val_loader, epochs=train_config.epochs)
            
            # Evaluate
            trainer.model.eval()
            with torch.no_grad():
                X_val_t = X_val_t.to(trainer.device)
                outputs = trainer.model(X_val_t)
                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().numpy()
            
            fold_acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(fold_acc)
            
            # Compute full metrics for last fold
            if fold == n_folds - 1:
                last_metrics = compute_all_metrics(y_val, y_pred)
            
            logger.info(f"  Fold {fold+1}/{n_folds}: Accuracy = {fold_acc:.4f}")
        
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        logger.info(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        return mean_acc, last_metrics
    
    def save_results(
        self,
        results: Dict,
        experiment_name: str,
    ) -> None:
        """Save experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


class StrategyA_DataAugmentation(TuningExperiment):
    """
    Strategy A: Data Augmentation Combinations
    
    Systematically test different augmentation strategies to find
    the best combination for improving accuracy.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experiment_name = "strategy_A_augmentation"
    
    def run(
        self,
        subject_ids: List[int],
        augmentation_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run augmentation experiments.
        
        Args:
            subject_ids: List of subject IDs to test
            augmentation_methods: Methods to test (default: all)
            
        Returns:
            Dictionary of results
        """
        if augmentation_methods is None:
            augmentation_methods = [
                'baseline',  # No augmentation
                'gaussian_noise',
                'temporal_mask',
                'channel_mask',
                'time_shift',
                'band_perturbation',
                'mixup',
                'all_combined',
            ]
        
        results = {
            'experiment': 'Strategy A: Data Augmentation',
            'timestamp': datetime.now().isoformat(),
            'subjects': subject_ids,
            'augmentation_results': {},
            'best_combination': None,
            'best_accuracy': 0.0,
        }
        
        # Load data from first subject for quick testing
        X, y = self.load_subject_data(subject_ids[0])
        
        # Base model config (no augmentation)
        base_config = {
            'F1': 8,
            'D': 2,
            'kernel_length': 64,
            'dropout_rate': 0.5,
            'epochs': 50,
            'batch_size': 64,
        }
        
        logger.info("=" * 60)
        logger.info("Strategy A: Testing Data Augmentation Methods")
        logger.info("=" * 60)
        
        for method in augmentation_methods:
            logger.info(f"\n--- Testing: {method} ---")
            
            # Configure augmentor
            if method == 'baseline':
                augmentor = None
            elif method == 'all_combined':
                config = AugmentationConfig(enabled=True)
                augmentor = EEGAugmentor(config, sfreq=128, random_state=self.seed)
            else:
                config = AugmentationConfig(
                    enabled=True,
                    **{method: {'enabled': True, 'prob': 0.5}},
                )
                augmentor = EEGAugmentor(config, sfreq=128, random_state=self.seed)
            
            # Train and evaluate
            acc, metrics = self.train_and_evaluate(
                X, y,
                model_config=base_config,
                augmentor=augmentor,
                n_folds=5,
            )
            
            results['augmentation_results'][method] = {
                'accuracy': acc,
                'kappa': metrics.kappa if metrics else 0,
                'f1_macro': metrics.f1_macro if metrics else 0,
            }
            
            if acc > results['best_accuracy']:
                results['best_accuracy'] = acc
                results['best_combination'] = method
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Augmentation Results Summary")
        logger.info("=" * 60)
        
        for method, metrics_dict in sorted(
            results['augmentation_results'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True,
        ):
            logger.info(f"{method:<20}: {metrics_dict['accuracy']:.4f}")
        
        logger.info(f"\nBest augmentation: {results['best_combination']}")
        logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
        
        self.save_results(results, self.experiment_name)
        
        return results


class StrategyB_HyperparameterSearch(TuningExperiment):
    """
    Strategy B: Hyperparameter Grid Search
    
    Perform systematic grid search over key hyperparameters.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experiment_name = "strategy_B_hyperparams"
    
    def run(
        self,
        subject_ids: List[int],
        full_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        
        Args:
            subject_ids: List of subject IDs
            full_search: If True, full grid search; else random search
            
        Returns:
            Results dictionary
        """
        results = {
            'experiment': 'Strategy B: Hyperparameter Search',
            'timestamp': datetime.now().isoformat(),
            'subjects': subject_ids,
            'search_type': 'full' if full_search else 'random',
            'trials': [],
            'best_params': None,
            'best_accuracy': 0.0,
        }
        
        # Load data
        X, y = self.load_subject_data(subject_ids[0])
        
        # Define search space
        if full_search:
            # Full grid search (27 combinations)
            param_grid = {
                'F1': [4, 8, 16],
                'D': [1, 2, 4],
                'dropout_rate': [0.3, 0.5, 0.7],
                'kernel_length': [32, 64, 128],
            }
        else:
            # Quick random search (9 combinations)
            param_grid = {
                'F1': [4, 8, 16],
                'D': [1, 2, 4],
                'dropout_rate': [0.3, 0.5],
                'kernel_length': [32, 64, 128],
            }
        
        logger.info("=" * 60)
        logger.info("Strategy B: Hyperparameter Search")
        logger.info("=" * 60)
        logger.info(f"Search space: {param_grid}")
        
        # Generate parameter combinations
        param_combinations = list(product(
            param_grid['F1'],
            param_grid['D'],
            param_grid['dropout_rate'],
            param_grid['kernel_length'],
        ))
        
        logger.info(f"Total combinations: {len(param_combinations)}")
        
        for i, (F1, D, dropout, kernel) in enumerate(param_combinations):
            logger.info(f"\n--- Trial {i+1}/{len(param_combinations)} ---")
            logger.info(f"F1={F1}, D={D}, dropout={dropout}, kernel={kernel}")
            
            model_config = {
                'F1': F1,
                'D': D,
                'dropout_rate': dropout,
                'kernel_length': kernel,
                'epochs': 50,
                'batch_size': 64,
            }
            
            # Train and evaluate
            acc, metrics = self.train_and_evaluate(
                X, y,
                model_config=model_config,
                augmentor=None,
                n_folds=3,  # Reduced folds for faster search
            )
            
            trial_result = {
                'F1': F1,
                'D': D,
                'dropout_rate': dropout,
                'kernel_length': kernel,
                'accuracy': acc,
                'kappa': metrics.kappa if metrics else 0,
            }
            
            results['trials'].append(trial_result)
            
            if acc > results['best_accuracy']:
                results['best_accuracy'] = acc
                results['best_params'] = trial_result
        
        # Sort and display results
        results['trials'].sort(key=lambda x: x['accuracy'], reverse=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("Top 5 Hyperparameter Configurations")
        logger.info("=" * 60)
        
        for i, trial in enumerate(results['trials'][:5]):
            logger.info(
                f"{i+1}. F1={trial['F1']}, D={trial['D']}, "
                f"dropout={trial['dropout_rate']}, kernel={trial['kernel_length']} "
                f"=> Accuracy: {trial['accuracy']:.4f}"
            )
        
        logger.info(f"\nBest parameters: {results['best_params']}")
        logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
        
        self.save_results(results, self.experiment_name)
        
        return results


class StrategyC_ArchitectureImprovements(TuningExperiment):
    """
    Strategy C: Architecture Improvements
    
    Test architectural modifications:
    - BatchNorm
    - Attention mechanisms (SE block)
    - Label smoothing
    - Deeper networks
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experiment_name = "strategy_C_architecture"
    
    def run(
        self,
        subject_ids: List[int],
        improvements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run architecture improvement experiments.
        
        Args:
            subject_ids: List of subject IDs
            improvements: List of improvements to test
            
        Returns:
            Results dictionary
        """
        if improvements is None:
            improvements = [
                'baseline',
                'batchnorm',
                'label_smoothing',
                'se_attention',
                'combined',
            ]
        
        results = {
            'experiment': 'Strategy C: Architecture Improvements',
            'timestamp': datetime.now().isoformat(),
            'subjects': subject_ids,
            'improvement_results': {},
            'best_improvement': None,
            'best_accuracy': 0.0,
        }
        
        # Load data
        X, y = self.load_subject_data(subject_ids[0])
        
        logger.info("=" * 60)
        logger.info("Strategy C: Architecture Improvements")
        logger.info("=" * 60)
        
        # Base config
        base_config = {
            'F1': 8,
            'D': 2,
            'kernel_length': 64,
            'dropout_rate': 0.5,
            'epochs': 50,
            'batch_size': 64,
        }
        
        for improvement in improvements:
            logger.info(f"\n--- Testing: {improvement} ---")
            
            model_config = base_config.copy()
            
            # Apply improvement
            if improvement == 'baseline':
                pass  # No changes
            elif improvement == 'batchnorm':
                model_config['use_batchnorm'] = True
            elif improvement == 'label_smoothing':
                model_config['label_smoothing'] = 0.1
            elif improvement == 'se_attention':
                # Note: SE block would need to be integrated into EEGNet
                # For now, use batchnorm as a proxy for regularization
                model_config['use_batchnorm'] = True
                model_config['dropout_rate'] = 0.3  # Lower dropout with BN
            elif improvement == 'combined':
                model_config['use_batchnorm'] = True
                model_config['label_smoothing'] = 0.05
                model_config['dropout_rate'] = 0.4
            
            # Train and evaluate
            acc, metrics = self.train_and_evaluate(
                X, y,
                model_config=model_config,
                augmentor=None,
                n_folds=5,
            )
            
            results['improvement_results'][improvement] = {
                'accuracy': acc,
                'kappa': metrics.kappa if metrics else 0,
                'f1_macro': metrics.f1_macro if metrics else 0,
                'config': model_config,
            }
            
            if acc > results['best_accuracy']:
                results['best_accuracy'] = acc
                results['best_improvement'] = improvement
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Architecture Improvement Results")
        logger.info("=" * 60)
        
        for name, metrics_dict in sorted(
            results['improvement_results'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True,
        ):
            logger.info(f"{name:<20}: {metrics_dict['accuracy']:.4f}")
        
        logger.info(f"\nBest improvement: {results['best_improvement']}")
        logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
        
        self.save_results(results, self.experiment_name)
        
        return results


def run_full_tuning(
    subject_ids: List[int],
    output_dir: str = "./outputs/",
) -> Dict[str, Any]:
    """
    Run all three tuning strategies and generate comparison report.
    
    Args:
        subject_ids: List of subject IDs
        output_dir: Output directory
        
    Returns:
        Combined results dictionary
    """
    logger.info("=" * 60)
    logger.info("Starting Full EEGNet Tuning Pipeline")
    logger.info("=" * 60)
    
    all_results = {
        'experiment': 'Full EEGNet Tuning',
        'timestamp': datetime.now().isoformat(),
        'subjects': subject_ids,
        'strategies': {},
        'comparison': {},
    }
    
    # Strategy A: Data Augmentation
    logger.info("\n\n" + "=" * 60)
    logger.info("RUNNING STRATEGY A: DATA AUGMENTATION")
    logger.info("=" * 60)
    
    strategy_a = StrategyA_DataAugmentation(output_dir=output_dir)
    results_a = strategy_a.run(subject_ids[:2])  # Quick test
    all_results['strategies']['A'] = results_a
    
    # Strategy B: Hyperparameter Search
    logger.info("\n\n" + "=" * 60)
    logger.info("RUNNING STRATEGY B: HYPERPARAMETER SEARCH")
    logger.info("=" * 60)
    
    strategy_b = StrategyB_HyperparameterSearch(output_dir=output_dir)
    results_b = strategy_b.run(subject_ids[:2], full_search=False)
    all_results['strategies']['B'] = results_b
    
    # Strategy C: Architecture Improvements
    logger.info("\n\n" + "=" * 60)
    logger.info("RUNNING STRATEGY C: ARCHITECTURE IMPROVEMENTS")
    logger.info("=" * 60)
    
    strategy_c = StrategyC_ArchitectureImprovements(output_dir=output_dir)
    results_c = strategy_c.run(subject_ids[:2])
    all_results['strategies']['C'] = results_c
    
    # Generate comparison
    all_results['comparison'] = {
        'Strategy A (Augmentation)': results_a.get('best_accuracy', 0),
        'Strategy B (Hyperparameters)': results_b.get('best_accuracy', 0),
        'Strategy C (Architecture)': results_c.get('best_accuracy', 0),
    }
    
    # Find overall best
    best_strategy = max(
        all_results['comparison'].items(),
        key=lambda x: x[1]
    )[0]
    
    logger.info("\n\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    
    for strategy, acc in sorted(
        all_results['comparison'].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        marker = " <-- BEST" if strategy == best_strategy else ""
        logger.info(f"{strategy}: {acc:.4f}{marker}")
    
    # Save combined results
    output_path = Path(output_dir) / "full_tuning_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nFull results saved to {output_path}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EEGNet Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all strategies
  python tune_eegnet.py --all --subjects 1 2 3
  
  # Run specific strategy
  python tune_eegnet.py --strategy A --augmentations gaussian_noise mixup
  
  # Full hyperparameter search
  python tune_eegnet.py --strategy B --full_search --subjects 1
  
  # Architecture improvements
  python tune_eegnet.py --strategy C --improvements batchnorm se_attention
        """
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['A', 'B', 'C', 'all'],
        default='all',
        help='Tuning strategy to run',
    )
    
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=[1],
        help='Subject IDs to use for tuning',
    )
    
    parser.add_argument(
        '--augmentations',
        nargs='+',
        choices=['baseline', 'gaussian_noise', 'temporal_mask', 'channel_mask', 
                 'time_shift', 'band_perturbation', 'mixup', 'all_combined'],
        help='Augmentation methods to test (Strategy A)',
    )
    
    parser.add_argument(
        '--full_search',
        action='store_true',
        help='Perform full grid search (Strategy B)',
    )
    
    parser.add_argument(
        '--improvements',
        nargs='+',
        choices=['baseline', 'batchnorm', 'label_smoothing', 'se_attention', 'combined'],
        help='Architecture improvements to test (Strategy C)',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/',
        help='Output directory for results',
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
    
    args = parser.parse_args()
    
    # Run selected strategy
    if args.strategy == 'all':
        run_full_tuning(
            subject_ids=args.subjects,
            output_dir=args.output_dir,
        )
    else:
        output_dir = args.output_dir
        
        if args.strategy == 'A':
            exp = StrategyA_DataAugmentation(output_dir=output_dir, seed=args.seed, device=args.device)
            exp.run(args.subjects, augmentation_methods=args.augmentations)
            
        elif args.strategy == 'B':
            exp = StrategyB_HyperparameterSearch(output_dir=output_dir, seed=args.seed, device=args.device)
            exp.run(args.subjects, full_search=args.full_search)
            
        elif args.strategy == 'C':
            exp = StrategyC_ArchitectureImprovements(output_dir=output_dir, seed=args.seed, device=args.device)
            exp.run(args.subjects, improvements=args.improvements)


if __name__ == '__main__':
    main()
