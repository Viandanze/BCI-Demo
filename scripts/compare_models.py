#!/usr/bin/env python3
"""
Model Comparison Script
Compare EEGNet, CSP, and Riemannian classifiers on the same dataset

This script provides a comprehensive comparison of different BCI approaches,
helping identify the best method for your specific requirements.

Usage:
    python compare_models.py --subjects 1 2 3 4 5
    python compare_models.py --subjects 1 --quick
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eegnet import EEGNetClassifier
from src.models.csp import CSPClassifier
from src.models.riemann_mdm import RiemannMDMClassifier, TangentSpaceClassifier, PYRIEMANN_AVAILABLE
from src.training.trainer import Trainer, TrainingConfig, set_seed, prepare_data_loaders
from src.training.augment import EEGAugmentor, AugmentationConfig
from src.data.loader import load_physionet_data
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.evaluation.metrics import (
    compute_all_metrics,
    print_results,
    format_results_table,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple BCI models on the same dataset.
    
    Supports:
    - EEGNet (deep learning)
    - CSP + LDA/SVM (classical)
    - Riemannian MDM (covariance-based)
    """
    
    def __init__(
        self,
        data_path: str = "./data/",
        output_dir: str = "./outputs/",
        seed: int = 42,
        device: str = "auto",
        cv_folds: int = 5,
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.cv_folds = cv_folds
        self.device = device
        
        set_seed(seed)
        
        # Results storage
        self.results = {}
        
        logger.info("ModelComparison initialized")
    
    def load_subject_data(
        self,
        subject_id: int,
        runs: List[int] = [4, 5, 6],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load data for a single subject."""
        logger.info(f"Loading data for subject {subject_id}")
        
        try:
            raw_dict = load_physionet_data(
                data_path=self.data_path,
                subjects=[subject_id],
                runs=runs,
            )
            raw = raw_dict[subject_id]
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
            from src.data.loader import _create_sample_data
            raw = _create_sample_data(subject_id, runs)
        
        # Preprocessing
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
        
        X = epochs.get_data()
        y = epochs.events[:, -1]
        
        # Convert to 0-indexed
        unique_labels = sorted(np.unique(y))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y])
        
        logger.info(f"Loaded {len(X)} epochs, shape {X.shape}")
        
        return X, y
    
    def train_eegnet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict,
        augmentor: Any = None,
    ) -> Dict[str, Any]:
        """Train EEGNet with cross-validation."""
        logger.info("Training EEGNet...")
        start_time = time.time()
        
        n_channels = X.shape[1]
        n_times = X.shape[2]
        n_classes = len(np.unique(y))
        
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            model = EEGNetClassifier(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=n_classes,
                F1=config.get('F1', 8),
                D=config.get('D', 2),
                kernel_length=config.get('kernel_length', 64),
                dropout_rate=config.get('dropout_rate', 0.5),
                use_batchnorm=config.get('use_batchnorm', False),
                apply_softmax=False,
            )
            
            # Training config
            train_config = TrainingConfig(
                epochs=config.get('epochs', 50),
                batch_size=config.get('batch_size', 64),
                learning_rate=config.get('lr', 0.001),
                device=self.device,
            )
            
            trainer = Trainer(model, train_config)
            
            # Data loaders
            train_loader, val_loader = prepare_data_loaders(
                X_train, y_train,
                X_val, y_val,
                batch_size=train_config.batch_size,
                augmentor=augmentor,
            )
            
            # Train
            trainer.train(train_loader, val_loader, epochs=train_config.epochs)
            
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
        
        elapsed = time.time() - start_time
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        metrics = compute_all_metrics(all_y_true, all_y_pred)
        
        logger.info(f"EEGNet: {mean_acc:.4f} ± {std_acc:.4f} ({elapsed:.1f}s)")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_results,
            'metrics': metrics,
            'training_time': elapsed,
        }
    
    def train_csp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict,
    ) -> Dict[str, Any]:
        """Train CSP classifier with cross-validation."""
        logger.info("Training CSP...")
        start_time = time.time()
        
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create classifier
            clf = CSPClassifier(
                n_components=config.get('n_components', 4),
                reg=config.get('reg', 'lwf'),
                classifier=config.get('classifier', 'lda'),
                normalize=True,
            )
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            
            fold_acc = accuracy_score(y_val, y_pred)
            fold_results.append(fold_acc)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
        
        elapsed = time.time() - start_time
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        metrics = compute_all_metrics(all_y_true, all_y_pred)
        
        logger.info(f"CSP: {mean_acc:.4f} ± {std_acc:.4f} ({elapsed:.1f}s)")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_results,
            'metrics': metrics,
            'training_time': elapsed,
        }
    
    def train_riemann(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict,
    ) -> Dict[str, Any]:
        """Train Riemannian classifier with cross-validation."""
        logger.info("Training Riemannian MDM...")
        start_time = time.time()
        
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create classifier
            if config.get('classifier_type', 'mdm') == 'mdm':
                clf = RiemannMDMClassifier(metric=config.get('metric', 'riemann'))
            else:
                clf = TangentSpaceClassifier(metric=config.get('metric', 'riemann'))
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            
            fold_acc = accuracy_score(y_val, y_pred)
            fold_results.append(fold_acc)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
        
        elapsed = time.time() - start_time
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        metrics = compute_all_metrics(all_y_true, all_y_pred)
        
        logger.info(f"Riemannian: {mean_acc:.4f} ± {std_acc:.4f} ({elapsed:.1f}s)")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_results,
            'metrics': metrics,
            'training_time': elapsed,
        }
    
    def run_comparison(
        self,
        subject_ids: List[int],
        runs: List[int] = [4, 5, 6],
        quick: bool = False,
    ) -> Dict[str, Any]:
        """
        Run comprehensive model comparison.
        
        Args:
            subject_ids: Subject IDs to include
            runs: PhysioNet runs
            quick: If True, use fewer epochs and folds
            
        Returns:
            Comparison results dictionary
        """
        if quick:
            logger.info("Running in QUICK mode (reduced epochs/folds)")
        
        results = {
            'experiment': 'Model Comparison',
            'timestamp': datetime.now().isoformat(),
            'subjects': subject_ids,
            'pyriemann_available': PYRIEMANN_AVAILABLE,
            'individual_results': {},
            'summary': {},
        }
        
        # Model configurations
        eegnet_config = {
            'F1': 8,
            'D': 2,
            'kernel_length': 64,
            'dropout_rate': 0.5,
            'epochs': 30 if quick else 50,
            'batch_size': 64,
            'lr': 0.001,
        }
        
        csp_config = {
            'n_components': 4,
            'reg': 'lwf',
            'classifier': 'lda',
        }
        
        riemann_config = {
            'classifier_type': 'mdm',
            'metric': 'riemann',
        }
        
        # Data augmentation for EEGNet
        if not quick:
            aug_config = AugmentationConfig(
                enabled=True,
                probability=0.5,
                gaussian_noise={'enabled': True, 'prob': 0.3, 'snr_db': 10},
            )
            augmentor = EEGAugmentor(aug_config, sfreq=128, random_state=self.seed)
        else:
            augmentor = None
        
        # Quick mode: fewer CV folds
        if quick:
            self.cv_folds = 3
        
        # Aggregate results across subjects
        all_eegnet_results = []
        all_csp_results = []
        all_riemann_results = []
        
        for subject_id in subject_ids:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Subject {subject_id}")
            logger.info(f"{'=' * 60}")
            
            try:
                X, y = self.load_subject_data(subject_id, runs)
                
                subject_results = {}
                
                # EEGNet
                eegnet_result = self.train_eegnet(X, y, eegnet_config, augmentor)
                subject_results['eegnet'] = eegnet_result
                all_eegnet_results.append(eegnet_result['mean_accuracy'])
                
                # CSP
                csp_result = self.train_csp(X, y, csp_config)
                subject_results['csp'] = csp_result
                all_csp_results.append(csp_result['mean_accuracy'])
                
                # Riemannian
                riemann_result = self.train_riemann(X, y, riemann_config)
                subject_results['riemann'] = riemann_result
                all_riemann_results.append(riemann_result['mean_accuracy'])
                
                results['individual_results'][f'subject_{subject_id}'] = subject_results
                
            except Exception as e:
                logger.error(f"Failed on subject {subject_id}: {e}")
                continue
        
        # Compute summary statistics
        if all_eegnet_results:
            results['summary']['eegnet'] = {
                'mean': np.mean(all_eegnet_results),
                'std': np.std(all_eegnet_results),
                'subjects_tested': len(all_eegnet_results),
            }
        
        if all_csp_results:
            results['summary']['csp'] = {
                'mean': np.mean(all_csp_results),
                'std': np.std(all_csp_results),
                'subjects_tested': len(all_csp_results),
            }
        
        if all_riemann_results:
            results['summary']['riemann'] = {
                'mean': np.mean(all_riemann_results),
                'std': np.std(all_riemann_results),
                'subjects_tested': len(all_riemann_results),
            }
        
        # Find best model
        if results['summary']:
            best_model = max(
                results['summary'].items(),
                key=lambda x: x[1]['mean']
            )[0]
            results['summary']['best_model'] = best_model
        
        # Print summary
        logger.info(f"\n\n{'=' * 60}")
        logger.info("FINAL COMPARISON SUMMARY")
        logger.info(f"{'=' * 60}")
        
        for model_name, stats in results['summary'].items():
            if model_name == 'best_model':
                continue
            logger.info(f"{model_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'best_model' in results['summary']:
            logger.info(f"\nBest Model: {results['summary']['best_model'].upper()}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"comparison_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {save_path}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare BCI Models (EEGNet, CSP, Riemannian)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
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
        default=[1, 2],
        help='Subject IDs to compare',
    )
    parser.add_argument(
        '--runs',
        type=int,
        nargs='+',
        default=[4, 5, 6],
        help='PhysioNet run numbers',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/',
        help='Output directory',
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Cross-validation folds',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (fewer epochs/folds)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    
    return parser.parse_args()


def main():
    """Main comparison function."""
    args = parse_args()
    
    logger.info("Starting Model Comparison")
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"CV Folds: {args.cv_folds}")
    logger.info(f"Quick mode: {args.quick}")
    
    comparison = ModelComparison(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )
    
    results = comparison.run_comparison(
        subject_ids=args.subjects,
        runs=args.runs,
        quick=args.quick,
    )
    
    return results


if __name__ == '__main__':
    main()
