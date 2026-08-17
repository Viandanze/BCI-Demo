#!/usr/bin/env python3
"""
CSP (Common Spatial Pattern) Training Script
Train CSP classifier on PhysioNet Motor Imagery dataset

CSP is a classical approach that finds spatial filters maximizing
variance for one class while minimizing for another.

Usage:
    python train_csp.py --subjects 1 2 3
    python train_csp.py --subjects 1 --n_components 6 --classifier svm
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.csp import CSPClassifier
from src.data.loader import load_physionet_data
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.evaluation.metrics import compute_all_metrics, print_results, save_results
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CSP on Motor Imagery BCI',
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
        help='PhysioNet run numbers',
    )
    
    # CSP arguments
    parser.add_argument(
        '--n_components',
        type=int,
        default=4,
        help='Number of CSP components (pairs)',
    )
    parser.add_argument(
        '--regularization',
        type=str,
        default='lwf',
        choices=['lwf', 'shrinkage', 'none'],
        help='Covariance regularization method',
    )
    
    # Classifier arguments
    parser.add_argument(
        '--classifier',
        type=str,
        default='lda',
        choices=['lda', 'svm'],
        help='Classifier after CSP',
    )
    parser.add_argument(
        '--svm_kernel',
        type=str,
        default='rbf',
        choices=['linear', 'rbf', 'poly'],
        help='SVM kernel type',
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
        help='Experiment name',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Cross-validation folds',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='YAML config file',
    )
    
    return parser.parse_args()


def load_data(
    data_path: str,
    subjects: list,
    runs: list,
) -> Dict[int, tuple]:
    """Load and preprocess data for multiple subjects."""
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
    
    data_dict = {}
    for subject_id, raw in raw_dict.items():
        try:
            epochs = pipeline.process_raw(raw)
            X = epochs.get_data()
            y = epochs.events[:, -1]
            
            # Convert to 0-indexed
            unique_labels = sorted(np.unique(y))
            label_map = {l: i for i, l in enumerate(unique_labels)}
            y = np.array([label_map[l] for l in y])
            
            data_dict[subject_id] = (X, y)
            logger.info(f"Subject {subject_id}: {len(X)} epochs, shape {X.shape}")
            
        except Exception as e:
            logger.error(f"Failed to process subject {subject_id}: {e}")
            continue
    
    return data_dict


def train_csp_subject(
    X: np.ndarray,
    y: np.ndarray,
    args,
    subject_id: int,
) -> Dict:
    """Train CSP on single subject data."""
    logger.info(f"Training CSP on data: {X.shape}")
    
    # Get regularization parameter
    reg = 0.1 if args.regularization == 'shrinkage' else args.regularization
    if args.regularization == 'none':
        reg = 1e-5  # Small value to avoid singular matrices
    
    # Create CSP classifier
    clf = CSPClassifier(
        n_components=args.n_components,
        reg=reg,
        classifier=args.classifier,
        svm_kernel=args.svm_kernel,
        normalize=True,
    )
    
    if args.cv_folds > 1:
        # Cross-validation
        kfold = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf_fold = CSPClassifier(
                n_components=args.n_components,
                reg=reg,
                classifier=args.classifier,
                svm_kernel=args.svm_kernel,
                normalize=True,
            )
            
            clf_fold.fit(X_train, y_train)
            y_pred = clf_fold.predict(X_val)
            
            fold_acc = (y_pred == y_val).mean()
            fold_results.append(fold_acc)
            
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            logger.info(f"  Fold {fold+1}: {fold_acc:.4f}")
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        logger.info(f"  Mean: {mean_acc:.4f} ± {std_acc:.4f}")
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        metrics = compute_all_metrics(all_y_true, all_y_pred)
        
        return {
            'fold_accuracies': fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'overall_metrics': metrics,
        }
    
    else:
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=True
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = (y_pred == y_test).mean()
        metrics = compute_all_metrics(y_test, y_pred)
        
        logger.info(f"  Test Accuracy: {acc:.4f}")
        
        return {
            'test_accuracy': acc,
            'test_metrics': metrics,
        }


def main():
    """Main training function."""
    args = parse_args()
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"csp_s{'-'.join(map(str, args.subjects))}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CSP Components: {args.n_components}")
    logger.info(f"Classifier: {args.classifier}")
    
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
        
        results = train_csp_subject(X, y, args, subject_id)
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
    
    for subject_id, results in all_results.items():
        if 'mean_accuracy' in results:
            logger.info(
                f"{subject_id}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
            )
        else:
            logger.info(f"{subject_id}: {results['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
