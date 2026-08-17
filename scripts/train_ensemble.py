#!/usr/bin/env python3
"""
Ensemble Training Script for BCI Classification
Trains multiple base models and combines them using various ensemble strategies

Supports strategies: voting, stacking, weighted, bagging, adaptive

Examples:
    # Train voting ensemble with EEGNet, CSP, and Riemannian
    python scripts/train_ensemble.py --strategy voting --models eegnet csp riemann
    
    # Train stacking ensemble with logistic meta-learner
    python scripts/train_ensemble.py --strategy stacking --meta_learner logistic
    
    # Train with specific subjects
    python scripts/train_ensemble.py --subjects 1 2 3 4 5 --models eegnet csp
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_physionet_data, get_subject_data, create_train_test_split
from src.data.preprocessing import PreprocessingConfig, preprocess_epochs
from src.models.eegnet import EEGNetClassifier
from src.models.csp import CSPClassifier
from src.models.riemann_mdm import RiemannMDMClassifier
from src.training.ensemble import (
    ModelConfig, EnsembleConfig, EnsembleResult,
    VotingEnsemble, StackingEnsemble, WeightedEnsemble,
    BaggingEnsemble, AdaptiveEnsemble, create_ensemble
)
from src.evaluation.metrics import compute_all_metrics, print_results, save_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ensemble models for BCI classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic voting ensemble
  python scripts/train_ensemble.py --strategy voting --models eegnet csp riemann
  
  # Stacking with SVM meta-learner
  python scripts/train_ensemble.py --strategy stacking --meta_learner svm
  
  # Weighted ensemble with custom models
  python scripts/train_ensemble.py --strategy weighted --models eegnet csp
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        '--data_path', type=str, default='./data/',
        help='Path to PhysioNet dataset'
    )
    data_group.add_argument(
        '--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Subject IDs to use for training'
    )
    data_group.add_argument(
        '--runs', type=int, nargs='+', default=[4, 5, 6],
        help='EEG runs to use (4-6 are motor imagery)'
    )
    data_group.add_argument(
        '--test_size', type=float, default=0.2,
        help='Fraction of data for testing'
    )
    data_group.add_argument(
        '--n_classes', type=int, default=2,
        help='Number of classes (2 for binary, 4 for all MI classes)'
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--models', type=str, nargs='+',
        default=['eegnet', 'csp', 'riemann'],
        choices=['eegnet', 'csp', 'riemann'],
        help='Base models to include in ensemble'
    )
    model_group.add_argument(
        '--eegnet_config', type=str, default='{"F1": 8, "D": 2, "kernel_length": 64}',
        help='JSON config for EEGNet'
    )
    model_group.add_argument(
        '--csp_config', type=str, default='{"n_components": 4}',
        help='JSON config for CSP'
    )
    model_group.add_argument(
        '--riemann_config', type=str, default='{"metric": "riemann", "classifier_type": "tangent"}',
        help='JSON config for Riemannian classifier'
    )
    
    # Ensemble arguments
    ensemble_group = parser.add_argument_group('Ensemble Options')
    ensemble_group.add_argument(
        '--strategy', type=str, default='voting',
        choices=['voting', 'stacking', 'weighted', 'bagging', 'adaptive'],
        help='Ensemble strategy to use'
    )
    ensemble_group.add_argument(
        '--voting_mode', type=str, default='soft',
        choices=['hard', 'soft'],
        help='Voting mode (for voting strategy)'
    )
    ensemble_group.add_argument(
        '--meta_learner', type=str, default='logistic',
        choices=['logistic', 'svm', 'ridge'],
        help='Meta-learner type (for stacking strategy)'
    )
    ensemble_group.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of CV folds for stacking'
    )
    ensemble_group.add_argument(
        '--n_bagging', type=int, default=10,
        help='Number of bagging iterations (for bagging strategy)'
    )
    ensemble_group.add_argument(
        '--bagging_ratio', type=float, default=0.8,
        help='Ratio of data per bagging iteration'
    )
    ensemble_group.add_argument(
        '--gating_hidden_dim', type=int, default=64,
        help='Gating network hidden dimension (for adaptive strategy)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs for neural models'
    )
    train_group.add_argument(
        '--batch_size', type=int, default=64,
        help='Training batch size'
    )
    train_group.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate'
    )
    train_group.add_argument(
        '--early_stopping', type=int, default=15,
        help='Early stopping patience'
    )
    train_group.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output_dir', type=str, default='./outputs/ensemble/',
        help='Output directory for results and models'
    )
    output_group.add_argument(
        '--save_models', action='store_true',
        help='Save trained models'
    )
    output_group.add_argument(
        '--verbose', action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


# ============================================================================
# Data Loading
# ============================================================================

def load_data(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """
    Load and preprocess training data.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    logger.info("=" * 60)
    logger.info("Loading data...")
    logger.info("=" * 60)
    
    # Load raw data
    try:
        raw_dict = load_physionet_data(
            data_path=args.data_path,
            subjects=args.subjects,
            runs=args.runs,
        )
    except Exception as e:
        logger.warning(f"Failed to load PhysioNet data: {e}")
        logger.info("Generating synthetic data for demonstration...")
        raw_dict = generate_synthetic_data(args.subjects, args.runs)
    
    # Preprocessing config
    preprocess_config = PreprocessingConfig(
        bandpass_low=4.0,
        bandpass_high=38.0,
        notch_freq=60.0,
        tmin=-1.0,
        tmax=4.0,
        baseline=(-1.0, 0.0),
        resample_freq=128.0,
        normalize=True,
    )
    
    # Collect all epochs
    all_X = []
    all_y = []
    
    for subject_id, raw in raw_dict.items():
        try:
            _, X, y = get_subject_data(
                raw,
                tmin=preprocess_config.tmin,
                tmax=preprocess_config.tmax,
                baseline=preprocess_config.baseline,
            )
            
            # Subsample to n_classes
            if args.n_classes == 2:
                # Binary: left vs right hand
                mask = y < 2
                X, y = X[mask], y[mask]
            
            all_X.append(X)
            all_y.append(y)
            
            logger.info(f"Subject {subject_id}: {len(X)} epochs")
            
        except Exception as e:
            logger.warning(f"Failed to process subject {subject_id}: {e}")
    
    if not all_X:
        raise RuntimeError("No data could be loaded")
    
    # Concatenate all subjects
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    logger.info(f"\nTotal data: {len(X)} epochs")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Classes: {np.bincount(y)}")
    
    # Preprocess
    logger.info("\nApplying preprocessing...")
    X = apply_preprocessing(X, preprocess_config)
    
    # Train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=True,
    )
    
    logger.info(f"\nTrain: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }


def generate_synthetic_data(subjects: List[int], runs: List[int]) -> Dict:
    """Generate synthetic EEG data for testing."""
    from mne.io import RawArray
    import mne
    
    raw_dict = {}
    
    for subject_id in subjects:
        sfreq = 128
        n_channels = 48
        duration = len(runs) * 120  # 2 minutes per run
        n_samples = int(sfreq * duration)
        
        # Generate synthetic signal
        t = np.arange(n_samples) / sfreq
        data = np.random.randn(n_channels, n_samples) * 10
        
        # Add alpha rhythm
        for ch in range(min(16, n_channels)):
            alpha = 20 * np.sin(2 * np.pi * 10 * t + ch * 0.3)
            data[ch] += alpha
        
        # Create Raw object
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = RawArray(data, info)
        
        # Add synthetic events
        events = []
        for run_idx in range(len(runs)):
            run_start = run_idx * int(120 * sfreq)
            for i in range(20):
                event_start = run_start + int((5 + i * 5) * sfreq)
                events.append([event_start, 0, 1 if i % 2 == 0 else 2])
        
        if events:
            events = np.array(events)
            annot = mne.annotations_from_events(
                events=events,
                sfreq=sfreq,
                event_desc={1: 'T1', 2: 'T2'},
            )
            raw = raw.set_annotations(annot)
        
        raw_dict[subject_id] = raw
    
    return raw_dict


def apply_preprocessing(X: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply preprocessing to data."""
    # Simple preprocessing: channel-wise z-score
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    X = (X - mean) / std
    
    return X


# ============================================================================
# Model Creation
# ============================================================================

def create_model_configs(args: argparse.Namespace) -> List[ModelConfig]:
    """
    Create ModelConfig objects from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of ModelConfig objects
    """
    configs = []
    
    # Parse JSON configs
    eegnet_config = json.loads(args.eegnet_config)
    csp_config = json.loads(args.csp_config)
    riemann_config = json.loads(args.riemann_config)
    
    for model_type in args.models:
        if model_type == 'eegnet':
            configs.append(ModelConfig(
                model_type='eegnet',
                name='eegnet',
                config=eegnet_config,
                is_torch=True,
                train_config={
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'early_stopping_patience': args.early_stopping,
                }
            ))
        
        elif model_type == 'csp':
            configs.append(ModelConfig(
                model_type='csp',
                name='csp',
                config=csp_config,
                is_torch=False,
            ))
        
        elif model_type == 'riemann':
            configs.append(ModelConfig(
                model_type='riemann',
                name='riemann',
                config=riemann_config,
                is_torch=False,
            ))
    
    return configs


def create_ensemble_from_args(
    args: argparse.Namespace,
    model_configs: List[ModelConfig],
) -> Any:
    """
    Create ensemble based on arguments.
    
    Args:
        args: Command line arguments
        model_configs: List of ModelConfig objects
        
    Returns:
        Configured ensemble instance
    """
    # Create ensemble config
    ensemble_config = EnsembleConfig(
        strategy=args.strategy,
        n_folds=args.n_folds,
        random_state=args.seed,
        save_base_models=args.save_models,
        n_bagging=args.n_bagging,
        bagging_ratio=args.bagging_ratio,
        gating_hidden_dim=args.gating_hidden_dim,
    )
    
    # Create ensemble based on strategy
    if args.strategy == 'voting':
        return create_ensemble(
            model_configs,
            strategy='voting',
            config=ensemble_config,
            voting_mode=args.voting_mode,
        )
    
    elif args.strategy == 'stacking':
        return create_ensemble(
            model_configs,
            strategy='stacking',
            config=ensemble_config,
            meta_learner=args.meta_learner,
        )
    
    elif args.strategy == 'weighted':
        return create_ensemble(
            model_configs,
            strategy='weighted',
            config=ensemble_config,
        )
    
    elif args.strategy == 'bagging':
        return create_ensemble(
            model_configs,
            strategy='bagging',
            config=ensemble_config,
            n_bagging=args.n_bagging,
            bagging_ratio=args.bagging_ratio,
        )
    
    elif args.strategy == 'adaptive':
        return create_ensemble(
            model_configs,
            strategy='adaptive',
            config=ensemble_config,
            gating_hidden_dim=args.gating_hidden_dim,
        )
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_ensemble(
    ensemble: Any,
    data: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> EnsembleResult:
    """
    Train the ensemble model.
    
    Args:
        ensemble: Ensemble instance
        data: Dictionary with train/test data
        args: Command line arguments
        
    Returns:
        EnsembleResult with predictions and metrics
    """
    logger.info("=" * 60)
    logger.info(f"Training {type(ensemble).__name__}...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        data['X_train'], data['y_train'],
        test_size=0.2,
        stratify=data['y_train'],
        random_state=args.seed,
    )
    
    # Train
    try:
        ensemble.fit(X_train, y_train, X_val, y_val)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    training_time = time.time() - start_time
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    
    predictions = ensemble.predict(data['X_test'])
    probabilities = ensemble.predict_proba(data['X_test'])
    
    # Compute metrics
    metrics = compute_all_metrics(
        data['y_test'],
        predictions,
        probabilities,
    )
    
    # Get individual model scores
    individual_scores = {}
    if hasattr(ensemble, 'get_individual_scores'):
        individual_scores = ensemble.get_individual_scores(
            data['X_test'], data['y_test']
        )
    
    # Get ensemble weights if available
    ensemble_weights = None
    if hasattr(ensemble, 'model_weights_'):
        ensemble_weights = ensemble.model_weights_
    
    return EnsembleResult(
        accuracy=metrics.accuracy,
        kappa=metrics.kappa,
        predictions=predictions,
        probabilities=probabilities,
        confidences=probabilities.max(axis=1),
        individual_scores=individual_scores,
        ensemble_weights=ensemble_weights,
        training_time=training_time,
    )


def evaluate_single_models(
    model_configs: List[ModelConfig],
    data: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Evaluate individual models for comparison.
    
    Args:
        model_configs: List of ModelConfig objects
        data: Dictionary with train/test data
        args: Command line arguments
        
    Returns:
        Dictionary mapping model name to accuracy
    """
    from src.training.trainer import Trainer, TrainingConfig
    from sklearn.metrics import accuracy_score
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    results = {}
    
    for config in model_configs:
        logger.info(f"\nEvaluating single model: {config.name}")
        
        # Create model
        n_channels = data['X_train'].shape[1]
        n_times = data['X_train'].shape[2]
        n_classes = len(np.unique(data['y_train']))
        
        if config.model_type == 'eegnet':
            model = EEGNetClassifier(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=n_classes,
                **config.config,
            )
            
            # Train
            train_config = config.train_config or {}
            trainer_config = TrainingConfig(
                epochs=train_config.get('epochs', args.epochs),
                batch_size=train_config.get('batch_size', args.batch_size),
                learning_rate=train_config.get('learning_rate', args.learning_rate),
            )
            
            X_tensor = torch.FloatTensor(data['X_train'])
            y_tensor = torch.LongTensor(data['y_train'])
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            trainer = Trainer(model=model, config=trainer_config)
            trainer.train(train_loader, epochs=args.epochs)
            
            # Evaluate
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                X_test = torch.FloatTensor(data['X_test']).to(device)
                preds = torch.argmax(model(X_test), dim=1).cpu().numpy()
            
            acc = accuracy_score(data['y_test'], preds)
            
        elif config.model_type == 'csp':
            model = CSPClassifier(**config.config)
            model.fit(data['X_train'], data['y_train'])
            preds = model.predict(data['X_test'])
            acc = accuracy_score(data['y_test'], preds)
            
        elif config.model_type == 'riemann':
            model = RiemannMDMClassifier(**config.config)
            model.fit(data['X_train'], data['y_train'])
            preds = model.predict(data['X_test'])
            acc = accuracy_score(data['y_test'], preds)
        
        results[config.name] = acc
        logger.info(f"  {config.name} accuracy: {acc:.4f}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(args)
    
    # Create model configs
    model_configs = create_model_configs(args)
    
    # Evaluate single models first
    logger.info("\n" + "=" * 60)
    logger.info("Single Model Evaluation (Baseline)")
    logger.info("=" * 60)
    
    single_model_results = evaluate_single_models(model_configs, data, args)
    
    # Create ensemble
    logger.info("\n" + "=" * 60)
    logger.info(f"Ensemble Training ({args.strategy})")
    logger.info("=" * 60)
    
    ensemble = create_ensemble_from_args(args, model_configs)
    
    # Train ensemble
    try:
        ensemble_result = train_ensemble(ensemble, data, args)
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        sys.exit(1)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    print("\nSingle Model Results:")
    for name, acc in single_model_results.items():
        print(f"  {name}: {acc:.4f}")
    
    print(f"\nEnsemble ({args.strategy}) Results:")
    print(f"  Accuracy: {ensemble_result.accuracy:.4f}")
    print(f"  Kappa: {ensemble_result.kappa:.4f}")
    print(f"  Training Time: {ensemble_result.training_time:.2f}s")
    
    if ensemble_result.individual_scores:
        print("\nIndividual Model Scores (on ensemble test):")
        for name, score in ensemble_result.individual_scores.items():
            print(f"  {name}: {score:.4f}")
    
    if ensemble_result.ensemble_weights:
        print("\nEnsemble Weights:")
        for name, weight in ensemble_result.ensemble_weights.items():
            print(f"  {name}: {weight:.4f}")
    
    # Improvement over best single model
    best_single = max(single_model_results.values())
    improvement = ensemble_result.accuracy - best_single
    print(f"\nImprovement over best single model: {improvement:+.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'strategy': args.strategy,
        'models': args.models,
        'single_model_results': single_model_results,
        'ensemble_accuracy': ensemble_result.accuracy,
        'ensemble_kappa': ensemble_result.kappa,
        'ensemble_training_time': ensemble_result.training_time,
        'ensemble_result': ensemble_result.to_dict(),
        'args': vars(args),
    }
    
    results_path = output_dir / f"results_{args.strategy}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save models if requested
    if args.save_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        try:
            ensemble.save_models(models_dir)
            logger.info(f"Models saved to: {models_dir}")
        except Exception as e:
            logger.warning(f"Failed to save models: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
