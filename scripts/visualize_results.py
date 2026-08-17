#!/usr/bin/env python3
"""
Visualization Script for BCI Training Results
Generate comprehensive visualizations from training output JSON files.

Usage:
    python visualize_results.py --mode training --input ./outputs/experiment/results.json
    python visualize_results.py --mode confusion --input ./outputs/experiment/results.json
    python visualize_results.py --mode topo --features ./outputs/features.npy --channels 64
    python visualize_results.py --mode tsne --input ./outputs/experiment/results.json
    python visualize_results.py --mode erp --input ./outputs/experiment/results.json --epochs ./data/

Modes:
    training: Training and validation curves
    confusion: Confusion matrix heatmap
    topo: EEG topographic map
    tsne: t-SNE feature visualization
    erp: Event-related potential waveforms
    all: Generate all visualizations
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.visualizer import (
    BCIVisualizer,
    MOTOR_IMAGERY_LABELS,
    load_training_results,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize BCI Training Results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with training results',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for figures (default: same as input)',
    )
    
    # Visualization mode
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['training', 'confusion', 'topo', 'tsne', 'erp', 'cross_subject', 'all'],
        help='Visualization mode',
    )
    
    # Visualization options
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for plots',
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='Class labels',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI',
    )
    
    # Feature input (for topo/tsne)
    parser.add_argument(
        '--features',
        type=str,
        help='Path to feature numpy file',
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=64,
        help='Number of EEG channels',
    )
    parser.add_argument(
        '--ch_names',
        type=str,
        nargs='+',
        help='Channel names',
    )
    
    # Subject ID (for multi-subject results)
    parser.add_argument(
        '--subject_id',
        type=int,
        default=1,
        help='Subject ID to visualize',
    )
    
    # ERP options
    parser.add_argument(
        '--epochs_path',
        type=str,
        help='Path to epoched EEG data (for ERP)',
    )
    parser.add_argument(
        '--tmin',
        type=float,
        default=-1.0,
        help='Epoch start time (s)',
    )
    parser.add_argument(
        '--tmax',
        type=float,
        default=4.0,
        help='Epoch end time (s)',
    )
    parser.add_argument(
        '--sfreq',
        type=float,
        default=128.0,
        help='Sampling frequency',
    )
    
    return parser.parse_args()


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load training results from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Results dictionary
    """
    logger.info(f"Loading results from {input_path}")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded results with keys: {list(results.keys())}")
    return results


def extract_subject_results(
    results: Dict[str, Any],
    subject_id: int,
) -> Dict[str, Any]:
    """
    Extract results for specific subject.
    
    Args:
        results: Full results dictionary
        subject_id: Subject ID
        
    Returns:
        Subject-specific results
    """
    subject_key = f"subject_{subject_id}"
    
    if subject_key in results:
        return results[subject_key]
    
    # If no subject key, assume results are at top level
    return results


def plot_training_curves(
    results: Dict[str, Any],
    viz: BCIVisualizer,
    title: str,
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        results: Training results
        viz: Visualizer instance
        title: Plot title
    """
    history = results.get('history', {})
    
    if not history:
        logger.warning("No training history found in results")
        return
    
    # Split history into train and val
    train_history = {
        'train_loss': history.get('train_loss', []),
        'train_acc': history.get('train_acc', []),
    }
    
    val_history = {
        'train_loss': history.get('val_loss', []),
        'train_acc': history.get('val_acc', []),
    }
    
    viz.plot_training_curves(
        train_history,
        val_history,
        title=title,
        save_name='training_curves',
    )
    
    logger.info("Training curves saved")


def plot_confusion_matrix(
    results: Dict[str, Any],
    viz: BCIVisualizer,
    title: str,
    labels: Optional[List[str]] = None,
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        results: Training results
        viz: Visualizer instance
        title: Plot title
        labels: Class labels
    """
    y_true = results.get('y_true')
    y_pred = results.get('y_pred')
    
    if y_true is None or y_pred is None:
        logger.warning("No predictions found in results")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_classes = len(np.unique(y_true))
    
    if labels is None:
        labels = [MOTOR_IMAGERY_LABELS.get(i, f'Class {i}') for i in range(n_classes)]
    
    viz.plot_confusion_matrix(
        y_true, y_pred,
        labels=labels,
        title=title,
        normalize=True,
        save_name='confusion_matrix',
    )
    
    logger.info("Confusion matrix saved")


def plot_topographic_map(
    features: np.ndarray,
    viz: BCIVisualizer,
    title: str,
    ch_names: Optional[List[str]] = None,
) -> None:
    """
    Plot EEG topographic map from feature vector.
    
    Args:
        features: Feature values per channel
        viz: Visualizer instance
        title: Plot title
        ch_names: Channel names
    """
    if features.ndim == 2:
        # Average over first dimension
        features = features.mean(axis=0)
    
    if features.size > 128:
        logger.warning(f"Feature vector too large ({features.size}), truncating")
        features = features[:128]
    
    viz.plot_topomap(
        features,
        ch_names=ch_names,
        title=title,
        save_name='topographic_map',
    )
    
    logger.info("Topographic map saved")


def plot_tsne_features(
    features: np.ndarray,
    labels: np.ndarray,
    viz: BCIVisualizer,
    title: str,
    labels_list: Optional[List[str]] = None,
) -> None:
    """
    Plot t-SNE visualization of features.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels
        viz: Visualizer instance
        title: Plot title
        labels_list: Class names
    """
    viz.plot_tsne_features(
        features,
        labels,
        title=title,
        perplexity=min(30, features.shape[0] // 4),
        classes=labels_list,
        save_name='tsne_features',
    )
    
    logger.info("t-SNE visualization saved")


def plot_erp_waveforms(
    epochs_path: str,
    viz: BCIVisualizer,
    title: str,
    tmin: float,
    tmax: float,
    sfreq: float,
    labels: Optional[List[str]] = None,
) -> None:
    """
    Plot ERP waveforms from epoched data.
    
    Args:
        epochs_path: Path to epoched data (npy file)
        viz: Visualizer instance
        title: Plot title
        tmin, tmax: Time window
        sfreq: Sampling frequency
        labels: Class labels
    """
    if not Path(epochs_path).exists():
        logger.warning(f"Epochs file not found: {epochs_path}")
        logger.info("Skipping ERP visualization")
        return
    
    data = np.load(epochs_path, allow_pickle=True)
    
    # Handle different data formats
    if isinstance(data, dict):
        X = data.get('X', data.get('data'))
        y = data.get('y', data.get('labels'))
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        X, y = data
    else:
        X = data
        y = None
    
    if y is None:
        logger.warning("No labels found in epochs data")
        return
    
    viz.plot_erp(
        X, y,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
        channels=None,
        title=title,
        classes=labels,
        save_name='erp_waveforms',
    )
    
    logger.info("ERP waveforms saved")


def plot_cross_subject_comparison(
    results: Dict[str, Any],
    viz: BCIVisualizer,
    title: str,
) -> None:
    """
    Plot cross-subject performance comparison.
    
    Args:
        results: Results with multiple subjects
        viz: Visualizer instance
        title: Plot title
    """
    subject_data = {}
    
    for key, value in results.items():
        if key.startswith('subject_'):
            # Extract accuracy
            if 'test_metrics' in value:
                acc = value['test_metrics'].get('accuracy')
            elif 'mean_accuracy' in value:
                acc = value['mean_accuracy']
            else:
                continue
            
            subject_id = key.split('_')[1]
            subject_data[f"Subject {subject_id}"] = [acc]
    
    if not subject_data:
        logger.warning("No subject data found for comparison")
        return
    
    viz.plot_cross_subject_comparison(
        subject_data,
        metric='Accuracy',
        title=title,
        plot_type='boxplot',
        save_name='cross_subject_comparison',
    )
    
    logger.info("Cross-subject comparison saved")


def load_features(features_path: str) -> Optional[np.ndarray]:
    """
    Load feature matrix from file.
    
    Args:
        features_path: Path to numpy file
        
    Returns:
        Feature array or None
    """
    if not Path(features_path).exists():
        logger.warning(f"Features file not found: {features_path}")
        return None
    
    features = np.load(features_path, allow_pickle=True)
    logger.info(f"Loaded features shape: {features.shape}")
    return features


def main():
    """Main visualization function."""
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    viz = BCIVisualizer(output_dir=str(output_dir), dpi=args.dpi)
    
    # Default title
    if args.title is None:
        experiment_name = Path(args.input).parent.name
        args.title = f"{experiment_name} - {args.mode.replace('_', ' ').title()}"
    
    # Default labels
    if args.labels is None:
        args.labels = ['Left Hand', 'Right Hand', 'Feet', 'Rest']
    
    # Load results
    try:
        results = load_results(args.input)
        subject_results = extract_subject_results(results, args.subject_id)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Execute visualization based on mode
    if args.mode == 'training':
        plot_training_curves(subject_results, viz, args.title)
    
    elif args.mode == 'confusion':
        plot_confusion_matrix(
            subject_results, viz, args.title,
            labels=args.labels,
        )
    
    elif args.mode == 'topo':
        # Try to get features from results or external file
        features = None
        
        if args.features:
            features = load_features(args.features)
        elif 'features' in subject_results:
            features = np.array(subject_results['features'])
        elif 'model_config' in subject_results:
            # Try to extract spatial patterns
            logger.info("No explicit features found, skipping topo plot")
            return
        
        if features is not None:
            plot_topographic_map(
                features, viz, args.title,
                ch_names=args.ch_names,
            )
    
    elif args.mode == 'tsne':
        # Load features for t-SNE
        features = None
        labels = None
        
        if args.features:
            features = load_features(args.features)
        
        if features is not None:
            # Try to get labels
            if 'y_true' in subject_results:
                labels = np.array(subject_results['y_true'])
            else:
                logger.warning("No labels found for t-SNE")
                return
            
            plot_tsne_features(
                features, labels, viz, args.title,
                labels_list=args.labels,
            )
        else:
            logger.warning("No features available for t-SNE")
            return
    
    elif args.mode == 'erp':
        plot_erp_waveforms(
            args.epochs_path or '',
            viz, args.title,
            tmin=args.tmin,
            tmax=args.tmax,
            sfreq=args.sfreq,
            labels=args.labels,
        )
    
    elif args.mode == 'cross_subject':
        plot_cross_subject_comparison(results, viz, args.title)
    
    elif args.mode == 'all':
        # Generate all visualizations
        logger.info("Generating all visualizations...")
        
        # Training curves
        if 'history' in subject_results:
            plot_training_curves(subject_results, viz, f"{args.title} - Training")
        
        # Confusion matrix
        if 'y_true' in subject_results:
            plot_confusion_matrix(
                subject_results, viz,
                f"{args.title} - Confusion Matrix",
                labels=args.labels,
            )
        
        # Cross-subject comparison
        if len(results) > 1:
            plot_cross_subject_comparison(results, viz, f"{args.title} - Cross-Subject")
        
        logger.info("All visualizations complete")
    
    logger.info(f"Output saved to {output_dir}")


if __name__ == '__main__':
    main()
