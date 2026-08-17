"""
BCI Visualization Module
Comprehensive visualization toolkit for EEG motor imagery BCI analysis.

Features:
    - Training curves with multi-experiment comparison
    - Confusion matrix heatmaps
    - EEG topographic maps (2D projection with 10-20 system)
    - Power spectral density analysis
    - ERP waveforms with confidence intervals
    - Cross-subject comparison (boxplot/violin)
    - t-SNE feature visualization
    - CSP pattern topography

Author: BCI_Projects
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Try to set Chinese font, fallback to default
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    logger.warning("Chinese font not available, using default")

# Global style settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Standard 10-20 system electrode positions (2D projection)
# Format: (name, x_position, y_position)
STANDARD_10_20_2D = {
    # Frontal
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'Fpz': (0.0, 0.9),
    # Frontal
    'AF7': (-0.6, 0.75), 'AF3': (-0.3, 0.75), 'AF4': (0.3, 0.75), 'AF8': (0.6, 0.75),
    # Anterior frontal
    'F7': (-0.8, 0.5), 'F5': (-0.5, 0.5), 'F3': (-0.3, 0.5), 'F1': (-0.15, 0.5),
    'Fz': (0.0, 0.5), 'F2': (0.15, 0.5), 'F4': (0.3, 0.5), 'F6': (0.5, 0.5), 'F8': (0.8, 0.5),
    # Central
    'FT7': (-0.85, 0.25), 'FC5': (-0.6, 0.25), 'FC3': (-0.35, 0.25), 'FC1': (-0.15, 0.25),
    'FCz': (0.0, 0.25), 'FC2': (0.15, 0.25), 'FC4': (0.35, 0.25), 'FC6': (0.6, 0.25), 'FT8': (0.85, 0.25),
    # Central
    'T7': (-1.0, 0.0), 'C5': (-0.65, 0.0), 'C3': (-0.35, 0.0), 'C1': (-0.15, 0.0),
    'Cz': (0.0, 0.0), 'C2': (0.15, 0.0), 'C4': (0.35, 0.0), 'C6': (0.65, 0.0), 'T8': (1.0, 0.0),
    # Post-central
    'TP7': (-0.85, -0.25), 'CP5': (-0.6, -0.25), 'CP3': (-0.35, -0.25), 'CP1': (-0.15, -0.25),
    'CPz': (0.0, -0.25), 'CP2': (0.15, -0.25), 'CP4': (0.35, -0.25), 'CP6': (0.6, -0.25), 'TP8': (0.85, -0.25),
    # Parietal
    'P7': (-0.8, -0.5), 'P5': (-0.5, -0.5), 'P3': (-0.3, -0.5), 'P1': (-0.15, -0.5),
    'Pz': (0.0, -0.5), 'P2': (0.15, -0.5), 'P4': (0.3, -0.5), 'P6': (0.5, -0.5), 'P8': (0.8, -0.5),
    # Occipital
    'PO7': (-0.6, -0.75), 'PO3': (-0.3, -0.75), 'PO4': (0.3, -0.75), 'PO8': (0.6, -0.75),
    # Occipital
    'O1': (-0.3, -0.9), 'Oz': (0.0, -0.9), 'O2': (0.3, -0.9),
    # Additional common channels
    'FpZ': (0.0, 0.9), 'Fpz': (0.0, 0.9), 'AFz': (0.0, 0.75), 'Fz': (0.0, 0.5),
    'Cz': (0.0, 0.0), 'Pz': (0.0, -0.5), 'Oz': (0.0, -0.9),
}

# Motor imagery class labels
MOTOR_IMAGERY_LABELS = {
    0: 'Left Hand',
    1: 'Right Hand',
    2: 'Feet',
    3: 'Rest/Tongue',
}

# Frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

# Color scheme for frequency bands
FREQ_BAND_COLORS = {
    'delta': '#1f77b4',  # blue
    'theta': '#ff7f0e',  # orange
    'alpha': '#2ca02c',  # green
    'beta': '#d62728',   # red
    'gamma': '#9467bd',  # purple
}


def set_chinese_font():
    """
    Set Chinese font for matplotlib with fallback.
    
    Tries SimHei first, then falls back to available fonts.
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        logger.warning(f"Chinese font setup failed: {e}")


def get_channel_positions(ch_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Get 2D positions for given channel names based on 10-20 system.
    
    Args:
        ch_names: List of channel names
        
    Returns:
        Tuple of (x, y, matched_names) where x/y are position arrays
    """
    x, y, matched = [], [], []
    
    for ch in ch_names:
        ch_upper = ch.upper()
        if ch_upper in STANDARD_10_20_2D:
            pos = STANDARD_10_20_2D[ch_upper]
            x.append(pos[0])
            y.append(pos[1])
            matched.append(ch)
        elif ch in STANDARD_10_20_2D:
            pos = STANDARD_10_20_2D[ch]
            x.append(pos[0])
            y.append(pos[1])
            matched.append(ch)
        else:
            # Assign generic position if not found
            idx = len(x)
            angle = (idx / max(1, len(ch_names))) * 2 * np.pi
            x.append(np.cos(angle) * 0.8)
            y.append(np.sin(angle) * 0.8)
            matched.append(ch)
            logger.debug(f"Channel {ch} not in standard positions, using generic")
    
    return np.array(x), np.array(y), matched


class BCIVisualizer:
    """
    Main visualization class for BCI analysis.
    
    Provides comprehensive plotting functions for:
        - Training curves
        - Confusion matrices
        - Topographic maps
        - Power spectra
        - ERP waveforms
        - Cross-subject comparisons
        - Feature embeddings
    
    Example:
        viz = BCIVisualizer(output_dir='./outputs/figures')
        viz.plot_training_curves(history, title='EEGNet Training')
        viz.plot_confusion_matrix(y_true, y_pred, labels=['left', 'right'])
        viz.plot_topomap(feature_map, ch_names)
        plt.show()
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        style: str = 'seaborn-v0_8',
        dpi: int = 300,
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except Exception:
            logger.warning(f"Style '{style}' not available, using default")
        
        set_chinese_font()
    
    def _save_figure(self, name: str, fig: plt.Figure) -> str:
        """Save figure and return path."""
        if self.output_dir:
            path = self.output_dir / f"{name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Figure saved: {path}")
            return str(path)
        return ""
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        val_history: Optional[Dict[str, List[float]]] = None,
        title: str = "Training Progress",
        metrics: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot training curves for loss and accuracy.
        
        Args:
            history: Dictionary with 'train_loss' and 'train_acc' lists
            val_history: Optional validation history dict
            title: Plot title
            metrics: List of metrics to plot
            save_name: Filename to save
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        n_plots = len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(history.get('train_loss', [0])) + 1)
        
        for ax, metric in zip(axes, metrics):
            if metric == 'loss':
                # Loss curve
                if 'train_loss' in history:
                    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
                if val_history and 'val_loss' in val_history:
                    ax.plot(epochs, val_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
                ax.set_ylabel('Loss', fontsize=12)
            elif metric == 'accuracy':
                # Accuracy curve
                if 'train_acc' in history:
                    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
                if val_history and 'val_acc' in val_history:
                    ax.plot(epochs, val_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
                ax.set_ylabel('Accuracy', fontsize=12)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_title(f'{title} - {metric.capitalize()}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multi_experiment_comparison(
        self,
        experiments: Dict[str, Dict[str, List[float]]],
        metric: str = 'val_acc',
        title: str = "Multi-Experiment Comparison",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare multiple experiments on same plot.
        
        Args:
            experiments: Dict mapping experiment name to history dict
            metric: Metric to compare
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        
        for (name, history), color in zip(experiments.items(), colors):
            epochs = range(1, len(history.get(metric, [0])) + 1)
            values = history.get(metric, [])
            
            if values:
                ax.plot(epochs, values, label=name, color=color, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        cmap: str = 'Blues',
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class names
            title: Plot title
            normalize: Whether to normalize rows
            cmap: Colormap name
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        
        n_classes = cm.shape[0]
        if labels is None:
            labels = [f'Class {i}' for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                text = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black', fontsize=12)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_topomap(
        self,
        data: np.ndarray,
        ch_names: Optional[List[str]] = None,
        title: str = "Topographic Map",
        cmap: str = 'RdBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_contours: bool = True,
        sensor_names: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot 2D topographic map of EEG channels.
        
        Args:
            data: Channel data values (n_channels,)
            ch_names: Channel names
            title: Plot title
            cmap: Colormap
            vmin, vmax: Color scale limits
            show_contours: Whether to show contour lines
            sensor_names: Alternative channel names
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        if ch_names is None and sensor_names is None:
            raise ValueError("Either ch_names or sensor_names must be provided")
        
        ch_names = ch_names or sensor_names
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get positions
        x, y, matched = get_channel_positions(ch_names)
        
        # Interpolate to grid
        n_grid = 100
        xi = np.linspace(-1.1, 1.1, n_grid)
        yi = np.linspace(-1.1, 1.1, n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        Zi = griddata((x, y), data, (Xi, Yi), method='cubic', fill_value=np.nan)
        
        # Mask outside head circle
        head_radius = 1.0
        mask = (Xi**2 + Yi**2) > head_radius**2
        Zi = np.ma.masked_where(mask, Zi)
        
        # Plot contour
        if show_contours:
            levels = np.linspace(vmin or Zi.min(), vmax or Zi.max(), 20)
            ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.8)
            ax.contour(Xi, Yi, Zi, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
        else:
            im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, shading='gouraud',
                              vmin=vmin, vmax=vmax)
        
        # Add colorbar
        if not show_contours:
            plt.colorbar(im, ax=ax, shrink=0.8, label='Value')
        
        # Plot head outline
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        
        # Plot nose indicator
        nose_x = [0, -0.1, 0, 0.1, 0]
        nose_y = [1.0, 1.1, 1.15, 1.1, 1.0]
        ax.plot(nose_x, nose_y, 'k-', linewidth=1.5)
        
        # Plot ear indicators
        ax.plot(-1.05, 0, 'k[', markersize=15, linewidth=1.5)
        ax.plot(1.05, 0, 'k]', markersize=15, linewidth=1.5)
        
        # Plot channel positions
        scatter = ax.scatter(x, y, c=data, cmap=cmap, s=80, edgecolors='black',
                            linewidths=1, vmin=vmin, vmax=vmax, zorder=5)
        
        # Add channel labels for key channels
        key_channels = ['Cz', 'C3', 'C4', 'FCz', 'CPz', 'Fz', 'Pz']
        for i, ch in enumerate(matched):
            if ch in key_channels:
                ax.annotate(ch, (x[i], y[i]), xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.1, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.7, label='Feature Value', pad=0.02)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_psd(
        self,
        X: np.ndarray,
        sfreq: float = 128,
        ch_names: Optional[List[str]] = None,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        title: str = "Power Spectral Density",
        average: bool = True,
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot power spectral density with frequency bands highlighted.
        
        Args:
            X: EEG data (n_epochs, n_channels, n_times) or (n_channels, n_times)
            sfreq: Sampling frequency
            ch_names: Channel names
            bands: Frequency band definitions
            title: Plot title
            average: Whether to average across epochs
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        if bands is None:
            bands = FREQ_BANDS
        
        # Handle multi-dimensional input
        if X.ndim == 3:
            if average:
                X = X.mean(axis=0)  # Average over epochs
            else:
                X = X[0]  # Take first epoch
        elif X.ndim == 2:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
        
        n_channels = X.shape[0]
        n_times = X.shape[1]
        
        # Compute PSD using Welch's method
        from scipy.signal import welch
        
        freqs, psd = welch(X, fs=sfreq, nperseg=min(256, n_times // 2),
                          noverlap=n_times // 4, axis=1)
        
        # Average across channels if many
        if n_channels > 10:
            psd_mean = psd.mean(axis=0)
            psd_std = psd.std(axis=0)
        else:
            psd_mean = psd.mean(axis=0)
            psd_std = psd.std(axis=0) if n_channels > 1 else np.zeros_like(psd_mean)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean PSD
        ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std,
                       alpha=0.3, color='steelblue')
        ax.plot(freqs, psd_mean, 'b-', linewidth=2, label='Mean PSD')
        
        # Highlight frequency bands
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                ax.axvspan(low, high, alpha=0.2, color=FREQ_BAND_COLORS.get(band_name, 'gray'),
                          label=f'{band_name.capitalize()} ({low}-{high} Hz)')
        
        # Add frequency band lines
        for band_name, (low, high) in bands.items():
            color = FREQ_BAND_COLORS.get(band_name, 'gray')
            ax.axvline(low, color=color, linestyle='--', alpha=0.5)
            ax.axvline(high, color=color, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power (µV²/Hz)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim([0, freqs.max()])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_erp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sfreq: float = 128,
        tmin: float = -1.0,
        tmax: float = 4.0,
        channels: Optional[List[str]] = None,
        title: str = "ERP Waveforms",
        classes: Optional[List[str]] = None,
        ci: float = 0.95,
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot event-related potentials with confidence intervals.
        
        Args:
            X: EEG data (n_epochs, n_channels, n_times)
            y: Labels (n_epochs,)
            sfreq: Sampling frequency
            tmin, tmax: Time window
            channels: Channel names for selection
            title: Plot title
            classes: Class names
            ci: Confidence interval (0-1)
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        n_epochs, n_channels, n_times = X.shape
        unique_classes = np.unique(y)
        
        if classes is None:
            classes = [f'Class {i}' for i in unique_classes]
        
        time_axis = np.linspace(tmin, tmax, n_times)
        n_plots = min(4, n_channels)  # Show up to 4 channels
        
        # Select channels to display
        if n_channels > n_plots:
            # Select motor-related channels by default
            display_chs = list(range(n_plots))
        else:
            display_chs = list(range(n_channels))
        
        n_subplot_cols = 2
        n_subplot_rows = (n_plots + n_subplot_cols - 1) // n_subplot_cols
        
        fig, axes = plt.subplots(n_subplot_rows, n_subplot_cols,
                                figsize=(12, 4 * n_subplot_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        
        for idx, ch_idx in enumerate(display_chs):
            ax = axes[idx]
            
            for class_idx, (cls, color) in enumerate(zip(unique_classes, colors)):
                # Get data for this class and channel
                mask = y == cls
                class_data = X[mask, ch_idx, :]
                
                # Compute mean and CI
                mean = class_data.mean(axis=0)
                std = class_data.std(axis=0)
                n = len(class_data)
                ci_value = stats.t.ppf((1 + ci) / 2, n - 1) * std / np.sqrt(n)
                
                # Plot
                ax.fill_between(time_axis, mean - ci_value, mean + ci_value,
                               alpha=0.3, color=color)
                ax.plot(time_axis, mean, color=color, linewidth=2,
                       label=classes[class_idx] if class_idx < len(classes) else f'Class {cls}')
            
            # Add vertical line at t=0
            ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            ch_name = channels[ch_idx] if channels and ch_idx < len(channels) else f'Ch {ch_idx}'
            ax.set_title(f'Channel: {ch_name}', fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude (µV)', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([tmin, tmax])
        
        # Hide unused subplots
        for idx in range(len(display_chs), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_cross_subject_comparison(
        self,
        subject_results: Dict[str, List[float]],
        metric: str = "Accuracy",
        title: str = "Cross-Subject Performance",
        plot_type: str = "boxplot",
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Compare performance across subjects using boxplot or violin plot.
        
        Args:
            subject_results: Dict mapping model name to list of subject accuracies
            metric: Metric name for y-axis
            title: Plot title
            plot_type: 'boxplot' or 'violin'
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = list(subject_results.keys())
        data = [subject_results[name] for name in model_names]
        
        if plot_type == 'boxplot':
            bp = ax.boxplot(data, labels=model_names, patch_artist=True)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            vp = ax.violinplot(data, positions=range(1, len(model_names) + 1),
                              showmeans=True, showmedians=True)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for body, color in zip(vp['bodies'], colors):
                body.set_facecolor(color)
                body.set_alpha(0.7)
        
        # Add individual points
        for i, values in enumerate(data):
            x = np.random.normal(i + 1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.5, s=30, c='black', zorder=3)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_tsne_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Feature Visualization",
        perplexity: float = 30.0,
        classes: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize high-dimensional features using t-SNE.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            title: Plot title
            perplexity: t-SNE perplexity
            classes: Class names
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Running t-SNE on {features.shape} features...")
        
        # Reduce dimensionality first if too high
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
        
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity,
                   random_state=42, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for class_idx, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = labels == label
            class_name = classes[label] if classes and label < len(classes) else f'Class {label}'
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[color], label=class_name, alpha=0.7, s=50,
                       edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_csp_patterns(
        self,
        spatial_filters: np.ndarray,
        ch_names: Optional[List[str]] = None,
        n_patterns: int = 4,
        title: str = "CSP Spatial Patterns",
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize Common Spatial Pattern (CSP) spatial patterns.
        
        Args:
            spatial_filters: CSP spatial filters (n_channels, n_patterns)
            ch_names: Channel names
            n_patterns: Number of patterns to show (pairs)
            title: Plot title
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        n_channels = spatial_filters.shape[0]
        n_to_show = min(n_patterns * 2, spatial_filters.shape[1])
        
        fig, axes = plt.subplots(1, n_to_show, figsize=(3 * n_to_show, 3))
        if n_to_show == 1:
            axes = [axes]
        
        for i in range(n_to_show):
            pattern = spatial_filters[:, i]
            ax = axes[i]
            
            # Normalize pattern
            pattern_normalized = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            
            # Plot as 1D bar plot with channel ordering
            x_pos = np.arange(n_channels)
            colors = plt.cm.RdBu_r(pattern_normalized)
            ax.bar(x_pos, pattern, color=colors, width=0.8)
            
            ax.set_title(f'Pattern {i + 1}', fontsize=11)
            ax.set_xlabel('Channel Index', fontsize=9)
            ax.set_ylabel('Weight', fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(save_name, fig)
        
        if show:
            plt.show()
        
        return fig
    
    def create_dashboard(
        self,
        history: Dict[str, List[float]],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        title: str = "BCI Analysis Dashboard",
        save_name: str = "dashboard",
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            history: Training history
            y_true: True labels
            y_pred: Predicted labels
            features: Feature matrix for t-SNE
            labels: Class labels
            title: Dashboard title
            save_name: Filename to save
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Training loss (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(history.get('train_loss', [0])) + 1)
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training accuracy (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'train_acc' in history:
            ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion matrix (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax3.imshow(cm_norm, cmap='Blues')
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        plt.colorbar(im, ax=ax3)
        
        # 4. t-SNE (middle-left)
        ax4 = fig.add_subplot(gs[1, :2])
        if features is not None:
            try:
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                features_2d = tsne.fit_transform(features[:500])  # Limit samples
                labels_subset = labels[:500] if labels is not None else y_true[:500]
                
                unique = np.unique(labels_subset)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
                for idx, (label, color) in enumerate(zip(unique, colors)):
                    mask = labels_subset == label
                    ax4.scatter(features_2d[mask, 0], features_2d[mask, 1],
                               c=[color], label=f'Class {label}', alpha=0.6, s=30)
                ax4.legend()
            except Exception as e:
                ax4.text(0.5, 0.5, f't-SNE failed: {e}', ha='center', va='center')
        ax4.set_title('t-SNE Feature Visualization')
        ax4.grid(True, alpha=0.3)
        
        # 5. Class distribution (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        unique, counts = np.unique(y_true, return_counts=True)
        ax5.bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
        ax5.set_title('Class Distribution')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Count')
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Summary stats (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        stats_text = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                    BCI Performance Summary                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Overall Accuracy:     {acc:.4f} ({acc*100:.2f}%)                           ║
        ║  Cohen's Kappa:         {kappa:.4f}                                   ║
        ║  Macro F1 Score:        {f1:.4f}                                   ║
        ║  Total Samples:         {len(y_true)}                                   ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
                fontsize=12, family='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        self._save_figure(save_name, fig)
        plt.show()
        
        return fig


def save_training_results(
    results: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Save training results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {save_path}")


def load_training_results(load_path: str) -> Dict[str, Any]:
    """
    Load training results from JSON file.
    
    Args:
        load_path: Path to JSON file
        
    Returns:
        Results dictionary
    """
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {load_path}")
    return results


# Export
__all__ = [
    "BCIVisualizer",
    "STANDARD_10_20_2D",
    "FREQ_BANDS",
    "FREQ_BAND_COLORS",
    "MOTOR_IMAGERY_LABELS",
    "get_channel_positions",
    "set_chinese_font",
    "save_training_results",
    "load_training_results",
]
