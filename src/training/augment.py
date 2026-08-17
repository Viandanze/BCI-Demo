"""
EEG Data Augmentation for Motor Imagery Classification
Implements various augmentation strategies to improve model generalization
"""

import logging
import random
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for EEG augmentation."""
    enabled: bool = True
    probability: float = 0.5  # Base probability of applying any augmentation
    
    # Individual augmentation settings
    temporal_mask: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.3,
        'mask_ratio': 0.3,
        'mask_value': 0.0,
    })
    
    channel_mask: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.2,
        'mask_ratio': 0.15,
        'mask_value': 0.0,
    })
    
    gaussian_noise: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.3,
        'snr_db': 10,
    })
    
    time_shift: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.2,
        'max_shift_samples': 20,
    })
    
    band_perturbation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.2,
        'band': (8, 30),
        'perturbation_scale': 0.1,
    })
    
    mixup: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'prob': 0.3,
        'alpha': 0.2,
    })


class EEGAugmentor:
    """
    EEG data augmentor for motor imagery classification.
    
    Provides multiple augmentation strategies that can be applied
    individually or in combination. All augmentations preserve
    the temporal structure and frequency content of the EEG signal.
    
    Supported augmentations:
        - Temporal Masking: Randomly mask consecutive time points
        - Channel Masking: Randomly mask entire channels
        - Gaussian Noise: Add SNR-controlled Gaussian noise
        - Time Shift: Small temporal translation
        - Band Perturbation: Perturb specific frequency bands
        - Mixup: Mix samples within/between classes
    
    Args:
        config: Augmentation configuration
        sfreq: Sampling frequency (Hz)
        random_state: Random seed for reproducibility
        
    Example:
        augmentor = EEGAugmentor(config, sfreq=128)
        
        # Augment single sample
        X_aug = augmentor.augment(X)
        
        # Augment batch
        X_aug, y_aug = augmentor.augment_batch(X, y)
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        sfreq: float = 128.0,
        random_state: Optional[int] = None,
    ):
        self.config = config or AugmentationConfig()
        self.sfreq = sfreq
        self.rng = np.random.RandomState(random_state)
        
        if random_state is not None:
            random.seed(random_state)
    
    def augment(self, X: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single sample or batch.
        
        Args:
            X: EEG data of shape (n_channels, n_times) or (batch, n_channels, n_times)
            
        Returns:
            Augmented data
        """
        if not self.config.enabled:
            return X.copy()
        
        # Ensure batch dimension
        if X.ndim == 2:
            X = X[np.newaxis, ...]
            single_sample = True
        else:
            single_sample = False
        
        X_aug = X.copy()
        
        # Apply augmentations with probability
        for i in range(len(X_aug)):
            if self.rng.random() > self.config.probability:
                continue
            
            # Apply each augmentation based on its individual probability
            if self.config.temporal_mask.get('enabled', False):
                X_aug[i] = self._temporal_mask(X_aug[i])
            
            if self.config.channel_mask.get('enabled', False):
                X_aug[i] = self._channel_mask(X_aug[i])
            
            if self.config.gaussian_noise.get('enabled', False):
                X_aug[i] = self._gaussian_noise(X_aug[i])
            
            if self.config.time_shift.get('enabled', False):
                X_aug[i] = self._time_shift(X_aug[i])
            
            if self.config.band_perturbation.get('enabled', False):
                X_aug[i] = self._band_perturbation(X_aug[i])
        
        return X_aug[0] if single_sample else X_aug
    
    def augment_batch(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to a batch with optional mixup.
        
        Args:
            X: Batch data (batch, channels, times)
            y: Labels (batch,)
            
        Returns:
            Tuple of (augmented X, augmented y if provided)
        """
        if not self.config.enabled:
            return X, y
        
        X_aug = self.augment(X)
        
        # Apply mixup
        if self.config.mixup.get('enabled', False) and y is not None:
            X_aug, y_aug = self._mixup(X_aug, y)
            return X_aug, y_aug
        
        return X_aug, y
    
    def _temporal_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Apply temporal masking.
        
        Randomly masks a continuous segment of time points with zeros
        or a constant value. This encourages the model to be robust
        to missing temporal information.
        
        Args:
            X: Data (channels, times)
            
        Returns:
            Masked data
        """
        cfg = self.config.temporal_mask
        
        if self.rng.random() > cfg.get('prob', 0.3):
            return X
        
        n_times = X.shape[-1]
        mask_ratio = cfg.get('mask_ratio', 0.3)
        mask_value = cfg.get('mask_value', 0.0)
        
        # Random segment length
        mask_length = int(n_times * mask_ratio)
        mask_length = max(1, mask_length)
        
        # Random start position
        max_start = n_times - mask_length
        start = self.rng.randint(0, max(1, max_start))
        
        # Apply mask
        X_aug = X.copy()
        X_aug[..., start:start + mask_length] = mask_value
        
        return X_aug
    
    def _channel_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Apply channel masking.
        
        Randomly masks entire channels with zeros or mean values.
        Similar to dropout but at the channel level.
        
        Args:
            X: Data (channels, times)
            
        Returns:
            Masked data
        """
        cfg = self.config.channel_mask
        
        if self.rng.random() > cfg.get('prob', 0.2):
            return X
        
        n_channels = X.shape[0]
        mask_ratio = cfg.get('mask_ratio', 0.15)
        mask_value = cfg.get('mask_value', 0.0)
        
        # Number of channels to mask
        n_mask = max(1, int(n_channels * mask_ratio))
        
        # Random channel indices
        mask_indices = self.rng.choice(n_channels, n_mask, replace=False)
        
        # Apply mask
        X_aug = X.copy()
        for idx in mask_indices:
            if mask_value == 'mean':
                X_aug[idx] = X_aug[idx].mean()
            else:
                X_aug[idx] = mask_value
        
        return X_aug
    
    def _gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise with controlled SNR.
        
        Args:
            X: Data (channels, times)
            
        Returns:
            Noisy data
        """
        cfg = self.config.gaussian_noise
        
        if self.rng.random() > cfg.get('prob', 0.3):
            return X
        
        snr_db = cfg.get('snr_db', 10)
        
        # Calculate signal power
        signal_power = np.mean(X ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = self.rng.randn(*X.shape) * np.sqrt(noise_power)
        
        return X + noise
    
    def _time_shift(self, X: np.ndarray) -> np.ndarray:
        """
        Apply small temporal shift.
        
        Args:
            X: Data (channels, times)
            
        Returns:
            Shifted data (zero-padded if needed)
        """
        cfg = self.config.time_shift
        
        if self.rng.random() > cfg.get('prob', 0.2):
            return X
        
        max_shift = cfg.get('max_shift_samples', 20)
        
        # Random shift amount
        shift = self.rng.randint(-max_shift, max_shift + 1)
        
        X_aug = np.roll(X, shift, axis=-1)
        
        # Zero out rolled-in values
        if shift > 0:
            X_aug[..., :shift] = 0
        elif shift < 0:
            X_aug[..., shift:] = 0
        
        return X_aug
    
    def _band_perturbation(self, X: np.ndarray) -> np.ndarray:
        """
        Perturb specific frequency bands.
        
        Adds small perturbations to specific frequency components,
        useful for motor imagery (mu/beta bands: 8-30 Hz).
        
        Args:
            X: Data (channels, times)
            
        Returns:
            Perturbed data
        """
        cfg = self.config.band_perturbation
        
        if self.rng.random() > cfg.get('prob', 0.2):
            return X
        
        band = cfg.get('band', (8, 30))
        scale = cfg.get('perturbation_scale', 0.1)
        
        # Apply bandpass filter to get band-limited signal
        from scipy import signal
        
        nyquist = self.sfreq / 2
        low = band[0] / nyquist
        high = band[1] / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filter to extract band
        try:
            band_signal = signal.filtfilt(b, a, X, axis=-1)
            
            # Add scaled perturbation
            perturbation = band_signal * scale
            X_perturbed = X + perturbation
            
            return X_perturbed
        except Exception:
            return X
    
    def _mixup(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mixup augmentation.
        
        Creates new training samples by linearly interpolating
        between random pairs of samples and their labels.
        
        Args:
            X: Batch data (batch, channels, times)
            y: Labels (batch,)
            
        Returns:
            Mixed data and labels
        """
        cfg = self.config.mixup
        
        if self.rng.random() > cfg.get('prob', 0.3):
            return X, y
        
        alpha = cfg.get('alpha', 0.2)
        batch_size = len(X)
        
        # Generate mixing coefficients
        lam = self.rng.beta(alpha, alpha, batch_size)
        
        # Random permutation
        indices = self.rng.permutation(batch_size)
        
        # Mix samples
        X_mixed = np.zeros_like(X)
        for i in range(batch_size):
            X_mixed[i] = lam[i] * X[i] + (1 - lam[i]) * X[indices[i]]
        
        # Mix labels (one-hot)
        y_onehot = np.zeros((batch_size, len(np.unique(y))))
        for i, label in enumerate(y):
            y_onehot[i, label] = 1
        
        y_mixed = np.zeros_like(y_onehot)
        for i in range(batch_size):
            y_mixed[i] = lam[i] * y_onehot[i] + (1 - lam[i]) * y_onehot[indices[i]]
        
        # Convert back to soft labels
        y_float = y_mixed
        
        return X_mixed, y_float
    
    def get_augmentation_summary(self) -> str:
        """Get a summary of enabled augmentations."""
        lines = ["EEG Augmentation Pipeline:", "=" * 40]
        
        if not self.config.enabled:
            lines.append("  All augmentations disabled")
            return "\n".join(lines)
        
        augs = []
        
        if self.config.temporal_mask.get('enabled'):
            augs.append(f"  - Temporal Mask: prob={self.config.temporal_mask['prob']}")
        
        if self.config.channel_mask.get('enabled'):
            augs.append(f"  - Channel Mask: prob={self.config.channel_mask['prob']}")
        
        if self.config.gaussian_noise.get('enabled'):
            augs.append(f"  - Gaussian Noise: SNR={self.config.gaussian_noise['snr_db']}dB")
        
        if self.config.time_shift.get('enabled'):
            augs.append(f"  - Time Shift: ±{self.config.time_shift['max_shift_samples']} samples")
        
        if self.config.band_perturbation.get('enabled'):
            band = self.config.band_perturbation['band']
            augs.append(f"  - Band Perturbation: {band[0]}-{band[1]} Hz")
        
        if self.config.mixup.get('enabled'):
            augs.append(f"  - Mixup: alpha={self.config.mixup['alpha']}")
        
        lines.append(f"Base probability: {self.config.probability}")
        lines.extend(augs if augs else ["  No augmentations enabled"])
        
        return "\n".join(lines)


def get_augmentation_pipeline(
    augmentations: List[str],
    sfreq: float = 128.0,
    random_state: Optional[int] = None,
) -> EEGAugmentor:
    """
    Factory function to create augmentor with specific methods.
    
    Args:
        augmentations: List of augmentation names to enable
        sfreq: Sampling frequency
        random_state: Random seed
        
    Returns:
        Configured EEGAugmentor
    """
    config = AugmentationConfig(
        enabled=True,
        temporal_mask={'enabled': 'temporal_mask' in augmentations},
        channel_mask={'enabled': 'channel_mask' in augmentations},
        gaussian_noise={'enabled': 'gaussian_noise' in augmentations},
        time_shift={'enabled': 'time_shift' in augmentations},
        band_perturbation={'enabled': 'band_perturbation' in augmentations},
        mixup={'enabled': 'mixup' in augmentations},
    )
    
    return EEGAugmentor(config=config, sfreq=sfreq, random_state=random_state)


def quick_augment(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'all',
    sfreq: float = 128.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick augmentation for a batch.
    
    Args:
        X: Input data
        y: Labels
        method: 'all', 'noise', 'mask', 'shift', 'mixup'
        sfreq: Sampling frequency
        
    Returns:
        Augmented (X, y)
    """
    aug_map = {
        'all': ['temporal_mask', 'channel_mask', 'gaussian_noise', 'time_shift'],
        'noise': ['gaussian_noise'],
        'mask': ['temporal_mask', 'channel_mask'],
        'shift': ['time_shift'],
        'mixup': ['mixup'],
    }
    
    methods = aug_map.get(method, aug_map['all'])
    augmentor = get_augmentation_pipeline(methods, sfreq=sfreq)
    
    return augmentor.augment_batch(X, y)


# Export
__all__ = [
    "EEGAugmentor",
    "AugmentationConfig",
    "get_augmentation_pipeline",
    "quick_augment",
]
