"""
EEG Preprocessing Pipeline for Motor Imagery Classification
Implements filtering, artifact removal, and epoching
"""

import logging
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, field

import mne
import numpy as np
from mne import Epochs
from mne.io import Raw

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Filtering
    bandpass_low: float = 4.0          # Low cutoff frequency (Hz)
    bandpass_high: float = 38.0        # High cutoff frequency (Hz)
    notch_freq: Optional[float] = 60.0 # Notch filter frequency (Hz)
    
    # ICA
    apply_ica: bool = False
    ica_n_components: int = 20
    ica_exclude: List[int] = field(default_factory=list)
    
    # Epoching
    tmin: float = -1.0                 # Start time (s)
    tmax: float = 4.0                 # End time (s)
    baseline: Optional[Tuple[float, float]] = (-1.0, 0.0)
    
    # Resampling
    resample_freq: Optional[float] = 128.0
    
    # Normalization
    normalize: bool = True
    
    # Channel selection
    picks: Optional[Union[str, List[str]]] = None  # 'eeg', 'meg', or list of names
    
    # Rejection criteria
    reject_threshold: Optional[Dict[str, float]] = None
    reject_eeg: Optional[float] = 500e-6  # 500 µV threshold (relaxed for PhysioNet)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for EEG motor imagery data.
    
    Implements a configurable sequence of preprocessing steps:
    1. Channel selection
    2. Band-pass filtering
    3. Notch filtering
    4. ICA artifact removal (optional)
    5. Epoching
    6. Baseline correction
    7. Resampling (optional)
    8. Normalization (optional)
    
    Example:
        config = PreprocessingConfig(bandpass_low=4, bandpass_high=38)
        pipeline = PreprocessingPipeline(config)
        
        # Process raw data
        epochs = pipeline.process_raw(raw, events, event_id)
        
        # Or use shortcut
        X, y = preprocess_epochs(raw, events, event_id, config)
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        verbose: bool = False,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration. Uses defaults if None.
            verbose: Enable verbose output.
        """
        self.config = config or PreprocessingConfig()
        self.verbose = verbose
        self._ica = None
        self._scaler = None
        self._fitted = False
    
    def process_raw(
        self,
        raw: Raw,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
    ) -> Epochs:
        """
        Apply full preprocessing pipeline to Raw data.
        
        Args:
            raw: Raw EEG data
            events: Event array (n_events, 3) if not using annotations
            event_id: Event ID mapping if using events
            
        Returns:
            Preprocessed Epochs object
        """
        logger.info("Starting preprocessing pipeline...")
        raw = raw.copy()
        
        # Step 1: Channel selection
        raw = self._select_channels(raw)
        
        # Step 2: Band-pass filtering
        raw = self._apply_bandpass(raw)
        
        # Step 3: Notch filtering
        raw = self._apply_notch(raw)
        
        # Step 4: ICA (optional)
        if self.config.apply_ica:
            raw = self._apply_ica(raw)
        
        # Step 5: Epoching
        epochs = self._create_epochs(raw, events, event_id)
        
        # Step 6: Baseline correction
        epochs = self._apply_baseline(epochs)
        
        # Step 7: Resampling
        epochs = self._resample(epochs)
        
        # Step 8: Normalization
        if self.config.normalize:
            epochs = self._normalize(epochs)
        
        self._fitted = True
        logger.info(f"Preprocessing complete: {len(epochs)} epochs, "
                   f"shape {epochs.get_data().shape}")
        
        return epochs
    
    def _select_channels(self, raw: Raw) -> Raw:
        """Select and set up channels."""
        picks = self.config.picks
        
        if picks is None:
            logger.info("Using all available channels")
            return raw
        
        try:
            if isinstance(picks, str):
                raw = raw.pick(picks)
                logger.info(f"Selected channel type: {picks}")
            elif isinstance(picks, list):
                # Verify channels exist
                available = set(raw.ch_names)
                selected = [p for p in picks if p in available]
                missing = set(picks) - set(selected)
                
                if missing:
                    logger.warning(f"Channels not found: {missing}")
                
                if selected:
                    raw = raw.pick(selected)
                    logger.info(f"Selected {len(selected)} channels")
                else:
                    logger.warning("No channels selected, using all")
            
            # Set standard montage
            try:
                raw.set_montage('standard_1005', on_missing='warn')
            except Exception as e:
                logger.warning(f"Could not set montage: {e}")
                
        except Exception as e:
            logger.error(f"Channel selection failed: {e}")
        
        return raw
    
    def _apply_bandpass(self, raw: Raw) -> Raw:
        """Apply band-pass filter."""
        low = self.config.bandpass_low
        high = self.config.bandpass_high
        
        logger.info(f"Applying bandpass filter: {low}-{high} Hz")
        
        try:
            raw.filter(
                l_freq=low,
                h_freq=high,
                fir_design='firwin',
                skip_by_annotation='edge',
                verbose=self.verbose,
            )
        except Exception as e:
            logger.error(f"Bandpass filtering failed: {e}")
            raise
        
        return raw
    
    def _apply_notch(self, raw: Raw) -> Raw:
        """Apply notch filter to remove line noise."""
        notch_freq = self.config.notch_freq
        
        if notch_freq is None:
            logger.info("Skipping notch filter")
            return raw
        
        logger.info(f"Applying notch filter: {notch_freq} Hz")
        
        try:
            raw.notch_filter(
                freqs=notch_freq,
                fir_design='firwin',
                skip_by_annotation='edge',
                verbose=self.verbose,
            )
        except Exception as e:
            logger.warning(f"Notch filtering failed: {e}")
        
        return raw
    
    def _apply_ica(self, raw: Raw) -> Raw:
        """Apply ICA for artifact removal."""
        logger.info("Applying ICA for artifact removal...")
        
        try:
            from mne.preprocessing import ICA, create_eog_epochs
            
            # Fit ICA
            ica = ICA(
                n_components=self.config.ica_n_components,
                random_state=42,
                max_iter=500,
                verbose=self.verbose,
            )
            
            ica.fit(raw, verbose=self.verbose)
            self._ica = ica
            
            # Detect bad components (eye blinks, muscle artifacts)
            # This is a simplified version; more sophisticated methods available
            if len(self.config.ica_exclude) == 0:
                # Auto-detect EOG artifacts
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=self.verbose)
                    exclude = eog_indices[:2]  # Exclude top 2 EOG components
                    logger.info(f"Auto-detected EOG components: {exclude}")
                except Exception:
                    exclude = []
                    logger.warning("Could not auto-detect EOG components")
                
                self.config.ica_exclude = exclude
            
            # Apply ICA
            raw = ica.apply(raw, exclude=self.config.ica_exclude, verbose=self.verbose)
            logger.info(f"ICA applied, excluded components: {self.config.ica_exclude}")
            
        except Exception as e:
            logger.warning(f"ICA processing failed: {e}, continuing without ICA")
            self.config.apply_ica = False
        
        return raw
    
    def _create_epochs(
        self,
        raw: Raw,
        events: Optional[np.ndarray],
        event_id: Optional[Dict[str, int]],
    ) -> Epochs:
        """Create epochs from continuous data."""
        logger.info(f"Creating epochs: {self.config.tmin}s to {self.config.tmax}s")
        
        # Get events from annotations if not provided
        if events is None:
            events, event_id = mne.events_from_annotations(
                raw,
                verbose=self.verbose,
            )
        
        # Create epochs
        reject = self.config.reject_eeg
        if isinstance(reject, dict):
            reject = reject.get('eeg', None)
        elif isinstance(reject, (int, float)):
            # Convert float threshold to dict format MNE expects
            reject = {'eeg': float(reject)}
        else:
            reject = None
        
        try:
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=event_id,
                tmin=self.config.tmin,
                tmax=self.config.tmax,
                baseline=self.config.baseline,
                reject=reject,
                preload=True,
                verbose=self.verbose,
            )
            
            n_rejected = len(epochs.drop_log) - sum(1 for x in epochs.drop_log if len(x) == 0)
            if n_rejected > 0:
                logger.info(f"Dropped {n_rejected} epochs due to artifacts")
            
            if len(epochs) == 0:
                logger.warning("All epochs were dropped by reject criteria! Retrying without rejection.")
                epochs = mne.Epochs(
                    raw,
                    events=events,
                    event_id=event_id,
                    tmin=self.config.tmin,
                    tmax=self.config.tmax,
                    baseline=self.config.baseline,
                    reject=None,
                    preload=True,
                    verbose=self.verbose,
                )
                logger.info(f"Loaded {len(epochs)} epochs without rejection")
            
        except Exception as e:
            logger.error(f"Epoch creation failed: {e}")
            raise
        
        return epochs
    
    def _apply_baseline(self, epochs: Epochs) -> Epochs:
        """Apply baseline correction."""
        if self.config.baseline is None:
            logger.info("Skipping baseline correction")
            return epochs
        
        logger.info(f"Applying baseline correction: {self.config.baseline}")
        epochs.apply_baseline(self.config.baseline, verbose=self.verbose)
        return epochs
    
    def _resample(self, epochs: Epochs) -> Epochs:
        """Resample epochs to target frequency."""
        target_freq = self.config.resample_freq
        
        if target_freq is None:
            logger.info("Skipping resampling")
            return epochs
        
        current_freq = epochs.info['sfreq']
        if abs(current_freq - target_freq) < 1:
            logger.info(f"Already at target frequency: {current_freq} Hz")
            return epochs
        
        logger.info(f"Resampling from {current_freq} to {target_freq} Hz")
        epochs.resample(target_freq, npad='auto', verbose=self.verbose)
        return epochs
    
    def _normalize(self, epochs: Epochs) -> Epochs:
        """Apply channel-wise z-score normalization."""
        logger.info("Applying channel-wise normalization")
        
        data = epochs.get_data()
        
        # Compute mean and std across time for each channel
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True)
        std[std < 1e-8] = 1  # Avoid division by zero
        
        normalized_data = (data - mean) / std
        epochs._data = normalized_data
        
        return epochs
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted pipeline.
        
        Note: Only applies normalization which was fitted on training data.
        For full preprocessing, use process_raw() with the original pipeline.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call process_raw() first.")
        
        if self._scaler is not None:
            return self._scaler.transform(X)
        
        return X
    
    def get_pipeline_summary(self) -> str:
        """Get a summary string of the preprocessing steps."""
        steps = []
        steps.append(f"1. Channel selection: {self.config.picks or 'all'}")
        steps.append(f"2. Bandpass filter: {self.config.bandpass_low}-{self.config.bandpass_high} Hz")
        
        if self.config.notch_freq:
            steps.append(f"3. Notch filter: {self.config.notch_freq} Hz")
        else:
            steps.append("3. Notch filter: disabled")
        
        if self.config.apply_ica:
            steps.append(f"4. ICA: {self.config.ica_n_components} components")
            steps.append(f"   Excluded: {self.config.ica_exclude}")
        else:
            steps.append("4. ICA: disabled")
        
        steps.append(f"5. Epoching: {self.config.tmin}s to {self.config.tmax}s")
        steps.append(f"6. Baseline: {self.config.baseline}")
        
        if self.config.resample_freq:
            steps.append(f"7. Resampling: {self.config.resample_freq} Hz")
        
        steps.append(f"8. Normalization: {'enabled' if self.config.normalize else 'disabled'}")
        
        return "\n".join(steps)


def preprocess_epochs(
    raw: Raw,
    config: Optional[PreprocessingConfig] = None,
    events: Optional[np.ndarray] = None,
    event_id: Optional[Dict[str, int]] = None,
    return_epochs: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Epochs]:
    """
    Convenience function for preprocessing motor imagery epochs.
    
    Args:
        raw: Raw EEG data
        config: Preprocessing configuration
        events: Event array (optional)
        event_id: Event ID mapping (optional)
        return_epochs: If True, return Epochs object; otherwise return (X, y)
        
    Returns:
        Either Epochs object or tuple of (X, y) arrays
        
    Example:
        X, y = preprocess_epochs(raw)
        
        # Or with custom config
        config = PreprocessingConfig(bandpass_low=2, bandpass_high=40)
        epochs = preprocess_epochs(raw, config, return_epochs=True)
    """
    pipeline = PreprocessingPipeline(config)
    epochs = pipeline.process_raw(raw, events, event_id)
    
    if return_epochs:
        return epochs
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Convert to 0-indexed labels
    unique_labels = sorted(list(set(y)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    
    return X, y


# Export
__all__ = [
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "preprocess_epochs",
]
