"""Real EEGNet decoder using trained model checkpoint."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RealDecoder:
    """
    Real EEGNet decoder using trained model checkpoint.

    Automatically handles the mismatch between BrainFlow Synthetic Board
    (N channels, 250Hz) and the trained EEGNet model (64 channels, 128Hz,
    4-5s window) by:
      - Resampling 250Hz -> 128Hz
      - Channel adaptation (repeat or truncate)
      - Bandpass filtering (4-38Hz)
      - Z-score normalization (per channel)

    The model architecture (n_channels, n_times) is inferred from the
    checkpoint state_dict, so it works regardless of training configuration.
    """

    MODEL_SAMPLE_RATE = 128.0
    BANDPASS_LOW = 4.0
    BANDPASS_HIGH = 38.0
    N_CLASSES = 4
    CLASS_LABELS = ['left_hand', 'right_hand', 'feet', 'rest']

    def __init__(self, model_path: Optional[str] = None, source_sample_rate: float = 250.0):
        import torch
        from scipy.signal import butter

        self.source_sample_rate = source_sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            import glob
            candidates = sorted(glob.glob("outputs/eegnet_s*/best_model.pt"))
            if not candidates:
                raise FileNotFoundError(
                    "No trained model found in outputs/. "
                    "Run: python scripts/train_eegnet.py --subjects 1 2 3 4 5 --epochs 50"
                )
            model_path = candidates[-1]

        logger.info(f"Loading EEGNet checkpoint: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        best_acc = checkpoint.get('best_accuracy', 'unknown')

        n_channels, n_times = self._infer_architecture(state_dict)
        logger.info(f"Model architecture: {n_channels}ch, {n_times} samples, "
                     f"best_acc={best_acc}")

        from src.models.eegnet import EEGNetClassifier
        self.model = EEGNetClassifier(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.N_CLASSES,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.model_channels = n_channels
        self.model_times = n_times

        nyq = self.MODEL_SAMPLE_RATE / 2.0
        self._sos = butter(
            4,
            [self.BANDPASS_LOW / nyq, self.BANDPASS_HIGH / nyq],
            btype='band',
            output='sos',
        )

        logger.info(f"RealDecoder ready (device={self.device}, "
                     f"model={n_channels}ch/{n_times}samples, "
                     f"source_rate={self.source_sample_rate}Hz)")

    def _infer_architecture(self, state_dict) -> tuple[int, int]:
        """Infer n_channels and n_times from checkpoint state_dict."""
        n_channels = 64
        n_times = 640

        for key, val in state_dict.items():
            if 'spatial' in key and 'weight' in key and val.ndim == 4:
                n_channels = val.shape[2]
                logger.debug(f"Inferred n_channels={n_channels} from {key}")
                break

        F1 = 8
        for key, val in state_dict.items():
            if 'temporal' in key and 'weight' in key and val.ndim == 4:
                F1 = val.shape[0]
                break

        D = 2
        pool_size = 4
        F2 = F1 * D

        for key, val in state_dict.items():
            if ('classifier' in key or 'fc' in key) and 'weight' in key and val.ndim == 2:
                if val.shape[0] == self.N_CLASSES:
                    pooled_time = val.shape[1] // F2
                    if pooled_time > 0:
                        n_times = pooled_time * (pool_size ** 2)
                        logger.debug(f"Inferred n_times={n_times} from {key} "
                                     f"(pooled_time={pooled_time}, F2={F2})")
                    break

        from src.models.eegnet import EEGNetClassifier
        verified = False
        for n_times_try in [n_times, n_times + 1, 640, 641, 512, 513]:
            try:
                test_model = EEGNetClassifier(
                    n_channels=n_channels,
                    n_times=n_times_try,
                    n_classes=self.N_CLASSES,
                )
                test_model.load_state_dict(state_dict, strict=True)
                n_times = n_times_try
                verified = True
                break
            except RuntimeError:
                continue

        if not verified:
            logger.warning(
                f"Could not verify model architecture. "
                f"Using n_channels={n_channels}, n_times={n_times}. "
                f"Model loading may fail."
            )

        return n_channels, n_times

    def _preprocess(self, eeg_data: np.ndarray, source_rate: float) -> np.ndarray:
        """Preprocess EEG data to match model input requirements.

        Steps:
          1. Resample to model's sampling rate (128Hz)
          2. Crop/pad to model's time window
          3. Bandpass filter (4-38Hz)
          4. Z-score normalization (per channel)
          5. Channel adaptation (repeat or truncate)
        """
        from scipy.signal import sosfiltfilt, resample

        if source_rate != self.MODEL_SAMPLE_RATE:
            n_target = int(eeg_data.shape[1] * self.MODEL_SAMPLE_RATE / source_rate)
            eeg_data = resample(eeg_data, n_target, axis=1)

        target_len = self.model_times
        if eeg_data.shape[1] > target_len:
            eeg_data = eeg_data[:, -target_len:]
        elif eeg_data.shape[1] < target_len:
            pad = np.zeros((eeg_data.shape[0], target_len - eeg_data.shape[1]))
            eeg_data = np.concatenate([pad, eeg_data], axis=1)

        eeg_data = sosfiltfilt(self._sos, eeg_data, axis=1)

        mean = eeg_data.mean(axis=1, keepdims=True)
        std = eeg_data.std(axis=1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-10)

        n_current = eeg_data.shape[0]
        if n_current < self.model_channels:
            repeats = (self.model_channels + n_current - 1) // n_current
            eeg_data = np.tile(eeg_data, (repeats, 1))[:self.model_channels, :]
        elif n_current > self.model_channels:
            eeg_data = eeg_data[:self.model_channels, :]

        return eeg_data.astype(np.float32)

    def predict(self, eeg_data: np.ndarray) -> tuple[int, list[float]]:
        """Predict motor imagery class from EEG data.

        Args:
            eeg_data: shape (n_channels, n_samples) at source sampling rate

        Returns:
            (predicted_label, probabilities)
        """
        import torch

        processed = self._preprocess(eeg_data, source_rate=self.source_sample_rate)
        tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        label = int(np.argmax(probabilities))
        return label, probabilities.tolist()
