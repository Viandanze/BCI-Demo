"""Mock EEG decoder for development without trained model."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MockDecoder:
    """
    Mock EEG decoder for development without trained model.

    Uses simple frequency-band power features to classify motor imagery.
    Not accurate, but produces varied outputs for testing the full pipeline.
    """

    def __init__(self, n_classes: int = 4, sample_rate: int = 250):
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        self._call_count = 0

    def predict(self, eeg_data: np.ndarray) -> tuple[int, list[float]]:
        """Predict motor imagery class from EEG data.

        Args:
            eeg_data: shape (n_channels, n_samples)

        Returns:
            (predicted_label, probabilities)
        """
        self._call_count += 1

        from scipy.signal import welch

        n_channels, n_samples = eeg_data.shape
        features = []

        for ch in range(n_channels):
            freqs, psd = welch(eeg_data[ch, :], fs=self.sample_rate, nperseg=min(256, n_samples))
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.mean(psd[(freqs >= 13) & (freqs < 30)])
            features.append(alpha_power / (beta_power + 1e-10))

        mean_ratio = np.mean(features)

        raw_scores = np.array([
            mean_ratio,
            1.0 / (mean_ratio + 1e-10),
            np.std(features),
            1.0 - mean_ratio,
        ])

        raw_scores += np.random.normal(0, 0.1, size=4)

        raw_scores = raw_scores - np.max(raw_scores)
        probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))

        label = int(np.argmax(probabilities))
        return label, probabilities.tolist()
