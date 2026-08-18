"""Mock EEG decoder for development without trained model."""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class MockDecoder:
    """
    Mock EEG decoder for development without trained model.

    Strategy: Intent Rotation
      Synthetic Board signals have stable frequency characteristics,
      which makes any fixed feature-based scoring collapse to a single
      class. To simulate a real user switching between motor imagery
      tasks, the decoder rotates its "target intent" every
      ``rotation_interval`` seconds and boosts that class so the
      IntentEncoder debounce can lock onto it.

      - Target intent rotates in order: LH -> RH -> Feet -> Tongue -> ...
      - The target class gets a large score boost (+2.0), giving it
        ~90%+ softmax confidence (well above the 0.6 threshold).
      - Small Gaussian noise keeps probabilities non-degenerate.
    """

    #: Seconds before switching to the next simulated motor imagery task.
    ROTATION_INTERVAL = 30.0

    #: Score boost applied to the current target class.
    TARGET_BOOST = 2.0

    #: Class rotation order (PhysioNet 4-class MI labels).
    _ROTATION = [0, 1, 2, 3]

    def __init__(self, n_classes: int = 4, sample_rate: int = 250):
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        self._call_count = 0
        self._rotation_idx = 0
        self._rotation_start = time.time()

    def _current_target(self) -> int:
        """Return the current simulated intent, rotating on schedule."""
        now = time.time()
        if now - self._rotation_start >= self.ROTATION_INTERVAL:
            self._rotation_idx = (self._rotation_idx + 1) % len(self._ROTATION)
            self._rotation_start = now
            logger.debug(
                "MockDecoder: rotating simulated intent to label %d",
                self._ROTATION[self._rotation_idx],
            )
        return self._ROTATION[self._rotation_idx]

    def predict(self, eeg_data: np.ndarray) -> tuple[int, list[float]]:
        """Predict motor imagery class from EEG data.

        Args:
            eeg_data: shape (n_channels, n_samples)

        Returns:
            (predicted_label, probabilities)
        """
        self._call_count += 1
        target = self._current_target()

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

        # Normalize feature-derived scores so no single statistic dominates.
        raw_scores = raw_scores / (np.max(np.abs(raw_scores)) + 1e-10)

        # Simulate the user imagining the target motor task.
        raw_scores[target] += self.TARGET_BOOST

        # Small noise keeps the distribution non-degenerate.
        raw_scores += np.random.normal(0, 0.1, size=4)

        raw_scores = raw_scores - np.max(raw_scores)
        probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))

        label = int(np.argmax(probabilities))
        return label, probabilities.tolist()
