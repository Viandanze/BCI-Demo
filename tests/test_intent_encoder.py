"""
Tests for IntentEncoder module.

Covers:
  - Cognitive mode mapping (4 MI classes → 4 cognitive modes)
  - Debounce logic (requires N consecutive identical predictions)
  - Confidence threshold filtering
  - Rest state detection
  - Selection encoding (candidate index mapping)
  - Reset behavior
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent.intent_encoder import (
    IntentEncoder,
    CognitiveMode,
    DecoderLabel,
    Intent,
)


class TestCognitiveMode:
    """Test CognitiveMode enum properties."""

    def test_mode_values(self):
        assert CognitiveMode.QUERY.value == "query"
        assert CognitiveMode.REASON.value == "reason"
        assert CognitiveMode.CREATE.value == "create"
        assert CognitiveMode.REVIEW.value == "review"

    def test_mode_descriptions(self):
        assert "knowledge" in CognitiveMode.QUERY.description.lower()
        assert "reason" in CognitiveMode.REASON.description.lower()
        assert "creat" in CognitiveMode.CREATE.description.lower()
        assert "summari" in CognitiveMode.REVIEW.description.lower()


class TestDecoderLabel:
    """Test DecoderLabel enum values."""

    def test_label_values(self):
        assert DecoderLabel.LEFT_HAND.value == 0
        assert DecoderLabel.RIGHT_HAND.value == 1
        assert DecoderLabel.FEET.value == 2
        assert DecoderLabel.TONGUE.value == 3
        assert DecoderLabel.REST.value == 4


class TestIntent:
    """Test Intent dataclass."""

    def test_is_confident_true(self):
        intent = Intent(
            mode=CognitiveMode.QUERY,
            confidence=0.8,
            raw_label=0,
            raw_probabilities=[0.8, 0.1, 0.05, 0.05, 0.0],
        )
        assert intent.is_confident is True

    def test_is_confident_false(self):
        intent = Intent(
            mode=CognitiveMode.QUERY,
            confidence=0.4,
            raw_label=0,
            raw_probabilities=[0.4, 0.3, 0.2, 0.1, 0.0],
        )
        assert intent.is_confident is False

    def test_is_confident_boundary(self):
        """Confidence exactly at threshold (0.6) should be confident."""
        intent = Intent(
            mode=CognitiveMode.QUERY,
            confidence=0.6,
            raw_label=0,
            raw_probabilities=[0.6, 0.2, 0.1, 0.1, 0.0],
        )
        assert intent.is_confident is True

    def test_to_dict(self):
        intent = Intent(
            mode=CognitiveMode.CREATE,
            confidence=0.75,
            raw_label=2,
            raw_probabilities=[0.1, 0.1, 0.75, 0.05, 0.0],
        )
        d = intent.to_dict()
        assert d["mode"] == "create"
        assert d["confidence"] == 0.75
        assert d["raw_label"] == 2
        assert "mode_label" in d
        assert "description" in d

    def test_mode_labels(self):
        for mode in CognitiveMode:
            intent = Intent(
                mode=mode,
                confidence=0.9,
                raw_label=0,
                raw_probabilities=[0.9, 0.05, 0.03, 0.02, 0.0],
            )
            assert intent.mode_label  # Non-empty


class TestIntentEncoderDebounce:
    """Test debounce logic."""

    def test_insufficient_frames(self):
        """Should return None when not enough debounce frames."""
        encoder = IntentEncoder(debounce_frames=3)
        probs = [0.9, 0.05, 0.03, 0.02, 0.0]

        # First frame: not enough
        result = encoder.encode(label=0, probabilities=probs)
        assert result is None

        # Second frame: still not enough
        result = encoder.encode(label=0, probabilities=probs)
        assert result is None

    def test_locks_after_n_frames(self):
        """Should lock intent after N consecutive identical predictions."""
        encoder = IntentEncoder(debounce_frames=3, confidence_threshold=0.5)
        probs = [0.9, 0.05, 0.03, 0.02, 0.0]

        encoder.encode(label=0, probabilities=probs)  # Frame 1
        encoder.encode(label=0, probabilities=probs)  # Frame 2
        result = encoder.encode(label=0, probabilities=probs)  # Frame 3

        assert result is not None
        assert result.mode == CognitiveMode.QUERY
        assert result.confidence == 0.9

    def test_resets_on_different_label(self):
        """Debounce counter resets when a different label appears."""
        encoder = IntentEncoder(debounce_frames=3, confidence_threshold=0.5)
        high_probs = [0.9, 0.05, 0.03, 0.02, 0.0]

        encoder.encode(label=0, probabilities=high_probs)  # Frame 1: left
        encoder.encode(label=0, probabilities=high_probs)  # Frame 2: left
        encoder.encode(label=1, probabilities=high_probs)  # Frame 3: right (breaks streak)
        result = encoder.encode(label=1, probabilities=high_probs)  # Frame 4: right
        # Only 2 consecutive right, need 3
        assert result is None


class TestIntentEncoderMapping:
    """Test MI class to cognitive mode mapping."""

    @pytest.fixture
    def encoder(self):
        return IntentEncoder(debounce_frames=1, confidence_threshold=0.5)

    def test_left_hand_to_query(self, encoder):
        probs = [0.9, 0.05, 0.03, 0.02, 0.0]
        result = encoder.encode(label=0, probabilities=probs)
        assert result.mode == CognitiveMode.QUERY

    def test_right_hand_to_reason(self, encoder):
        probs = [0.05, 0.9, 0.03, 0.02, 0.0]
        result = encoder.encode(label=1, probabilities=probs)
        assert result.mode == CognitiveMode.REASON

    def test_feet_to_create(self, encoder):
        probs = [0.05, 0.03, 0.9, 0.02, 0.0]
        result = encoder.encode(label=2, probabilities=probs)
        assert result.mode == CognitiveMode.CREATE

    def test_tongue_to_review(self, encoder):
        probs = [0.05, 0.03, 0.02, 0.9, 0.0]
        result = encoder.encode(label=3, probabilities=probs)
        assert result.mode == CognitiveMode.REVIEW


class TestIntentEncoderConfidence:
    """Test confidence threshold filtering."""

    def test_low_confidence_rejected(self):
        encoder = IntentEncoder(debounce_frames=1, confidence_threshold=0.6)
        probs = [0.5, 0.3, 0.1, 0.05, 0.05]  # 0.5 < 0.6
        result = encoder.encode(label=0, probabilities=probs)
        assert result is None

    def test_high_confidence_accepted(self):
        encoder = IntentEncoder(debounce_frames=1, confidence_threshold=0.6)
        probs = [0.8, 0.1, 0.05, 0.03, 0.02]
        result = encoder.encode(label=0, probabilities=probs)
        assert result is not None
        assert result.confidence == 0.8


class TestIntentEncoderRest:
    """Test rest state detection."""

    def test_rest_label_returns_none(self):
        encoder = IntentEncoder(debounce_frames=1, confidence_threshold=0.5)
        probs = [0.05, 0.05, 0.05, 0.05, 0.8]  # Rest = label 4
        result = encoder.encode(label=4, probabilities=probs)
        assert result is None


class TestIntentEncoderSelection:
    """Test candidate selection encoding."""

    def test_valid_selection(self):
        encoder = IntentEncoder(debounce_frames=1, confidence_threshold=0.5)
        probs = [0.05, 0.9, 0.03, 0.02, 0.0]  # label 1 has 0.9 confidence
        result = encoder.encode_selection(
            label=1, probabilities=probs, n_candidates=3
        )
        assert result == 1

    def test_selection_out_of_range(self):
        encoder = IntentEncoder(debounce_frames=1, confidence_threshold=0.5)
        probs = [0.05, 0.05, 0.05, 0.9, 0.0]
        # Label 3 but only 3 candidates (0,1,2)
        result = encoder.encode_selection(
            label=3, probabilities=probs, n_candidates=3
        )
        assert result is None


class TestIntentEncoderReset:
    """Test reset behavior."""

    def test_reset_clears_history(self):
        encoder = IntentEncoder(debounce_frames=3, confidence_threshold=0.5)
        probs = [0.9, 0.05, 0.03, 0.02, 0.0]

        encoder.encode(label=0, probabilities=probs)
        encoder.encode(label=0, probabilities=probs)
        encoder.reset()

        # After reset, need 3 more frames
        encoder.encode(label=0, probabilities=probs)
        encoder.encode(label=0, probabilities=probs)
        result = encoder.encode(label=0, probabilities=probs)
        assert result is not None
