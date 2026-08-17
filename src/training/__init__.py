"""
Training module - training loops, data augmentation, and ensemble learning
"""

from .trainer import Trainer, create_optimizer, set_seed
from .augment import EEGAugmentor, get_augmentation_pipeline
from .ensemble import (
    ModelConfig,
    EnsembleConfig,
    EnsembleResult,
    BaseEnsemble,
    VotingEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
    BaggingEnsemble,
    AdaptiveEnsemble,
    create_ensemble,
)

__all__ = [
    # Trainer
    "Trainer",
    "create_optimizer",
    "set_seed",
    
    # Augmentation
    "EEGAugmentor",
    "get_augmentation_pipeline",
    
    # Ensemble
    "ModelConfig",
    "EnsembleConfig",
    "EnsembleResult",
    "BaseEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "WeightedEnsemble",
    "BaggingEnsemble",
    "AdaptiveEnsemble",
    "create_ensemble",
]
