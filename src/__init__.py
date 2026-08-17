"""
BCI_Projects - Motor Imagery BCI Analysis Package
A complete toolkit for EEG-based Motor Imagery classification
"""

__version__ = "1.0.0"
__author__ = "UE5AssetAnalyzer Owner"

from .data import loader, preprocessing
from .models import eegnet, csp, riemann_mdm
from .training import trainer, augment
from .evaluation import metrics
from .utils import config

__all__ = [
    "loader",
    "preprocessing", 
    "eegnet",
    "csp",
    "riemann_mdm",
    "trainer",
    "augment",
    "metrics",
    "config",
]
