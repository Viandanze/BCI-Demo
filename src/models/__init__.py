"""
Models module - neural network and classical ML models
"""

from .eegnet import EEGNet, EEGNetClassifier
from .csp import CSPClassifier, CSPFeatures
from .riemann_mdm import RiemannMDMClassifier

__all__ = [
    "EEGNet",
    "EEGNetClassifier",
    "CSPClassifier", 
    "CSPFeatures",
    "RiemannMDMClassifier",
]
