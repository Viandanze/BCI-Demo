"""
Riemannian MDM (Minimum Distance to Mean) Classifier using pyRiemann
Implements covariance-based classification using Riemannian geometry
"""

import logging
from typing import Tuple, Optional, Dict, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Try to import pyRiemann, with fallback for when not available
try:
    import pyriemann
    from pyriemann.classification import MDM, TSClassifier as TSclassifier
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False
    MDM = None
    TSclassifier = None
    Covariances = None
    TangentSpace = None
    logger.warning("pyRiemann not available. Riemannian classifiers will use fallback.")


class RiemannMDMClassifier(BaseEstimator, ClassifierMixin):
    """
    Riemannian MDM (Minimum Distance to Mean) Classifier.
    
    This classifier works on EEG covariance matrices using Riemannian geometry:
    1. Compute covariance matrices for each trial
    2. Compute mean covariance (Riemannian mean) for each class
    3. Classify by minimum distance to class means
    
    The Riemannian mean is computed using iterative algorithms that respect
    the manifold structure of SPD (Symmetric Positive Definite) matrices.
    
    Args:
        metric: Riemannian metric to use ('riemann', 'logeuclid', 'euclid', 'logdet')
                - 'riemann': Affine-invariant Riemannian metric (default, best performance)
                - 'logeuclid': Log-Euclidean metric (faster approximation)
                - 'euclid': Euclidean metric (fastest, least accurate)
                - 'logdet': Log-determinant divergence
        n_jobs: Number of parallel jobs (default: 1)
        
    Example:
        # Data shape: (n_samples, n_channels, n_times)
        clf = RiemannMDMClassifier(metric='riemann')
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
    """
    
    def __init__(
        self,
        metric: str = 'riemann',
        n_jobs: int = 1,
        classifier_type: str = 'tangent',
    ):
        if not PYRIEMANN_AVAILABLE:
            raise ImportError(
                "pyRiemann is required for Riemannian classifiers. "
                "Install with: pip install pyriemann>=0.3.0"
            )
        
        self.metric = metric
        self.n_jobs = n_jobs
        self.classifier_type = classifier_type
        
        # Covariance estimator - converts raw EEG (n, C, T) to cov matrices (n, C, C)
        self.cov_est_ = Covariances(estimator='lwf')
        
        # Initialize classifier: tangent (TangentSpace+LDA) is default, much better than pure MDM
        if classifier_type == 'tangent':
            self.clf_ = TSclassifier(metric=metric)
            self._use_tangent = True
        else:
            self.mdm_ = MDM(metric=metric, n_jobs=n_jobs)
            self._use_tangent = False
        
        # Label encoder
        self.le_ = LabelEncoder()
        
        # Fitted attributes
        self.classes_ = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RiemannMDMClassifier':
        """
        Fit the Riemannian MDM classifier.
        
        Args:
            X: Training data of shape (n_samples, n_channels, n_times)
            y: Labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (samples, channels, times), got {X.shape}")
        
        # Encode labels
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        
        logger.info(f"Fitting Riemannian classifier with {len(self.classes_)} classes, "
                   f"metric={self.metric}, type={self.classifier_type}")
        
        # Compute covariance matrices from raw EEG data
        X_cov = self.cov_est_.fit_transform(X)
        logger.info(f"  Covariance matrices shape: {X_cov.shape}")
        
        # Fit classifier
        if self._use_tangent:
            self.clf_.fit(X_cov, y_encoded)
        else:
            self.mdm_.fit(X_cov, y_encoded)
        self._fitted = True
        
        # Log class distribution
        for c in self.classes_:
            n_samples = (y == c).sum()
            logger.info(f"  Class {c}: {n_samples} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Data of shape (n_samples, n_channels, n_times)
            
        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        # Compute covariance matrices
        X_cov = self.cov_est_.transform(X)
        
        if self._use_tangent:
            y_pred_encoded = self.clf_.predict(X_cov)
        else:
            y_pred_encoded = self.mdm_.predict(X_cov)
        return self.le_.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Note: MDM doesn't natively support probabilities, 
        so we compute distance-based pseudo-probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        # Compute covariance matrices
        X_cov = self.cov_est_.transform(X)
        
        if self._use_tangent:
            if hasattr(self.clf_, 'predict_proba'):
                return self.clf_.predict_proba(X_cov)
            # Fallback for TSclassifier without predict_proba
            preds = self.clf_.predict(X_cov)
            proba = np.zeros((len(X), len(self.classes_)))
            for i, p in enumerate(preds):
                proba[i, p] = 1.0
            return proba
        else:
            # MDM: get distances to each class mean
            dists = self.mdm_.transform(X_cov)  # Shape: (n_samples, n_classes)
            # Convert distances to probabilities using softmax
            proba = np.exp(-dists) / np.exp(-dists).sum(axis=1, keepdims=True)
            return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return (self.predict(X) == y).mean()
    
    def get_class_means(self) -> Dict:
        """
        Get the Riemannian mean covariance matrices for each class.
        
        Returns:
            Dictionary mapping class labels to mean covariance matrices
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        means = {}
        for i, c in enumerate(self.classes_):
            means[c] = self.mdm_.covmeans_[i]
        
        return means


class TangentSpaceClassifier(BaseEstimator, ClassifierMixin):
    """
    Tangent Space Classifier using pyRiemann.
    
    Projects SPD matrices to tangent space and uses a standard classifier.
    More flexible than MDM as it can use any classifier on tangent features.
    
    Args:
        metric: Covariance estimator metric (default: 'riemann')
        tsupdate: Update tangent space mean during predict (default: True)
        classifier: Backend classifier ('lda', 'svc', or sklearn classifier)
        
    Example:
        clf = TangentSpaceClassifier(metric='riemann', classifier='lda')
        clf.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        metric: str = 'riemann',
        tsupdate: bool = True,
        classifier: str = 'lda',
    ):
        if not PYRIEMANN_AVAILABLE:
            raise ImportError("pyRiemann is required for TangentSpace classifier.")
        
        self.metric = metric
        self.tsupdate = tsupdate
        
        # Initialize tangent space mapper
        self.ts_ = TangentSpace(metric=metric, tsupdate=tsupdate)
        
        # Initialize classifier
        if classifier == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.clf_ = LinearDiscriminantAnalysis()
        elif classifier == 'svc':
            from sklearn.svm import SVC
            self.clf_ = SVC(kernel='rbf', probability=True)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        self.le_ = LabelEncoder()
        self.classes_ = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TangentSpaceClassifier':
        """Fit tangent space projection and classifier."""
        if X.ndim != 3:
            raise ValueError(f"X must be 3D, got {X.shape}")
        
        self.classes_ = np.unique(y)
        y_encoded = self.le_.fit_transform(y)
        
        logger.info(f"Fitting TangentSpace classifier with {len(self.classes_)} classes")
        
        # Project to tangent space
        X_tangent = self.ts_.fit_transform(X)
        
        # Fit classifier
        self.clf_.fit(X_tangent, y_encoded)
        self._fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        X_tangent = self.ts_.transform(X)
        y_pred_encoded = self.clf_.predict(X_tangent)
        return self.le_.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        X_tangent = self.ts_.transform(X)
        
        if hasattr(self.clf_, 'predict_proba'):
            return self.clf_.predict_proba(X_tangent)
        
        # Fallback
        pred = self.clf_.predict(X_tangent)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[:, i] = (pred == i).astype(float)
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return (self.predict(X) == y).mean()


class FallbackMDMClassifier(BaseEstimator, ClassifierMixin):
    """
    Fallback MDM implementation when pyRiemann is not available.
    
    Uses Euclidean distance to class mean covariances.
    Less accurate but works without pyRiemann.
    """
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.classes_ = None
        self.mean_covs_ = None
        self.le_ = LabelEncoder()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FallbackMDMClassifier':
        """Fit by computing mean covariance for each class."""
        self.classes_ = np.unique(y)
        y_encoded = self.le_.fit_transform(y)
        
        # Compute covariances
        covs = []
        for i in range(len(X)):
            trial = X[i]
            # Subsample time dimension for efficiency
            step = max(1, trial.shape[1] // self.n_components)
            trial_sub = trial[:, ::step]
            cov = np.cov(trial_sub)
            covs.append(cov)
        covs = np.array(covs)
        
        # Compute mean covariance for each class
        self.mean_covs_ = []
        for c in range(len(self.classes_)):
            class_covs = covs[y_encoded == c]
            mean_cov = class_covs.mean(axis=0)
            self.mean_covs_.append(mean_cov)
        self.mean_covs_ = np.array(self.mean_covs_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by minimum distance to class means."""
        predictions = []
        
        for trial in X:
            # Compute covariance
            step = max(1, trial.shape[1] // self.n_components)
            trial_sub = trial[:, ::step]
            cov = np.cov(trial_sub)
            
            # Compute distance to each class mean
            dists = []
            for mean_cov in self.mean_covs_:
                dist = np.linalg.norm(cov - mean_cov, ord='fro')
                dists.append(dist)
            
            pred = np.argmin(dists)
            predictions.append(self.classes_[pred])
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return (self.predict(X) == y).mean()


def create_riemann_classifier(
    classifier_type: str = 'mdm',
    metric: str = 'riemann',
    **kwargs,
) -> Union[RiemannMDMClassifier, TangentSpaceClassifier]:
    """
    Factory function to create Riemannian classifier.
    
    Args:
        classifier_type: 'mdm' or 'tangent'
        metric: Riemannian metric
        **kwargs: Additional arguments
        
    Returns:
        Classifier instance
    """
    if not PYRIEMANN_AVAILABLE:
        logger.warning("pyRiemann not available, using fallback classifier")
        return FallbackMDMClassifier(**kwargs)
    
    if classifier_type == 'mdm':
        return RiemannMDMClassifier(metric=metric, **kwargs)
    elif classifier_type == 'tangent':
        return TangentSpaceClassifier(metric=metric, **kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


# Export
__all__ = [
    "RiemannMDMClassifier",
    "TangentSpaceClassifier",
    "FallbackMDMClassifier",
    "create_riemann_classifier",
    "PYRIEMANN_AVAILABLE",
]
