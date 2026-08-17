"""
Common Spatial Pattern (CSP) Classifier for Motor Imagery BCI
Implements classical CSP with regularization options and classifier integration
"""

import logging
from typing import Tuple, Optional, Dict, List, Union

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CSPFeatures:
    """
    Common Spatial Pattern feature extraction.
    
    CSP finds spatial filters that maximize variance for one class
    while minimizing for another, making it ideal for motor imagery
    classification (e.g., left vs right hand).
    
    The algorithm:
    1. Compute covariance matrices for each class
    2. Compute composite covariance and whitened transformation
    3. Find generalized eigenvectors
    4. Project data using top/bottom eigenvectors
    
    Args:
        n_components: Number of CSP components (pairs) to extract (default: 4)
                     Total features = 2 * n_components
        reg: Regularization parameter (default: 0.1)
        estimator: Covariance estimator type: 'lwf' (ledoit-wolf), 'shrunk', or float
    """
    
    def __init__(
        self,
        n_components: int = 4,
        reg: Union[str, float] = 0.1,
        estimator: str = 'lwf',
    ):
        self.n_components = n_components
        self.reg = reg
        self.estimator = estimator
        
        # Fitted attributes
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPFeatures':
        """
        Fit CSP filters to training data.
        
        Args:
            X: Training data of shape (n_samples, n_channels, n_times)
            y: Labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (samples, channels, times), got shape {X.shape}")
        
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")
        
        n_channels = X.shape[1]
        
        logger.info(f"Fitting CSP with {self.n_components} components "
                   f"on {len(X)} samples, {n_channels} channels")
        
        # Compute covariance matrices for each class
        covs = []
        for c in classes:
            # Select trials for this class
            X_c = X[y == c]
            
            # Compute average covariance matrix
            cov_c = self._compute_covariance(X_c)
            covs.append(cov_c)
        
        # Regularize covariances
        covs = self._regularize_covariances(covs, n_channels)
        
        # Compute composite covariance
        cov_total = sum(covs)
        
        # Whitening transform
        try:
            # Eigendecomposition of composite covariance
            eigvals, eigvecs = linalg.eigh(cov_total)
            
            # Sort eigenvalues in descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Whitening matrix
            whitened = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-10)) @ eigvecs.T
            
            # Transform class covariances
            covs_whitened = [whitened @ cov @ whitened.T for cov in covs]
            
            # Generalized eigenvalue decomposition
            eigvals_csp, eigvecs_csp = linalg.eig(*covs_whitened)
            
            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigvals_csp))[::-1]
            eigvecs_csp = eigvecs_csp[:, idx]
            
            # Select top n_components and bottom n_components
            filters = np.vstack([
                eigvecs_csp[:, :self.n_components],  # Top (max variance for class 0)
                eigvecs_csp[:, -self.n_components:],  # Bottom (min variance for class 0)
            ])
            
            self.filters_ = filters
            
        except Exception as e:
            logger.error(f"CSP fitting failed: {e}")
            raise
        
        self._fitted = True
        
        # Compute feature statistics for normalization
        features = self.transform(X)
        self.mean_ = features.mean(axis=0)
        self.std_ = features.std(axis=0) + 1e-10
        
        logger.info(f"CSP fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted CSP filters.
        
        Args:
            X: Data of shape (n_samples, n_channels, n_times)
            
        Returns:
            CSP features of shape (n_samples, 2 * n_components)
        """
        if not self._fitted:
            raise RuntimeError("CSP not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        n_features = 2 * self.n_components
        
        features = np.zeros((n_samples, n_features))
        
        for i, trial in enumerate(X):
            # Apply spatial filters
            projected = self.filters_ @ trial  # (2*n_components, n_times)
            
            # Compute variance for each filter pair
            var_total = np.sum(projected ** 2, axis=1) + 1e-10
            
            # Log ratio for each component pair
            for j in range(self.n_components):
                # Feature: log(var_top / (var_top + var_bottom))
                var_ratio = var_total[j] / (var_total[j] + var_total[j + self.n_components] + 1e-10)
                features[i, j] = np.log(var_ratio + 1e-10)
                features[i, j + self.n_components] = np.log(1 - var_ratio + 1e-10)
        
        return features
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute average covariance matrix for a set of trials."""
        n_trials, n_channels, n_times = X.shape
        
        # Reshape to (n_channels, n_trials * n_times)
        X_2d = X.reshape(n_trials, n_channels, -1).transpose(1, 0, 2)
        X_flat = X_2d.reshape(n_channels, -1)
        
        # Compute covariance
        cov = np.cov(X_flat)
        
        # Ensure symmetry
        cov = (cov + cov.T) / 2
        
        return cov
    
    def _regularize_covariances(
        self, 
        covs: List[np.ndarray], 
        n_channels: int
    ) -> List[np.ndarray]:
        """Apply regularization to covariance matrices."""
        if isinstance(self.reg, float):
            # Add scaled identity
            trace = np.mean([np.trace(c) for c in covs])
            reg_mat = self.reg * trace * np.eye(n_channels) / n_channels
            return [c + reg_mat for c in covs]
        
        elif self.reg == 'lwf' or self.reg == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            return [lw.fit(c).covariance_ for c in covs]
        
        return covs


class CSPClassifier(BaseEstimator, ClassifierMixin):
    """
    Complete CSP-based motor imagery classifier.
    
    Combines CSP feature extraction with a classifier (LDA or SVM).
    
    Args:
        n_components: Number of CSP components (default: 4)
        reg: Regularization parameter (default: 0.1)
        classifier: Type of classifier ('lda' or 'svm', default: 'lda')
        svm_kernel: SVM kernel type if using SVM (default: 'rbf')
        normalize: Apply feature normalization (default: True)
        
    Example:
        clf = CSPClassifier(n_components=4, classifier='lda')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = (predictions == y_test).mean()
    """
    
    def __init__(
        self,
        n_components: int = 4,
        reg: Union[str, float] = 0.1,
        classifier: str = 'lda',
        svm_kernel: str = 'rbf',
        normalize: bool = True,
    ):
        self.n_components = n_components
        self.reg = reg
        self.classifier = classifier
        self.svm_kernel = svm_kernel
        self.normalize = normalize
        
        self.csp_ = CSPFeatures(n_components=n_components, reg=reg)
        self.scaler_ = StandardScaler() if normalize else None
        
        # Initialize classifier
        if classifier == 'lda':
            self.clf_ = LinearDiscriminantAnalysis()
        elif classifier == 'svm':
            self.clf_ = SVC(kernel=svm_kernel, probability=True)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPClassifier':
        """
        Fit CSP features and classifier.
        
        Args:
            X: Training data (n_samples, n_channels, n_times)
            y: Labels (n_samples,)
            
        Returns:
            Self
        """
        # Extract CSP features
        csp_features = self.csp_.fit_transform(X, y)
        
        # Normalize features
        if self.normalize:
            csp_features = self.scaler_.fit_transform(csp_features)
        
        # Fit classifier
        self.clf_.fit(csp_features, y)
        
        logger.info(f"CSPClassifier fitted with {self.n_components} components")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        features = self.csp_.transform(X)
        if self.normalize:
            features = self.scaler_.transform(features)
        return self.clf_.predict(features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        features = self.csp_.transform(X)
        if self.normalize:
            features = self.scaler_.transform(features)
        
        if hasattr(self.clf_, 'predict_proba'):
            return self.clf_.predict_proba(features)
        
        # Fallback for classifiers without predict_proba
        pred = self.clf_.predict(features)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[:, i] = (pred == c).astype(float)
        return proba
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to CSP features."""
        features = self.csp_.transform(X)
        if self.normalize:
            features = self.scaler_.transform(features)
        return features
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return (self.predict(X) == y).mean()
    
    @property
    def classes_(self) -> np.ndarray:
        """Get class labels."""
        return self.clf_.classes_


# Regularized CSP variants
class RegularizedCSP(CSPFeatures):
    """
    CSP with advanced regularization options.
    
    Regularization helps when:
    - Few training samples
    - Noisy covariance estimates
    - High-dimensional data
    """
    
    def __init__(
        self,
        n_components: int = 4,
        reg_type: str = 'lwf',  # 'lwf', 'oas', 'shrinkage', 'diagonal'
        reg_param: float = 0.1,
    ):
        super().__init__(n_components=n_components, reg=reg_param)
        self.reg_type = reg_type


class RCSPClassifier(CSPClassifier):
    """
    Regularized CSP Classifier.
    
    Uses Ledoit-Wolf shrinkage for more stable covariance estimation.
    """
    
    def __init__(self, n_components: int = 4, shrinkage: float = 0.1):
        super().__init__(
            n_components=n_components,
            reg='lwf',
            classifier='lda',
        )


# Export
__all__ = [
    "CSPFeatures",
    "CSPClassifier",
    "RegularizedCSP",
    "RCSPClassifier",
]
