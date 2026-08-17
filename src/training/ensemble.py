"""
Ensemble Learning Module for BCI Classification
Implements voting, stacking, weighted, bagging, and adaptive ensemble strategies

This module provides a unified framework for combining multiple BCI models
(EEGNet, CSP, Riemannian classifiers) to improve classification accuracy
and robustness through ensemble learning.

Author: BCI_Projects Team
"""

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable
)
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.eegnet import EEGNetClassifier
from src.models.csp import CSPClassifier
from src.models.riemann_mdm import RiemannMDMClassifier
from src.training.trainer import Trainer, TrainingConfig
from src.evaluation.metrics import compute_all_metrics, EvaluationMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single base model in the ensemble."""
    model_type: str  # 'eegnet', 'csp', 'riemann'
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    is_torch: bool = False
    weight: float = 1.0
    train_config: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training and evaluation."""
    strategy: str = 'voting'  # 'voting', 'stacking', 'weighted', 'bagging', 'adaptive'
    voting_mode: str = 'soft'  # 'hard' or 'soft'
    n_folds: int = 5
    random_state: int = 42
    save_base_models: bool = True
    ensemble_dir: str = "./models/ensemble/"
    
    # Stacking specific
    meta_learner: str = 'logistic'  # 'logistic', 'svm', 'ridge'
    meta_cv: bool = True  # Use cross-validation for meta-features
    
    # Bagging specific
    n_bagging: int = 10
    bagging_ratio: float = 0.8  # Ratio of training data per bagging model
    
    # Adaptive specific
    gating_hidden_dim: int = 64
    gating_lr: float = 0.001
    
    # General ensemble
    require_agreement: bool = False  # Require minimum agreement threshold
    agreement_threshold: float = 0.5  # Minimum fraction of models agreeing


@dataclass
class EnsembleResult:
    """Results from ensemble evaluation."""
    accuracy: float
    kappa: float
    predictions: np.ndarray
    probabilities: np.ndarray
    confidences: np.ndarray
    individual_scores: Dict[str, float]
    ensemble_weights: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'kappa': self.kappa,
            'predictions': self.predictions.tolist(),
            'confidences': self.confidences.tolist(),
            'individual_scores': self.individual_scores,
            'ensemble_weights': self.ensemble_weights,
            'training_time': self.training_time,
        }


# ============================================================================
# Base Ensemble Class
# ============================================================================

class BaseEnsemble(ClassifierMixin, BaseEstimator):
    """
    Abstract base class for ensemble classifiers.
    
    All ensemble strategies inherit from this class which provides:
    - Model management
    - Training coordination
    - Prediction coordination
    - State persistence
    
    Args:
        models: List of ModelConfig objects
        config: EnsembleConfig object
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
    ):
        self.models = models
        self.config = config or EnsembleConfig()
        self._fitted_models: Dict[str, Any] = {}
        self._model_predictions: Dict[str, np.ndarray] = {}
        self._model_probabilities: Dict[str, np.ndarray] = {}
        self._is_fitted = False
        self.n_classes_: Optional[int] = None
        
        # Validate model types
        self._validate_models()
        
        # Initialize ensemble directory
        if self.config.save_base_models:
            Path(self.config.ensemble_dir).mkdir(parents=True, exist_ok=True)
    
    def _validate_models(self) -> None:
        """Validate model configurations."""
        valid_types = {'eegnet', 'csp', 'riemann'}
        
        for i, model_config in enumerate(self.models):
            if model_config.model_type not in valid_types:
                raise ValueError(
                    f"Invalid model type '{model_config.model_type}' at index {i}. "
                    f"Valid types: {valid_types}"
                )
            
            # Set default name if not provided
            if not model_config.name:
                model_config.name = f"{model_config.model_type}_{i}"
    
    def _create_model_instance(
        self, 
        model_config: ModelConfig,
        n_channels: int,
        n_times: int,
        n_classes: int,
    ) -> Tuple[Any, bool]:
        """
        Create a model instance based on configuration.
        
        Returns:
            Tuple of (model_instance, is_torch_model)
        """
        model_type = model_config.model_type
        model_config_dict = model_config.config.copy()
        is_torch = False
        
        if model_type == 'eegnet':
            model = EEGNetClassifier(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=n_classes,
                **model_config_dict,
            )
            is_torch = True
            
        elif model_type == 'csp':
            model = CSPClassifier(
                **model_config_dict,
            )
            
        elif model_type == 'riemann':
            model = RiemannMDMClassifier(
                **model_config_dict,
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model, is_torch
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'BaseEnsemble':
        """
        Fit all base models in the ensemble.
        
        Args:
            X: Training features of shape (n_samples, n_channels, n_times)
            y: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def _fit_base_model(
        self,
        model_config: ModelConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Fit a single base model.
        
        Args:
            model_config: Model configuration
            X_train: Training data
            y_train: Training labels
            X_val: Optional validation data
            y_val: Optional validation labels
            
        Returns:
            Tuple of (fitted_model, training_history)
        """
        n_samples, n_channels, n_times = X_train.shape
        n_classes = len(np.unique(y_train))
        
        self.n_classes_ = n_classes
        
        model, is_torch = self._create_model_instance(
            model_config, n_channels, n_times, n_classes
        )
        
        training_history = {}
        
        if is_torch:
            # PyTorch model training
            train_config = model_config.train_config or {
                'epochs': 100,
                'batch_size': 64,
                'learning_rate': 0.001,
            }
            
            trainer_config = TrainingConfig(**train_config)
            
            # Create data loaders
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.LongTensor(y_train)
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=train_config.get('batch_size', 64),
                shuffle=True,
            )
            
            val_loader = None
            if X_val is not None:
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.LongTensor(y_val)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=64)
            
            trainer = Trainer(
                model=model,
                config=trainer_config,
            )
            
            history = trainer.train(train_loader, val_loader)
            training_history = {k: v[-1] for k, v in history.items()}
            
        else:
            # Sklearn-style model training
            model.fit(X_train, y_train)
            training_history['train_acc'] = accuracy_score(y_train, model.predict(X_train))
            
            if X_val is not None:
                training_history['val_acc'] = accuracy_score(y_val, model.predict(X_val))
        
        return model, training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raise NotImplementedError("Subclasses must implement predict_proba()")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return accuracy_score(y, self.predict(X))
    
    def save_models(self, path: Optional[str] = None) -> None:
        """Save all base models to disk."""
        if not self._fitted_models:
            raise RuntimeError("No models fitted. Call fit() first.")
        
        save_dir = Path(path) if path else Path(self.config.ensemble_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model_info in self._fitted_models.items():
            model = model_info['model']
            
            # Determine save path
            if isinstance(model, nn.Module):
                model_path = save_dir / f"{name}.pt"
                torch.save(model.state_dict(), model_path)
            else:
                import joblib
                model_path = save_dir / f"{name}.pkl"
                joblib.dump(model, model_path)
            
            logger.info(f"Saved {name} to {model_path}")
    
    def load_models(self, path: Optional[str] = None) -> None:
        """Load all base models from disk."""
        load_dir = Path(path) if path else Path(self.config.ensemble_dir)
        
        for model_config in self.models:
            name = model_config.name
            
            if model_config.is_torch:
                model_path = load_dir / f"{name}.pt"
                if model_path.exists():
                    # Load torch model
                    model, _ = self._create_model_instance(
                        model_config, 
                        self.n_channels_,
                        self.n_times_,
                        self.n_classes_,
                    )
                    model.load_state_dict(torch.load(model_path))
                    self._fitted_models[name] = {'model': model, 'is_torch': True}
            else:
                import joblib
                model_path = load_dir / f"{name}.pkl"
                if model_path.exists():
                    model = joblib.load(model_path)
                    self._fitted_models[name] = {'model': model, 'is_torch': False}
        
        self._is_fitted = True


# ============================================================================
# Voting Ensemble
# ============================================================================

class VotingEnsemble(BaseEnsemble):
    """
    Voting Ensemble combining multiple models through voting.
    
    Supports both hard voting (majority vote) and soft voting (probability average).
    
    Args:
        models: List of ModelConfig objects for base models
        config: EnsembleConfig object
        voting_mode: 'hard' for majority vote, 'soft' for probability averaging
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
        voting_mode: str = 'soft',
    ):
        super().__init__(models, config)
        self.voting_mode = voting_mode
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'VotingEnsemble':
        """Fit all base models."""
        logger.info(f"Fitting VotingEnsemble with {len(self.models)} models ({self.voting_mode} voting)")
        
        start_time = time.time()
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        
        # Train each base model
        for i, model_config in enumerate(self.models):
            logger.info(f"\n[{i+1}/{len(self.models)}] Training {model_config.name}")
            
            model, history = self._fit_base_model(
                model_config, X, y, X_val, y_val
            )
            
            self._fitted_models[model_config.name] = {
                'model': model,
                'is_torch': model_config.is_torch or model_config.model_type == 'eegnet',
                'history': history,
            }
            
            logger.info(f"  {model_config.name} train acc: {history.get('train_acc', 'N/A')}")
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        logger.info(f"\nVotingEnsemble training completed in {self._training_time:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using voting."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        if self.voting_mode == 'soft':
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
        
        # Hard voting
        votes = np.zeros((len(X), self.n_classes_))
        
        for name, model_info in self._fitted_models.items():
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(next(model.parameters()).device)
                    outputs = model(X_tensor)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = model.predict(X)
            
            # Count votes
            for i, pred in enumerate(preds):
                votes[i, pred] += 1
        
        return np.argmax(votes, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using soft voting."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        probas = []
        weights = []
        
        for model_config in self.models:
            model_info = self._fitted_models[model_config.name]
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    proba = np.zeros((len(X), self.n_classes_))
                    for i, p in enumerate(preds):
                        proba[i, p] = 1.0
            
            probas.append(proba)
            weights.append(model_config.weight)
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_proba = np.zeros((len(X), self.n_classes_))
        
        for proba, weight in zip(probas, weights):
            ensemble_proba += proba * weight
        
        return ensemble_proba
    
    def get_individual_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Get accuracy scores for each base model."""
        scores = {}
        
        for model_config in self.models:
            model_info = self._fitted_models[model_config.name]
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()
            else:
                preds = model.predict(X)
            
            scores[model_config.name] = accuracy_score(y, preds)
        
        return scores


# ============================================================================
# Stacking Ensemble
# ============================================================================

class StackingEnsemble(BaseEnsemble):
    """
    Stacking Ensemble using meta-learner for final prediction.
    
    First trains base models, then uses their predictions as features
    for a meta-learner (e.g., Logistic Regression, SVM).
    
    Args:
        models: List of ModelConfig objects for base models
        config: EnsembleConfig object
        meta_learner: Type of meta-learner ('logistic', 'svm', 'ridge')
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
        meta_learner: str = 'logistic',
    ):
        super().__init__(models, config)
        self.meta_learner_type = meta_learner
        self.meta_model: Optional[Any] = None
    
    def _create_meta_learner(self) -> Any:
        """Create the meta-learner model."""
        if self.meta_learner_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                multi_class='multinomial',
            )
        elif self.meta_learner_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config.random_state,
            )
        elif self.meta_learner_type == 'ridge':
            from sklearn.linear_model import RidgeClassifier
            return RidgeClassifier(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'StackingEnsemble':
        """Fit base models and meta-learner."""
        logger.info(f"Fitting StackingEnsemble with {len(self.models)} base models")
        
        start_time = time.time()
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        n_classes = len(np.unique(y))
        self.n_classes_ = n_classes
        
        # Step 1: Train base models
        logger.info("Step 1: Training base models...")
        
        for i, model_config in enumerate(self.models):
            logger.info(f"\n[{i+1}/{len(self.models)}] Training {model_config.name}")
            
            model, history = self._fit_base_model(
                model_config, X, y, X_val, y_val
            )
            
            self._fitted_models[model_config.name] = {
                'model': model,
                'is_torch': model_config.model_type == 'eegnet',
                'history': history,
            }
            
            logger.info(f"  {model_config.name} train acc: {history.get('train_acc', 'N/A')}")
        
        # Step 2: Generate meta-features using cross-validation
        logger.info("\nStep 2: Generating meta-features (cross-validation)...")
        
        if self.config.meta_cv:
            meta_features = self._generate_meta_features_cv(X, y)
        else:
            meta_features = self._generate_meta_features(X)
        
        # Step 3: Train meta-learner
        logger.info(f"\nStep 3: Training meta-learner ({self.meta_learner_type})...")
        
        self.meta_model = self._create_meta_learner()
        self.meta_model.fit(meta_features, y)
        
        logger.info(f"Meta-learner trained with {len(meta_features)} meta-features")
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        logger.info(f"\nStackingEnsemble training completed in {self._training_time:.2f}s")
        
        return self
    
    def _generate_meta_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features = []
        
        for model_config in self.models:
            model_info = self._fitted_models[model_config.name]
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    proba = np.zeros((len(X), self.n_classes_))
                    for i, p in enumerate(preds):
                        proba[i, p] = 1.0
            
            meta_features.append(proba)
        
        # Concatenate all probability vectors
        return np.hstack(meta_features)
    
    def _generate_meta_features_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Generate meta-features using cross-validation to prevent overfitting.
        
        For each fold, train base models on training folds and predict
        on validation fold. This ensures meta-learner sees out-of-fold predictions.
        """
        n_samples = len(X)
        n_classes = self.n_classes_
        n_models = len(self.models)
        
        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"  Fold {fold_idx + 1}/{self.config.n_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            fold_meta_features = []
            
            # Train and predict for each base model
            for model_config in self.models:
                # Create fresh model instance
                model, is_torch = self._create_model_instance(
                    model_config,
                    X.shape[1],
                    X.shape[2],
                    n_classes,
                )
                
                if is_torch:
                    # Quick training for CV
                    X_tensor = torch.FloatTensor(X_fold_train)
                    y_tensor = torch.LongTensor(y_fold_train)
                    train_dataset = TensorDataset(X_tensor, y_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                    
                    trainer = Trainer(
                        model=model,
                        config=TrainingConfig(epochs=50, batch_size=64),
                    )
                    trainer.train(train_loader, epochs=50)
                    
                    model.eval()
                    device = next(model.parameters()).device
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_fold_val).to(device)
                        outputs = model(X_val_tensor)
                        proba = torch.softmax(outputs, dim=1).cpu().numpy()
                else:
                    model.fit(X_fold_train, y_fold_train)
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_fold_val)
                    else:
                        preds = model.predict(X_fold_val)
                        proba = np.zeros((len(X_fold_val), n_classes))
                        for i, p in enumerate(preds):
                            proba[i, p] = 1.0
                
                fold_meta_features.append(proba)
            
            # Store meta-features for validation fold
            fold_meta = np.hstack(fold_meta_features)
            meta_features[val_idx] = fold_meta
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using meta-learner."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using meta-learner."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        meta_features = self._generate_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        
        # Fallback for classifiers without predict_proba
        preds = self.meta_model.predict(meta_features)
        proba = np.zeros((len(X), self.n_classes_))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba


# ============================================================================
# Weighted Ensemble
# ============================================================================

class WeightedEnsemble(BaseEnsemble):
    """
    Weighted Ensemble using validation performance to determine weights.
    
    Weights are determined by each model's validation accuracy, making
    better-performing models have more influence on the final prediction.
    
    Args:
        models: List of ModelConfig objects for base models
        config: EnsembleConfig object
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
    ):
        super().__init__(models, config)
        self.model_weights_: Optional[Dict[str, float]] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'WeightedEnsemble':
        """Fit base models and compute weights based on validation accuracy."""
        logger.info(f"Fitting WeightedEnsemble with {len(self.models)} models")
        
        if X_val is None or y_val is None:
            # Use training data split for weight determination
            from sklearn.model_selection import train_test_split
            X_fit, X_val, y_fit, y_val = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=self.config.random_state,
            )
            logger.info("No validation set provided, using 80/20 split")
        else:
            X_fit, y_fit = X, y
        
        start_time = time.time()
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        self.n_classes_ = len(np.unique(y))
        
        # Train each base model
        validation_scores = {}
        
        for i, model_config in enumerate(self.models):
            logger.info(f"\n[{i+1}/{len(self.models)}] Training {model_config.name}")
            
            model, history = self._fit_base_model(
                model_config, X_fit, y_fit, X_val, y_val
            )
            
            self._fitted_models[model_config.name] = {
                'model': model,
                'is_torch': model_config.model_type == 'eegnet',
                'history': history,
            }
            
            # Compute validation accuracy
            val_acc = history.get('val_acc', 0.0)
            validation_scores[model_config.name] = val_acc
            
            logger.info(f"  {model_config.name} val acc: {val_acc:.4f}")
        
        # Compute weights based on validation scores
        self._compute_weights(validation_scores)
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        logger.info(f"\nComputed weights: {self.model_weights_}")
        logger.info(f"WeightedEnsemble training completed in {self._training_time:.2f}s")
        
        return self
    
    def _compute_weights(
        self,
        validation_scores: Dict[str, float],
    ) -> None:
        """Compute model weights based on validation scores."""
        # Softmax-based weighting: higher score = higher weight
        scores = np.array(list(validation_scores.values()))
        
        # Temperature-scaled softmax
        temperature = 0.1
        weights = np.exp(scores / temperature)
        weights = weights / weights.sum()
        
        self.model_weights_ = {
            name: float(weights[i])
            for i, name in enumerate(validation_scores.keys())
        }
        
        logger.info(f"Weights computed using softmax with temperature={temperature}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted averaging of probabilities."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted averaging."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        ensemble_proba = np.zeros((len(X), self.n_classes_))
        total_weight = 0.0
        
        for model_config in self.models:
            model_info = self._fitted_models[model_config.name]
            model = model_info['model']
            is_torch = model_info['is_torch']
            weight = self.model_weights_[model_config.name]
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    proba = np.zeros((len(X), self.n_classes_))
                    for i, p in enumerate(preds):
                        proba[i, p] = 1.0
            
            ensemble_proba += proba * weight
            total_weight += weight
        
        return ensemble_proba / total_weight


# ============================================================================
# Bagging Ensemble
# ============================================================================

class BaggingEnsemble(BaseEnsemble):
    """
    Bagging Ensemble: trains same model on different data subsets.
    
    For each base model type, creates multiple instances trained on
    bootstrapped samples, then combines through voting.
    
    Args:
        models: List of ModelConfig objects (will be replicated for bagging)
        config: EnsembleConfig object
        n_bagging: Number of models per base type
        bagging_ratio: Ratio of data to sample for each bootstrap
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
        n_bagging: int = 10,
        bagging_ratio: float = 0.8,
    ):
        super().__init__(models, config)
        self.n_bagging = n_bagging
        self.bagging_ratio = bagging_ratio
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'BaggingEnsemble':
        """Fit bagged models."""
        logger.info(f"Fitting BaggingEnsemble: {self.n_bagging} bags, ratio={self.bagging_ratio}")
        
        start_time = time.time()
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        self.n_classes_ = len(np.unique(y))
        
        n_samples = len(X)
        n_bag_samples = int(n_samples * self.bagging_ratio)
        
        for base_idx, base_model_config in enumerate(self.models):
            logger.info(f"\nBagging model type: {base_model_config.model_type}")
            
            for bag_idx in range(self.n_bagging):
                # Bootstrap sample
                np.random.seed(self.config.random_state + bag_idx)
                indices = np.random.choice(n_samples, size=n_bag_samples, replace=True)
                
                X_bag = X[indices]
                y_bag = y[indices]
                
                bag_name = f"{base_model_config.name}_bag{bag_idx}"
                logger.info(f"  [{bag_idx+1}/{self.n_bagging}] {bag_name} "
                           f"(samples: {n_bag_samples})")
                
                model, history = self._fit_base_model(
                    base_model_config, X_bag, y_bag, X_val, y_val
                )
                
                self._fitted_models[bag_name] = {
                    'model': model,
                    'is_torch': base_model_config.model_type == 'eegnet',
                    'history': history,
                    'base_config': base_model_config,
                }
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        logger.info(f"\nBaggingEnsemble training completed in {self._training_time:.2f}s")
        logger.info(f"Total models trained: {len(self._fitted_models)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority voting across all bagged models."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        votes = np.zeros((len(X), self.n_classes_))
        
        for name, model_info in self._fitted_models.items():
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()
            else:
                preds = model.predict(X)
            
            for i, pred in enumerate(preds):
                votes[i, pred] += 1
        
        return np.argmax(votes, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities by averaging across all bagged models."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        ensemble_proba = np.zeros((len(X), self.n_classes_))
        n_models = 0
        
        for name, model_info in self._fitted_models.items():
            model = model_info['model']
            is_torch = model_info['is_torch']
            
            if is_torch:
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    proba = np.zeros((len(X), self.n_classes_))
                    for i, p in enumerate(preds):
                        proba[i, p] = 1.0
            
            ensemble_proba += proba
            n_models += 1
        
        return ensemble_proba / n_models


# ============================================================================
# Adaptive Ensemble (Gating Network)
# ============================================================================

class AdaptiveEnsemble(BaseEnsemble):
    """
    Adaptive Ensemble using a gating network for dynamic weight assignment.
    
    The gating network learns to assign different weights to base models
    based on the input features, allowing the ensemble to adapt its
    combination strategy for different inputs.
    
    Args:
        models: List of ModelConfig objects for base models
        config: EnsembleConfig object
        gating_hidden_dim: Hidden layer dimension for gating network
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[EnsembleConfig] = None,
        gating_hidden_dim: int = 64,
    ):
        super().__init__(models, config)
        self.gating_hidden_dim = gating_hidden_dim
        self.gating_network: Optional[nn.Module] = None
    
    def _create_gating_network(self) -> nn.Module:
        """Create the gating network for weight assignment."""
        n_models = len(self.models)
        n_features = self.n_channels_  # Use channel count as feature dimension
        
        class GatingNetwork(nn.Module):
            """Simple MLP gating network."""
            
            def __init__(self, n_models: int, n_features: int, hidden_dim: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, n_models),
                    nn.Softmax(dim=1),
                )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Aggregate spatial features for gating
                x_pooled = x.mean(dim=-1)  # (batch, channels)
                return self.network(x_pooled)
        
        return GatingNetwork(n_models, n_features, self.gating_hidden_dim)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'AdaptiveEnsemble':
        """Fit base models and gating network."""
        logger.info(f"Fitting AdaptiveEnsemble with {len(self.models)} models")
        
        start_time = time.time()
        self.n_channels_ = X.shape[1]
        self.n_times_ = X.shape[2]
        self.n_classes_ = len(np.unique(y))
        n_models = len(self.models)
        
        # Step 1: Train base models
        logger.info("Step 1: Training base models...")
        
        for i, model_config in enumerate(self.models):
            logger.info(f"\n[{i+1}/{n_models}] Training {model_config.name}")
            
            model, history = self._fit_base_model(
                model_config, X, y, X_val, y_val
            )
            
            self._fitted_models[model_config.name] = {
                'model': model,
                'is_torch': model_config.model_type == 'eegnet',
                'history': history,
            }
        
        # Step 2: Train gating network
        logger.info("\nStep 2: Training gating network...")
        
        self.gating_network = self._create_gating_network()
        self._train_gating_network(X, y, X_val, y_val)
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        logger.info(f"\nAdaptiveEnsemble training completed in {self._training_time:.2f}s")
        
        return self
    
    def _train_gating_network(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        """Train the gating network to predict optimal model weights."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gating_network.to(device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        val_loader = None
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.gating_network.parameters(),
            lr=self.config.gating_lr,
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            self.gating_network.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Get gating weights
                gate_weights = self.gating_network(batch_x)  # (batch, n_models)
                
                # Get predictions from each base model
                ensemble_logits = []
                
                for model_config in self.models:
                    model = self._fitted_models[model_config.name]['model']
                    model.eval()
                    with torch.no_grad():
                        outputs = model(batch_x)
                    ensemble_logits.append(outputs)
                
                # Weighted combination
                ensemble_logits = torch.stack(ensemble_logits, dim=0)  # (n_models, batch, classes)
                gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, n_models, 1, 1)
                weighted_logits = (ensemble_logits * gate_weights_expanded).sum(dim=0)  # (batch, classes)
                
                loss = criterion(weighted_logits, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            if val_loader:
                self.gating_network.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        gate_weights = self.gating_network(batch_x)
                        
                        ensemble_logits = []
                        for model_config in self.models:
                            model = self._fitted_models[model_config.name]['model']
                            model.eval()
                            with torch.no_grad():
                                outputs = model(batch_x)
                            ensemble_logits.append(outputs)
                        
                        ensemble_logits = torch.stack(ensemble_logits, dim=0)
                        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)
                        weighted_logits = (ensemble_logits * gate_weights_expanded).sum(dim=0)
                        
                        preds = torch.argmax(weighted_logits, dim=1)
                        correct += (preds == batch_y).sum().item()
                        total += batch_y.size(0)
                
                val_acc = correct / total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"  Gating network early stopping at epoch {epoch+1}")
                        break
        
        logger.info(f"  Gating network best val accuracy: {best_val_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using adaptive weighting from gating network."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using gating network weights."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.gating_network.eval()
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            # Get gating weights
            gate_weights = self.gating_network(X_tensor)  # (batch, n_models)
            
            # Get predictions from each base model
            ensemble_proba = np.zeros((len(X), self.n_classes_))
            
            for i, model_config in enumerate(self.models):
                model = self._fitted_models[model_config.name]['model']
                model.eval()
                
                if self._fitted_models[model_config.name]['is_torch']:
                    with torch.no_grad():
                        X_dev = X_tensor
                        outputs = model(X_dev)
                        proba = torch.softmax(outputs, dim=1).cpu().numpy()
                else:
                    proba = model.predict_proba(X)
                
                ensemble_proba += proba * gate_weights[:, i].cpu().numpy()
        
        return ensemble_proba


# ============================================================================
# Factory Function
# ============================================================================

def create_ensemble(
    models: List[ModelConfig],
    strategy: str = 'voting',
    config: Optional[EnsembleConfig] = None,
    **kwargs,
) -> BaseEnsemble:
    """
    Factory function to create ensemble by strategy.
    
    Args:
        models: List of ModelConfig objects
        strategy: Ensemble strategy ('voting', 'stacking', 'weighted', 'bagging', 'adaptive')
        config: Optional EnsembleConfig
        **kwargs: Additional arguments passed to ensemble constructor
        
    Returns:
        Configured ensemble instance
        
    Example:
        models = [
            ModelConfig(model_type='eegnet', name='eegnet_1'),
            ModelConfig(model_type='csp', name='csp_1'),
            ModelConfig(model_type='riemann', name='riemann_1'),
        ]
        
        ensemble = create_ensemble(models, strategy='voting', voting_mode='soft')
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
    """
    ensemble_config = config or EnsembleConfig()
    ensemble_config.strategy = strategy
    
    if strategy == 'voting':
        voting_mode = kwargs.get('voting_mode', 'soft')
        return VotingEnsemble(models, ensemble_config, voting_mode)
    
    elif strategy == 'stacking':
        meta_learner = kwargs.get('meta_learner', 'logistic')
        return StackingEnsemble(models, ensemble_config, meta_learner)
    
    elif strategy == 'weighted':
        return WeightedEnsemble(models, ensemble_config)
    
    elif strategy == 'bagging':
        n_bagging = kwargs.get('n_bagging', 10)
        bagging_ratio = kwargs.get('bagging_ratio', 0.8)
        return BaggingEnsemble(models, ensemble_config, n_bagging, bagging_ratio)
    
    elif strategy == 'adaptive':
        gating_hidden_dim = kwargs.get('gating_hidden_dim', 64)
        return AdaptiveEnsemble(models, ensemble_config, gating_hidden_dim)
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Valid options: voting, stacking, weighted, bagging, adaptive"
        )


# Export
__all__ = [
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
