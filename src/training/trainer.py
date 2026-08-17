"""
Training utilities for BCI models
Provides unified training loop, optimizer creation, and utilities
"""

import os
import sys
import logging
import random
import time
from typing import Dict, Optional, Tuple, Any, Callable, List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam, SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    early_stopping_patience: int = 15
    label_smoothing: float = 0.0
    gradient_clip: Optional[float] = None
    log_interval: int = 10
    save_best_only: bool = True
    device: str = 'auto'


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data.
    
    Handles conversion from numpy arrays to tensors and
    optional data augmentation.
    
    Args:
        X: Feature data (samples, channels, times)
        y: Labels (samples,)
        augmentor: Optional EEGAugmentor for data augmentation
        transform: Optional transform function
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augmentor: Optional[Any] = None,
        transform: Optional[Callable] = None,
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.augmentor = augmentor
        self.transform = transform
        self._augmented = False
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.X[idx]
        
        # Apply augmentation (on-the-fly)
        if self.augmentor is not None and not self._augmented:
            x_np = x.numpy()
            x_np = self.augmentor.augment(x_np)
            x = torch.FloatTensor(x_np)
        
        if self.y is not None:
            return x, self.y[idx]
        
        return x
    
    def enable_augmentation(self) -> None:
        """Enable augmentation mode."""
        self._augmented = False  # Augment every time
    
    def disable_augmentation(self) -> None:
        """Disable augmentation mode."""
        self._augmented = True  # Don't augment
    
    def set_augmented(self, value: bool) -> None:
        """Set augmented flag."""
        self._augmented = value


class Trainer:
    """
    Unified trainer for BCI classification models.
    
    Provides:
    - Flexible training loop with callbacks
    - Multiple optimizer and scheduler options
    - Early stopping
    - Metrics tracking
    - Model checkpointing
    
    Args:
        model: PyTorch model
        config: Training configuration
        device: Device to train on
        optimizer: Optimizer (auto-created if None)
        scheduler: Learning rate scheduler (optional)
        criterion: Loss function
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        
        # Set device
        if device is None or device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Optimizer
        if optimizer is None:
            self.optimizer = create_optimizer(
                model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Loss function
        if criterion is None:
            if self.config.label_smoothing > 0:
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        
        # Callbacks
        self.callbacks = []
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def add_callback(self, callback: Callable) -> 'Trainer':
        """Add a callback function."""
        self.callbacks.append(callback)
        return self
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Run training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Callbacks: on_epoch_start
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_start'):
                    callback.on_epoch_start(self, epoch)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            if epoch % self.config.log_interval == 0 or epoch == epochs - 1:
                msg = f"Epoch {epoch+1}/{epochs} | "
                msg += f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                logger.info(msg)
            
            # Early stopping check
            if val_acc is not None:
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.patience_counter = 0
                    if self.config.save_best_only:
                        self.save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            elif train_loss < self.best_loss:
                self.best_loss = train_loss
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
            
            # Callbacks: on_epoch_end
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self, epoch, train_loss, val_loss, train_acc, val_acc)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_accuracy:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Training", leave=False)
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(loader), correct / total
    
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset."""
        return self._validate(loader)
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = outputs.max(1)
        
        return predicted.cpu().numpy()
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
        }, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Checkpoint loaded from {path}")


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_type: 'adam', 'sgd', 'adamw'
        lr: Learning rate
        weight_decay: L2 regularization
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'plateau',
    **kwargs,
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'plateau', 'cosine', 'step'
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler
    """
    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            **kwargs
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def prepare_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    augmentor: Optional[Any] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare PyTorch data loaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        batch_size: Batch size
        augmentor: Data augmentor (optional)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train, augmentor=augmentor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = EEGDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    
    return train_loader, val_loader


# Export
__all__ = [
    "Trainer",
    "TrainingConfig",
    "EEGDataset",
    "create_optimizer",
    "create_scheduler",
    "set_seed",
    "prepare_data_loaders",
]
