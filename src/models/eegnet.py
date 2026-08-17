"""
EEGNet v2 Implementation
Based on Lawhern et al. 2018 "EEGNet: A Compact CNN for EEG-based BCIs"

Implements the original architecture with configurable parameters:
- Temporal convolution with specified kernel length
- Depthwise spatial convolution
- Separable temporal convolution
- Classification head with dropout
"""

import math
import logging
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EEGNet(nn.Module):
    """
    EEGNet v2: Compact CNN for EEG-based Brain-Computer Interfaces.
    
    Architecture:
        Input -> Temporal Conv (F1 filters) -> Depthwise Spatial Conv (D*F1) 
              -> Separable Temporal Conv -> Classification Head
    
    The network uses depthwise and separable convolutions to reduce 
    parameters while maintaining performance.
    
    Args:
        n_channels: Number of EEG channels (electrodes)
        n_times: Number of time points per epoch
        n_classes: Number of output classes
        F1: Number of temporal filters (default: 8)
        D: Depth multiplier for spatial filters (default: 2)
            Total spatial filters = F1 * D
        kernel_length: Length of temporal convolution kernel (default: 64)
        dropout_rate: Dropout probability (default: 0.5)
        pool_size: Size of average pooling (default: 4)
        use_batchnorm: Add BatchNorm after each conv layer (default: False)
        in_channels: Input channels (default: 1 for EEG)
    
    Example:
        model = EEGNet(
            n_channels=48,
            n_times=641,  # 5s at 128Hz
            n_classes=2,
            F1=8,
            D=2,
            kernel_length=64,
            dropout_rate=0.5
        )
        
        # Forward pass
        x = torch.randn(32, 1, 48, 641)  # batch, ch, height, width
        output = model(x)
        print(output.shape)  # torch.Size([32, 2])
    """
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
        pool_size: int = 4,
        use_batchnorm: bool = False,
        in_channels: int = 1,
    ):
        super(EEGNet, self).__init__()
        
        # Validate inputs
        if F1 <= 0:
            raise ValueError(f"F1 must be positive, got {F1}")
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if kernel_length <= 0:
            raise ValueError(f"kernel_length must be positive, got {kernel_length}")
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        
        # Calculate padding for 'same' output size
        # Temporal conv output: (n_times - kernel_length + 2*padding) + 1
        # For 'same': padding = (kernel_length - 1) // 2
        self.temporal_padding = (kernel_length - 1) // 2
        
        # ===== Block 1: Temporal Convolution =====
        # Input: (batch, 1, channels, time_points)
        # Output: (batch, F1, channels, time_points)
        self.conv_temporal = nn.Conv2d(
            in_channels=in_channels,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, self.temporal_padding),
            bias=False if not use_batchnorm else True,
        )
        
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(F1)
        
        # ===== Block 2: Depthwise Spatial Convolution =====
        # Input: (batch, F1, channels, time_points)
        # Output: (batch, F1*D, 1, time_points)
        # Each spatial filter is applied to all time points for one temporal filter
        F2 = F1 * D
        self.conv_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(n_channels, 1),
            groups=F1,  # Depthwise: each input channel processed separately
            bias=False if not use_batchnorm else True,
        )
        
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(F2)
        
        # Pooling after spatial conv
        self.pool1 = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, pool_size),
        )
        
        # Dropout after first block
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # ===== Block 3: Separable Temporal Convolution =====
        # Separable = Depthwise Conv + Pointwise Conv
        # Input: (batch, F1*D, 1, time_points/pool_size)
        # First do depthwise temporal conv
        self.conv_separable_depth = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F2,  # Depthwise: each channel processed separately
            bias=False if not use_batchnorm else True,
        )
        
        # Then pointwise conv (1x1)
        self.conv_separable_point = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False if not use_batchnorm else True,
        )
        
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(F2)
        
        # Pooling after separable conv
        self.pool2 = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, pool_size),
        )
        
        # Dropout after second block
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # ===== Classification Head =====
        # Calculate the final feature dimension correctly
        # After temporal conv (same padding): n_times
        # After pool1: floor(n_times / pool_size)
        # After separable depthwise (kernel=16, pad=8): pool1_out + 1
        # After separable pointwise (1x1): same
        # After pool2: floor(separable_out / pool_size)
        time_after_pool1 = math.floor(n_times / pool_size)
        # Separable depthwise: kernel_size=(1,16), padding=(0,8)
        # output = floor((input + 2*8 - 16) / 1 + 1) = input + 1
        time_after_separable = time_after_pool1 + 1
        time_after_pool2 = math.floor(time_after_separable / pool_size)
        
        self.final_features = F2 * time_after_pool2
        
        if self.final_features <= 0:
            logger.warning(
                f"Calculated feature dimension is {self.final_features}. "
                f"This may cause issues. Check n_times={n_times} and pool_size={pool_size}."
            )
            self.final_features = max(F2, 1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_features, n_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"EEGNet initialized: {self.count_parameters()} parameters, "
                   f"final features: {self.final_features}")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv layers only (before classifier).
        
        Args:
            x: Input tensor of shape (batch, 1, channels, time_points)
            
        Returns:
            Feature tensor of shape (batch, F2, 1, time_after_pool2)
        """
        # Block 1: Temporal convolution
        x = self.conv_temporal(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.elu(x)
        
        # Block 2: Spatial convolution
        x = self.conv_spatial(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable temporal convolution
        x = self.conv_separable_depth(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.elu(x)
        x = self.conv_separable_point(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EEGNet.
        
        Args:
            x: Input tensor of shape (batch, 1, channels, time_points)
            
        Returns:
            Output logits of shape (batch, n_classes)
        """
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class EEGNetClassifier(nn.Module):
    """
    EEGNet wrapper with optional softmax output and utility methods.
    
    This class provides a higher-level interface for EEGNet including:
    - Automatic input dimension inference
    - Optional softmax output
    - Label smoothing loss support
    - Feature extraction mode
    """
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
        pool_size: int = 4,
        use_batchnorm: bool = False,
        apply_softmax: bool = True,
    ):
        """
        Initialize EEGNet classifier.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points per epoch
            n_classes: Number of output classes (default: 2)
            F1: Number of temporal filters (default: 8)
            D: Depth multiplier (default: 2)
            kernel_length: Temporal kernel length (default: 64)
            dropout_rate: Dropout probability (default: 0.5)
            pool_size: Pooling size (default: 4)
            use_batchnorm: Use batch normalization (default: False)
            apply_softmax: Apply softmax to output (default: True)
        """
        super(EEGNetClassifier, self).__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.apply_softmax = apply_softmax
        
        # Build EEGNet model
        self.backbone = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            F1=F1,
            D=D,
            kernel_length=kernel_length,
            dropout_rate=dropout_rate,
            pool_size=pool_size,
            use_batchnorm=use_batchnorm,
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, time) or (batch, 1, channels, time)
            return_features: If True, return features before classification
            
        Returns:
            Class predictions (probabilities or logits)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Forward through backbone conv layers
        conv_out = self.backbone.forward_features(x)
        
        # Classification
        features = torch.flatten(conv_out, 1)
        logits = self.backbone.classifier[-1](features)  # Linear layer
        
        if return_features:
            return features, logits
        
        if self.apply_softmax:
            return F.softmax(logits, dim=-1)
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'n_channels': self.n_channels,
            'n_times': self.n_times,
            'n_classes': self.n_classes,
            'F1': self.backbone.F1,
            'D': self.backbone.D,
            'kernel_length': self.backbone.kernel_length,
            'dropout_rate': self.backbone.dropout_rate,
            'pool_size': self.backbone.pool_size,
            'use_batchnorm': self.backbone.use_batchnorm,
            'n_parameters': self.backbone.count_parameters(),
        }
    
    def __repr__(self) -> str:
        config = self.get_config()
        return (
            f"EEGNetClassifier(\n"
            f"  n_channels={config['n_channels']}, "
            f"n_times={config['n_times']}, "
            f"n_classes={config['n_classes']}\n"
            f"  F1={config['F1']}, D={config['D']}, "
            f"kernel_length={config['kernel_length']}\n"
            f"  dropout={config['dropout_rate']}, "
            f"batchnorm={config['use_batchnorm']}\n"
            f"  Parameters: {config['n_parameters']:,}\n"
            f")"
        )


def create_eegnet(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    config: Optional[Dict[str, Any]] = None,
) -> EEGNetClassifier:
    """
    Factory function to create EEGNet model from config.
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes
        config: Optional configuration dictionary
        
    Returns:
        EEGNetClassifier instance
    """
    if config is None:
        config = {}
    
    return EEGNetClassifier(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=config.get('F1', 8),
        D=config.get('D', 2),
        kernel_length=config.get('kernel_length', 64),
        dropout_rate=config.get('dropout_rate', 0.5),
        pool_size=config.get('pool_size', 4),
        use_batchnorm=config.get('use_batchnorm', False),
    )


# Export
__all__ = [
    "EEGNet",
    "EEGNetClassifier",
    "create_eegnet",
]
