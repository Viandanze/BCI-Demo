"""
Advanced BCI Models Module
Implements state-of-the-art neural network architectures for EEG classification.

Models:
    1. ShallowConvNet - Schirrmeister 2017, shallow wide CNN for EEG
    2. EEG-Conformer - Song 2023, transformer-based model with attention
    3. TCN - Temporal Convolutional Network with dilated convolutions

All models follow the same interface:
    - Input: X (batch, 1, channels, time_points)
    - Output: logits (batch, n_classes)
    - Optional: feature vector for visualization/transfer

Author: BCI_Projects
"""

import math
import logging
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class Swish(nn.Module):
    """
    Swish activation function.
    
    swish(x) = x * sigmoid(x)
    Often outperforms ReLU in deep networks.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class ResBlock1D(nn.Module):
    """
    1D Residual Block for temporal feature extraction.
    
    Architecture:
        Input -> Conv1D -> BN -> Activation -> Conv1D -> BN -> Add -> Activation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = F.elu(out)
        
        return out


# ============================================================================
# ShallowConvNet (Schirrmeister 2017)
# ============================================================================

class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet - Wide shallow CNN for EEG processing.
    
    From "Deep learning with convolutional neural networks for EEG decoding 
    and visualization" (Schirrmeister et al., 2017)
    
    Architecture:
        Input (1, C, T)
        -> Temporal Conv (40, 1, 25) + ELU + BatchNorm
        -> Spatial Conv (40, C, 1) + ELU + AvgPool
        -> Dropout
        -> Reshape
        -> FC (n_classes)
        -> LogSoftmax
    
    Key characteristics:
        - Wide temporal convolution captures frequency information
        - Spatial convolution learns common spatial patterns
        - No bias in conv layers (BN handles it)
        - Designed for raw EEG input
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of output classes
        n_filters: Number of temporal filters (default: 40)
        filter_time_length: Temporal convolution length (default: 25)
        pool_time_length: Pooling window length (default: 75)
        pool_time_stride: Pooling stride (default: 15)
        dropout_rate: Dropout probability (default: 0.5)
        in_channels: Input channels (default: 1)
    
    Example:
        model = ShallowConvNet(n_channels=64, n_times=641, n_classes=4)
        x = torch.randn(32, 1, 64, 641)
        logits = model(x)
        print(logits.shape)  # torch.Size([32, 4])
    """
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        n_filters: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout_rate: float = 0.5,
        in_channels: int = 1,
    ):
        super().__init__()
        
        # Validate parameters
        if n_filters <= 0:
            raise ValueError(f"n_filters must be positive, got {n_filters}")
        if filter_time_length <= 0:
            raise ValueError(f"filter_time_length must be positive")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1)")
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=(1, filter_time_length),
            stride=1,
            padding=(0, filter_time_length // 2),
            bias=False,
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=(n_channels, 1),
            stride=1,
            bias=False,
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(n_filters)
        
        # Pooling
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_length),
            stride=(1, pool_time_stride),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate output size
        time_after_pool = self._calculate_output_time(n_times, pool_time_length, pool_time_stride)
        self.final_features = n_filters * time_after_pool
        
        if self.final_features <= 0:
            logger.warning(f"Feature dimension calculated as {self.final_features}")
            self.final_features = max(n_filters, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_features, n_classes),
            nn.LogSoftmax(dim=1),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"ShallowConvNet initialized: {self.count_parameters()} parameters")
    
    def _calculate_output_time(self, n_times: int, pool_len: int, pool_stride: int) -> int:
        """Calculate output time dimension after pooling."""
        return math.floor((n_times - pool_len) / pool_stride) + 1
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, channels, time_points)
            
        Returns:
            Log probabilities (batch, n_classes)
        """
        # Temporal convolution
        x = self.temporal_conv(x)  # (B, n_filters, C, T)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, n_filters, 1, T)
        
        # Batch norm + ELU
        x = self.bn(x)
        x = F.elu(x)
        
        # Pooling
        x = self.pool(x)  # (B, n_filters, 1, T')
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'ShallowConvNet',
            'n_channels': self.n_channels,
            'n_times': self.n_times,
            'n_classes': self.n_classes,
            'n_filters': self.n_filters,
            'dropout_rate': self.dropout_rate,
            'n_parameters': self.count_parameters(),
        }


# ============================================================================
# EEG-Conformer (Song 2023)
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for EEG signals.
    
    Splits input into patches and projects to embedding dimension.
    
    Args:
        patch_size: Number of time points per patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        patch_size: int = 25,
        in_channels: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Project each patch to embedding dimension
        # Conv over time dimension with kernel = patch_size, stride = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, int]:
        """
        Args:
            x: (batch, 1, channels, time_points)
            
        Returns:
            (batch * channels, n_patches, embed_dim), n_patches
        """
        batch, _, channels, time_points = x.shape
        
        # Reshape: (B, 1, C, T) -> (B*C, 1, T) -> (B*C, embed_dim, n_patches)
        x = x.view(batch * channels, 1, time_points)
        x = self.proj(x)  # (B*C, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B*C, n_patches, embed_dim)
        x = self.dropout(x)
        
        n_patches = x.shape[1]
        return x, n_patches


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for sequence modeling.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_patches, embed_dim)
            
        Returns:
            (batch, n_patches, embed_dim)
        """
        batch, n_patches, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch, n_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch, n_patches, self.embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with pre-norm architecture.
    
    Architecture:
        LayerNorm -> MHSA -> Dropout -> Residual
        LayerNorm -> FFN -> Dropout -> Residual
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim multiplier
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        x = x + self.dropout1(self.attn(self.norm1(x)))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class LocalConvBlock(nn.Module):
    """
    Local Convolutional block for capturing local temporal patterns.
    
    Complements the global attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2,
                     groups=embed_dim, bias=False),  # Depthwise
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1, bias=False),  # Pointwise
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_patches, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, n_patches)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x


class EEGConformer(nn.Module):
    """
    EEG-Conformer: Conformer-based architecture for EEG classification.
    
    From "EEG-Conformer: Convolutional Neural Network for EEG Classification
    with Hybrid Convolution and Self-Attention" (Song et al., 2023)
    
    Architecture:
        Input -> Patch Embedding
        -> [LocalConvBlock * N]
        -> [TransformerBlock * N] (with Conformer-style convolution module)
        -> Global Average Pool -> FC -> Output
    
    Key features:
        - Patch-based embedding for efficiency
        - Hybrid local convolution + global attention
        - Conformer-style multi-scale feature fusion
        - Suitable for longer EEG sequences
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of output classes
        embed_dim: Embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3)
        patch_size: Patch size in time points (default: 25)
        kernel_size: Local conv kernel size (default: 3)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        pool_type: Pooling type 'cls' or 'mean' (default: 'mean')
        in_channels: Input channels (default: 1)
    
    Example:
        model = EEGConformer(n_channels=64, n_times=641, n_classes=4)
        x = torch.randn(32, 1, 64, 641)
        logits = model(x)
        print(logits.shape)  # torch.Size([32, 4])
    """
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        patch_size: int = 25,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pool_type: str = 'mean',
        in_channels: int = 1,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pool_type = pool_type
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # Local convolution blocks
        self.local_convs = nn.ModuleList([
            LocalConvBlock(embed_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Transformer blocks
        self.transformers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Conformer-style fusion: Conv module interleaved with attention
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, 1),
            nn.BatchNorm1d(embed_dim * 2),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
        )
        
        # Global pooling and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"EEGConformer initialized: {self.count_parameters()} parameters")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, channels, time_points)
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (logits, features) if return_features else logits
        """
        batch, _, channels, time_points = x.shape
        
        # Patch embedding: (B, 1, C, T) -> (B*C, n_patches, embed_dim)
        x, n_patches = self.patch_embed(x)
        
        # Reshape for processing: (B*C, n_patches, embed_dim) -> (B, C*n_patches, embed_dim)
        x = x.view(batch, channels * n_patches, self.embed_dim)
        
        # Process through conformer layers
        features_list = []
        for i in range(self.num_layers):
            # Local convolution
            x_conv = self.local_convs[i](x)
            
            # Transformer
            x_trans = self.transformers[i](x)
            
            # Conformer-style fusion: half feed-forward convolution
            x_conv_t = x_conv.transpose(1, 2)
            x_conv_t = self.conv_fusion(x_conv_t)
            x_conv = x_conv_t.transpose(1, 2)
            
            # Gated additive fusion
            x = torch.tanh(x_trans) * torch.sigmoid(x_conv) + x
            features_list.append(x)
        
        # Global pooling
        x = self.norm(x)
        
        if self.pool_type == 'cls':
            # Use first token (CLS-like)
            x = x[:, 0]
        else:
            # Average pooling over sequence
            x = x.mean(dim=1)
        
        x = self.head_dropout(x)
        logits = self.classifier(x)
        
        if return_features:
            return logits, x
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'EEGConformer',
            'n_channels': self.n_channels,
            'n_times': self.n_times,
            'n_classes': self.n_classes,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'n_parameters': self.count_parameters(),
        }


# ============================================================================
# TCN (Temporal Convolutional Network)
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal Block with dilated convolution and residual connection.
    
    Architecture:
        Input -> Causal Conv -> Dropout -> Activation -> Causal Conv -> Dropout -> Add
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout rate
        causal: Whether to use causal convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.5,
        causal: bool = True,
    ):
        super().__init__()
        
        self.causal = causal
        padding = (kernel_size - 1) * dilation if causal else (kernel_size - 1) * dilation // 2
        
        # First conv block
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        
        # Ensure same length through downsample if needed
        self.downsample = None
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        
        # First conv
        out = self.conv1(x)
        if self.causal:
            # Cut off right side for causality
            out = out[:, :, :-self.conv1.padding[0]]
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # Second conv
        out = self.conv2(out)
        if self.causal:
            out = out[:, :, :-self.conv2.padding[0]]
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        # Match lengths if needed
        min_len = min(out.size(2), residual.size(2))
        out = out[:, :, :min_len]
        residual = residual[:, :, :min_len]
        
        return out + residual


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for EEG classification.
    
    Based on "An Empirical Evaluation of Generic Convolutional and 
    Recurrent Networks for Sequence Modeling" (Bai et al., 2018).
    
    Architecture:
        Input -> Initial Conv -> TCN Blocks (with increasing dilation)
        -> Global Pooling -> FC -> Output
    
    Key features:
        - Dilated causal convolutions for long-range dependencies
        - Residual connections for gradient flow
        - Exponential dilation growth: 1, 2, 4, 8, ...
        - Suitable for tasks requiring temporal context
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of output classes
        n_filters: Number of convolution filters (default: 32)
        kernel_size: Convolution kernel size (default: 7)
        n_layers: Number of TCN layers (default: 4)
        dropout: Dropout rate (default: 0.5)
        causal: Use causal convolutions (default: True)
        in_channels: Input channels (default: 1)
    
    Example:
        model = TCN(n_channels=64, n_times=641, n_classes=4)
        x = torch.randn(32, 1, 64, 641)
        logits = model(x)
        print(logits.shape)  # torch.Size([32, 4])
    """
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        n_filters: int = 32,
        kernel_size: int = 7,
        n_layers: int = 4,
        dropout: float = 0.5,
        causal: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_layers = n_layers
        
        # Calculate receptive field
        self.receptive_field = kernel_size * (2 ** n_layers - 1)
        logger.info(f"TCN receptive field: {self.receptive_field} time steps")
        
        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, (1, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        
        # TCN blocks with increasing dilation
        self.tcn_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TemporalBlock(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                )
            )
        
        # Global temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Spatial attention over channels
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(n_filters, n_filters // 4, 1),
            nn.ELU(),
            nn.Conv1d(n_filters // 4, n_channels, 1),
            nn.Sigmoid(),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_filters + n_channels, n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters, n_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"TCN initialized: {self.count_parameters()} parameters")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, channels, time_points)
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (logits, features) if return_features else logits
        """
        batch, _, channels, time_points = x.shape
        
        # Initial projection
        x = self.input_conv(x)  # (B, n_filters, C, T)
        
        # Reshape: (B, n_filters, C, T) -> (B, n_filters, C*T)
        x = x.view(batch, self.n_filters, channels * time_points)
        
        # Process through TCN layers
        for layer in self.tcn_layers:
            x = layer(x)
        
        # Temporal features (after TCN)
        x_temp = x
        temporal_features = self.pool(x_temp).squeeze(-1)  # (B, n_filters)
        
        # Spatial attention
        attn = self.spatial_attention(x_temp)  # (B, C, 1)
        
        # Apply attention to input (original channel dimension needed)
        # For simplicity, use average pooled channel-wise representation
        channel_features = attn.squeeze(-1)  # (B, C)
        
        # Combine temporal and spatial features
        combined = torch.cat([temporal_features, channel_features], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        if return_features:
            return logits, combined
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TCN',
            'n_channels': self.n_channels,
            'n_times': self.n_times,
            'n_classes': self.n_classes,
            'n_filters': self.n_filters,
            'n_layers': self.n_layers,
            'kernel_size': 7,  # Fixed
            'receptive_field': self.receptive_field,
            'n_parameters': self.count_parameters(),
        }


# ============================================================================
# Model Factory
# ============================================================================

def create_model(
    model_type: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create BCI models.
    
    Args:
        model_type: One of 'shallowconvnet', 'conformer', 'tcn'
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of output classes
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
        
    Example:
        model = create_model('eegnet', n_channels=64, n_times=641, n_classes=4)
    """
    model_type = model_type.lower()
    
    if model_type == 'shallowconvnet':
        return ShallowConvNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            n_filters=kwargs.get('n_filters', 40),
            filter_time_length=kwargs.get('filter_time_length', 25),
            dropout_rate=kwargs.get('dropout', 0.5),
        )
    elif model_type in ('eegconformer', 'conformer'):
        return EEGConformer(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            embed_dim=kwargs.get('embed_dim', 64),
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == 'tcn':
        return TCN(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            n_filters=kwargs.get('n_filters', 32),
            kernel_size=kwargs.get('kernel_size', 7),
            n_layers=kwargs.get('n_layers', 4),
            dropout=kwargs.get('dropout', 0.5),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: shallowconvnet, conformer, tcn")


# Export
__all__ = [
    "ShallowConvNet",
    "EEGConformer",
    "TCN",
    "create_model",
    "Swish",
    "ResBlock1D",
]
