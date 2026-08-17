"""
Transfer Learning Module for EEG-based BCI
实现跨受试者迁移学习，支持多种迁移策略

主要功能:
1. Fine-tuning策略: 冻结特征层/逐层解冻/差分学习率
2. Domain Adaptation: DANN + CORAL
3. Pretrain-Finetune流程
4. Leave-One-Subject-Out评估

Author: BCI_Projects
"""

import os
import copy
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from ..models.eegnet import EEGNet, EEGNetClassifier
from .trainer import Trainer, TrainingConfig, EEGDataset, set_seed

logger = logging.getLogger(__name__)


class FreezeStrategy(Enum):
    """冻结策略枚举"""
    NONE = "none"              # 不冻结，全部可训练
    FROZEN_BACKBONE = "frozen_backbone"  # 冻结backbone，仅训练分类头
    GRADUAL_UNFREEZE = "gradual_unfreeze"  # 逐层解冻
    DISCRIMINATIVE = "discriminative"  # 差分学习率


class TransferMode(Enum):
    """迁移模式枚举"""
    FINE_TUNE = "fine_tune"
    DANN = "dann"
    CORAL = "coral"
    PRETRAIN_FINETUNE = "pretrain_finetune"


@dataclass
class TransferConfig:
    """
    迁移学习配置数据类
    
    Attributes:
        mode: 迁移模式
        freeze_strategy: 冻结策略
        base_lr: 基础学习率
        feature_lr_factor: 特征层学习率因子
        classifier_lr_factor: 分类层学习率因子
        weight_decay: 权重衰减
        dann_lambda: DANN域对抗损失权重
        coral_alpha: CORAL对齐损失权重
        gradual_unfreeze_epochs: 逐层解冻的epoch间隔
        warmup_epochs: 预热epoch数
        device: 设备
        seed: 随机种子
    """
    mode: TransferMode = TransferMode.FINE_TUNE
    freeze_strategy: FreezeStrategy = FreezeStrategy.DISCRIMINATIVE
    base_lr: float = 0.001
    feature_lr_factor: float = 0.1
    classifier_lr_factor: float = 1.0
    weight_decay: float = 0.01
    dann_lambda: float = 1.0
    coral_alpha: float = 1.0
    gradual_unfreeze_epochs: int = 20
    warmup_epochs: int = 5
    device: str = "auto"
    seed: int = 42
    # 额外参数
    early_stopping_patience: int = 15
    batch_size: int = 64
    max_epochs: int = 100
    log_interval: int = 10
    
    def __post_init__(self):
        """类型转换"""
        if isinstance(self.mode, str):
            self.mode = TransferMode(self.mode)
        if isinstance(self.freeze_strategy, str):
            self.freeze_strategy = FreezeStrategy(self.freeze_strategy)


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (Gradient Reversal Layer)
    
    在前向传播时保持不变，在反向传播时反转梯度
    用于DANN (Domain Adversarial Neural Network)
    
    公式:
        Forward: y = x
        Backward: ∂L/∂x = -λ * ∂L/∂y
    
    Args:
        lambda_: 反转强度因子
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，保持不变"""
        return x
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        反向传播，反转梯度
        
        Args:
            grad_output: 来自上层的梯度
            
        Returns:
            反转后的梯度
        """
        return -self.lambda_ * grad_output


class DomainClassifier(nn.Module):
    """
    域分类器 (Domain Classifier)
    
    用于DANN方法，判断样本来自源域还是目标域
    包含GRL (Gradient Reversal Layer)
    
    结构:
        GlobalAvgPool -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        lambda_: float = 1.0,
    ):
        """
        Args:
            in_features: 输入特征维度
            hidden_dim: 隐藏层维度
            lambda_: GRL反转强度
        """
        super().__init__()
        
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_)
        
        # 域分类网络
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),  # 二分类: 源域/目标域
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征张量
            
        Returns:
            域预测概率
        """
        x = self.grl(x)
        x = self.classifier(x)
        return torch.sigmoid(x)


class TransferModel(nn.Module):
    """
    迁移学习模型包装器
    
    支持多种迁移策略:
    1. 标准迁移: backbone + 分类头
    2. DANN: backbone + 分类头 + 域分类器
    3. CORAL: backbone + 分类头 + CORAL损失计算
    
    Args:
        backbone: 特征提取器 (EEGNet)
        n_classes: 类别数
        use_domain_classifier: 是否使用域分类器
        dann_lambda: DANN的GRL强度
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        use_domain_classifier: bool = False,
        dann_lambda: float = 1.0,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.n_classes = n_classes
        self.use_domain_classifier = use_domain_classifier
        
        # 获取backbone的输出特征维度
        if hasattr(backbone, 'final_features'):
            feat_dim = backbone.final_features
        elif hasattr(backbone, 'classifier'):
            # 从分类器获取
            feat_dim = self._get_backbone_features_dim(backbone)
        else:
            feat_dim = 64  # 默认值
        
        # 分类头
        self.classifier = nn.Linear(feat_dim, n_classes)
        
        # 域分类器 (用于DANN)
        if use_domain_classifier:
            self.domain_classifier = DomainClassifier(
                in_features=feat_dim,
                hidden_dim=64,
                lambda_=dann_lambda,
            )
        else:
            self.domain_classifier = None
    
    def _get_backbone_features_dim(self, backbone: nn.Module) -> int:
        """推断backbone特征维度"""
        # 尝试通过前向传播推断
        try:
            dummy = torch.randn(1, 1, 64, 321)
            with torch.no_grad():
                # 获取池化前的特征
                x = backbone.conv_temporal(dummy)
                x = backbone.conv_spatial(x)
                x = backbone.pool1(x)
                x = backbone.dropout1(x)
                x = backbone.conv_separable_depth(x)
                x = backbone.conv_separable_point(x)
                x = backbone.pool2(x)
                return x.shape[1] * x.shape[2] * x.shape[3]
        except Exception:
            return 64
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征 (用于域分析)
        
        Args:
            x: 输入EEG数据
            
        Returns:
            特征向量
        """
        # 前向传播直到池化层
        x = self.backbone.conv_temporal(x)
        if hasattr(self.backbone, 'bn1'):
            x = self.backbone.bn1(x)
        x = F.elu(x)
        
        x = self.backbone.conv_spatial(x)
        if hasattr(self.backbone, 'bn2'):
            x = self.backbone.bn2(x)
        x = F.elu(x)
        x = self.backbone.pool1(x)
        x = self.backbone.dropout1(x)
        
        x = self.backbone.conv_separable_depth(x)
        if hasattr(self.backbone, 'bn3'):
            x = self.backbone.bn3(x)
        x = F.elu(x)
        x = self.backbone.conv_separable_point(x)
        x = F.elu(x)
        x = self.backbone.pool2(x)
        x = self.backbone.dropout2(x)
        
        # 展平
        features = x.view(x.size(0), -1)
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_domain: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据
            return_domain: 是否返回域预测
            
        Returns:
            包含分类logits和域预测的字典
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        
        result = {'logits': logits, 'features': features}
        
        if self.use_domain_classifier and return_domain:
            domain_pred = self.domain_classifier(features)
            result['domain_pred'] = domain_pred
        
        return result
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            x: 输入张量
            
        Returns:
            numpy特征数组
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(1)
            features = self.get_features(x)
            return features.cpu().numpy()


class TransferTrainer:
    """
    迁移学习训练器
    
    支持:
    1. 标准微调
    2. DANN域对抗迁移
    3. CORAL对齐
    4. 逐层解冻
    
    Args:
        model: TransferModel实例
        config: TransferConfig配置
    """
    
    def __init__(
        self,
        model: TransferModel,
        config: TransferConfig,
    ):
        self.model = model
        self.config = config
        
        # 设置设备
        if config.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        
        self.model.to(self.device)
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'domain_loss': [], 'coral_loss': [],
            'lr': [],
        }
        
        # 冻结策略应用
        self._apply_freeze_strategy()
    
    def _apply_freeze_strategy(self) -> None:
        """应用冻结策略"""
        if self.config.freeze_strategy == FreezeStrategy.NONE:
            return
        
        if self.config.freeze_strategy == FreezeStrategy.FROZEN_BACKBONE:
            # 冻结backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen, only classifier trainable")
        
        elif self.config.freeze_strategy == FreezeStrategy.GRADUAL_UNFREEZE:
            # 准备逐层解冻，但不立即执行
            self._unfreeze_schedule = []
            logger.info("Gradual unfreezing scheduled")
        
        elif self.config.freeze_strategy == FreezeStrategy.DISCRIMINATIVE:
            # 差分学习率在optimizer中设置
            logger.info("Discriminative learning rates configured")
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        创建优化器，支持差分学习率
        
        特征层: lr * feature_lr_factor
        分类层: lr * classifier_lr_factor
        """
        feature_params = []
        classifier_params = []
        
        # 分类头参数
        classifier_params.extend(list(self.model.classifier.parameters()))
        
        # 域分类器参数
        if self.model.domain_classifier is not None:
            classifier_params.extend(
                list(self.model.domain_classifier.parameters())
            )
        
        # Backbone参数
        feature_params.extend(list(self.model.backbone.parameters()))
        
        optimizer_params = [
            {'params': feature_params, 'lr': self.config.base_lr * self.config.feature_lr_factor},
            {'params': classifier_params, 'lr': self.config.base_lr * self.config.classifier_lr_factor},
        ]
        
        return AdamW(optimizer_params, weight_decay=self.config.weight_decay)
    
    def _get_scheduler(self, optimizer: torch.optim.Optimizer):
        """创建学习率调度器"""
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6,
        )
    
    def _compute_coral_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算CORAL损失
        
        CORAL = 1/4 * ||C_S - C_T||_F^2
        
        其中C_S和C_T是源域和目标域的协方差矩阵
        
        Args:
            source_features: 源域特征
            target_features: 目标域特征
            
        Returns:
            CORAL损失
        """
        def coral_cov(x: torch.Tensor) -> torch.Tensor:
            """计算协方差矩阵"""
            x = x - x.mean(dim=0, keepdim=True)
            # 使用d-dimensional向量计算协方差
            d = x.shape[1]
            # 添加l2正则化确保可逆
            cov = torch.mm(x.t(), x) / (x.shape[0] - 1) + torch.eye(d, device=x.device) * 1e-3
            return cov
        
        c_source = coral_cov(source_features)
        c_target = coral_cov(target_features)
        
        # Frobenius范数平方
        loss = torch.sum((c_source - c_target) ** 2) / 4
        return loss
    
    def _compute_dann_loss(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算DANN域对抗损失
        
        Args:
            domain_logits: 域预测logits
            domain_labels: 域标签 (0=源域, 1=目标域)
            
        Returns:
            域分类交叉熵损失
        """
        return F.binary_cross_entropy(domain_logits, domain_labels)
    
    def _train_epoch_dann(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        source_domain_label: float = 0.0,
        target_domain_label: float = 1.0,
    ) -> Dict[str, float]:
        """
        DANN训练一个epoch
        
        混合源域和目标域数据训练
        
        Args:
            loader: 包含(source_x, target_x, labels)的数据加载器
            optimizer: 优化器
            source_domain_label: 源域标签
            target_domain_label: 目标域标签
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        correct = 0
        total = 0
        
        for batch in loader:
            # 解包数据
            if len(batch) == 3:
                source_x, target_x, labels = batch
            else:
                source_x, labels = batch
                target_x = None
            
            # 源域数据
            source_x = source_x.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(source_x, return_domain=True)
            logits = outputs['logits']
            
            # 分类损失
            class_loss = F.cross_entropy(logits, labels)
            
            # 域对抗损失 (仅源域数据)
            if target_x is not None and self.model.domain_classifier is not None:
                target_x = target_x.to(self.device)
                
                # 合并数据用于域分类
                combined_x = torch.cat([source_x, target_x], dim=0)
                combined_outputs = self.model(combined_x, return_domain=True)
                
                # 域标签
                n_source = source_x.size(0)
                n_target = target_x.size(0)
                domain_labels = torch.tensor(
                    [source_domain_label] * n_source + [target_domain_label] * n_target,
                    device=self.device,
                ).unsqueeze(1).float()
                
                # 域损失
                domain_loss = self._compute_dann_loss(
                    combined_outputs['domain_pred'], domain_labels
                )
                
                # 总损失
                loss = class_loss + self.config.dann_lambda * domain_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_class_loss += class_loss.item()
                total_domain_loss += domain_loss.item()
                
                # 统计
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            else:
                loss = class_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_class_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        n_batches = max(len(loader), 1)
        return {
            'loss': total_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'domain_loss': total_domain_loss / n_batches,
            'accuracy': correct / total if total > 0 else 0.0,
        }
    
    def _train_epoch_coral(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        CORAL训练一个epoch
        
        交替使用源域和目标域数据
        
        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            optimizer: 优化器
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_coral_loss = 0.0
        correct = 0
        total = 0
        
        # 创建目标域迭代器
        target_iter = iter(target_loader)
        
        for source_batch in source_loader:
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            source_x, labels = source_batch
            target_x = target_batch if isinstance(target_batch, torch.Tensor) else target_batch[0]
            
            source_x = source_x.to(self.device)
            target_x = target_x.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # 提取特征
            source_features = self.model.get_features(source_x)
            target_features = self.model.get_features(target_x)
            
            # 分类损失
            logits = self.model.classifier(source_features)
            class_loss = F.cross_entropy(logits, labels)
            
            # CORAL损失
            coral_loss = self._compute_coral_loss(source_features, target_features)
            
            # 总损失
            loss = class_loss + self.config.coral_alpha * coral_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_coral_loss += coral_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        n_batches = max(len(source_loader), 1)
        return {
            'loss': total_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'coral_loss': total_coral_loss / n_batches,
            'accuracy': correct / total if total > 0 else 0.0,
        }
    
    def _train_epoch_standard(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """标准训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = F.cross_entropy(outputs['logits'], y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        n_batches = max(len(loader), 1)
        return {
            'loss': total_loss / n_batches,
            'accuracy': correct / total if total > 0 else 0.0,
        }
    
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = F.cross_entropy(outputs['logits'], y)
                
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        n_batches = max(len(loader), 1)
        return {
            'loss': total_loss / n_batches,
            'accuracy': correct / total if total > 0 else 0.0,
        }
    
    def _gradual_unfreeze_step(self, epoch: int) -> None:
        """
        逐层解冻步骤
        
        在指定epoch后解冻backbone的更多层
        
        Args:
            epoch: 当前epoch
        """
        if self.config.freeze_strategy != FreezeStrategy.GRADUAL_UNFREEZE:
            return
        
        # 转换为列表以便修改
        backbone_params = list(self.model.backbone.parameters())
        n_layers = len(backbone_params)
        unfreeze_after = self.config.gradual_unfreeze_epochs
        
        # 计算应该解冻多少层
        layers_to_unfreeze = min(
            (epoch // unfreeze_after) * (n_layers // 5),
            n_layers
        )
        
        for i, param in enumerate(backbone_params):
            if i >= n_layers - layers_to_unfreeze:
                param.requires_grad = True
        
        if layers_to_unfreeze > 0:
            logger.info(f"Epoch {epoch}: Unfroze {layers_to_unfreeze}/{n_layers} backbone layers")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        source_val_loader: Optional[DataLoader] = None,
        target_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            source_val_loader: 源域验证加载器 (用于DANN/CORAL)
            target_loader: 目标域加载器 (用于DANN/CORAL)
            epochs: 训练轮数
            
        Returns:
            训练历史
        """
        epochs = epochs or self.config.max_epochs
        
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # 逐层解冻
            self._gradual_unfreeze_step(epoch)
            
            # 训练
            if self.config.mode == TransferMode.DANN:
                metrics = self._train_epoch_dann(train_loader, optimizer)
            elif self.config.mode == TransferMode.CORAL and target_loader is not None:
                metrics = self._train_epoch_coral(
                    train_loader, target_loader, optimizer
                )
            else:
                metrics = self._train_epoch_standard(train_loader, optimizer)
            
            # 更新历史
            self.training_history['train_loss'].append(metrics.get('loss', 0))
            self.training_history['train_acc'].append(metrics.get('accuracy', 0))
            
            if 'domain_loss' in metrics:
                self.training_history['domain_loss'].append(metrics['domain_loss'])
            if 'coral_loss' in metrics:
                self.training_history['coral_loss'].append(metrics['coral_loss'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_acc'].append(val_metrics['accuracy'])
                
                # Early stopping
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # 学习率调度
            scheduler.step()
            self.training_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 日志
            if epoch % self.config.log_interval == 0:
                msg = f"Epoch {epoch+1}/{epochs} | "
                msg += f"Loss: {metrics.get('loss', 0):.4f}, "
                msg += f"Acc: {metrics.get('accuracy', 0):.4f}"
                if val_loader is not None:
                    msg += f" | Val Acc: {val_metrics['accuracy']:.4f}"
                logger.info(msg)
        
        return self.training_history
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            loader: 测试数据加载器
            
        Returns:
            评估指标
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                outputs = self.model(X)
                probs = F.softmax(outputs['logits'], dim=1)
                _, preds = outputs['logits'].max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = (all_preds == all_labels).mean()
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
        }
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'n_classes': self.model.n_classes,
                'use_domain_classifier': self.model.use_domain_classifier,
            },
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Model loaded from {path}")


def create_transfer_model(
    n_channels: int,
    n_times: int,
    n_classes: int,
    config: TransferConfig,
) -> TransferModel:
    """
    创建迁移学习模型
    
    Args:
        n_channels: EEG通道数
        n_times: 时间点数
        n_classes: 类别数
        config: 迁移学习配置
        
    Returns:
        TransferModel实例
    """
    # 创建EEGNet backbone
    backbone = EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=8,
        D=2,
        kernel_length=64,
        dropout_rate=0.5,
    )
    
    # 决定是否使用域分类器
    use_domain_classifier = (
        config.mode == TransferMode.DANN or
        config.freeze_strategy == FreezeStrategy.DISCRIMINATIVE
    )
    
    model = TransferModel(
        backbone=backbone,
        n_classes=n_classes,
        use_domain_classifier=use_domain_classifier,
        dann_lambda=config.dann_lambda,
    )
    
    return model


def pretrain_on_subjects(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    config: TransferConfig,
    val_subject: Optional[int] = None,
    n_epochs: int = 50,
) -> TransferModel:
    """
    多受试者预训练
    
    使用多个源域受试者的数据预训练模型
    
    Args:
        subject_data: 字典，key为受试者ID，value为(data, labels)
        config: 迁移学习配置
        val_subject: 验证用的受试者ID
        n_epochs: 每个受试者的预训练轮数
        
    Returns:
        预训练好的模型
    """
    # 合并所有受试者数据
    all_X = []
    all_y = []
    
    for subj_id, (X, y) in subject_data.items():
        if subj_id != val_subject:
            all_X.append(X)
            all_y.append(y)
    
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    
    # 随机打乱
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # 创建数据加载器
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    
    # 创建模型
    model = create_transfer_model(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=len(np.unique(y_train)),
        config=config,
    )
    
    # 创建训练器
    trainer = TransferTrainer(model, config)
    
    # 创建数据加载器
    train_dataset = EEGDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # 验证数据
    val_loader = None
    if val_subject is not None and val_subject in subject_data:
        X_val, y_val = subject_data[val_subject]
        val_dataset = EEGDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 预训练
    logger.info(f"Pretraining on {len(X_train)} samples for {n_epochs} epochs")
    trainer.train(train_loader, val_loader, epochs=n_epochs)
    
    return model


def finetune_on_target(
    pretrained_model: TransferModel,
    source_data: Tuple[np.ndarray, np.ndarray],
    target_data: Tuple[np.ndarray, np.ndarray],
    config: TransferConfig,
    n_epochs: Optional[int] = None,
) -> Tuple[TransferModel, Dict[str, Any]]:
    """
    在目标受试者上微调预训练模型
    
    Args:
        pretrained_model: 预训练模型
        source_data: 源域数据 (X, y)
        target_data: 目标域数据 (X, y)
        config: 迁移学习配置
        n_epochs: 微调轮数
        
    Returns:
        (微调后模型, 训练历史)
    """
    X_source, y_source = source_data
    X_target, y_target = target_data
    
    # 配置微调模式
    finetune_config = copy.deepcopy(config)
    finetune_config.mode = TransferMode.FINE_TUNE
    finetune_config.max_epochs = n_epochs or config.max_epochs
    
    # 重新初始化训练器（使用预训练模型）
    trainer = TransferTrainer(pretrained_model, finetune_config)
    
    # 创建数据加载器
    if config.mode == TransferMode.DANN:
        # DANN: 混合源域和目标域数据
        source_dataset = EEGDataset(X_source, y_source)
        target_dataset = EEGDataset(X_target, np.zeros(len(X_target)))  # 伪标签
        
        source_loader = DataLoader(
            source_dataset,
            batch_size=config.batch_size // 2,
            shuffle=True,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=config.batch_size // 2,
            shuffle=True,
        )
        
        # 创建配对加载器
        class PairedLoader:
            def __init__(self, source, target):
                self.source_iter = iter(source)
                self.target_iter = iter(target)
                self.source = source
                self.target = target
            
            def __iter__(self):
                self.source_iter = iter(self.source)
                self.target_iter = iter(self.target)
                return self
            
            def __next__(self):
                try:
                    source_batch = next(self.source_iter)
                except StopIteration:
                    raise StopIteration
                
                try:
                    target_batch = next(self.target_iter)
                except StopIteration:
                    self.target_iter = iter(self.target)
                    target_batch = next(self.target_iter)
                
                if len(source_batch) == 2:
                    source_x, labels = source_batch
                    return source_x, target_batch[0], labels
                else:
                    return source_batch, target_batch[0], source_batch[1]
            
            def __len__(self):
                return max(len(self.source), len(self.target))
        
        train_loader = PairedLoader(source_loader, target_loader)
        
        history = trainer.train(
            train_loader,
            target_loader=target_loader,
            epochs=n_epochs,
        )
    elif config.mode == TransferMode.CORAL:
        # CORAL: 分离的源域和目标域加载器
        source_dataset = EEGDataset(X_source, y_source)
        target_dataset = EEGDataset(X_target, np.zeros(len(X_target)))
        
        source_loader = DataLoader(source_dataset, batch_size=config.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=config.batch_size, shuffle=True)
        
        history = trainer.train(
            source_loader,
            target_loader=target_loader,
            epochs=n_epochs,
        )
    else:
        # 标准微调: 仅使用源域数据
        # 混合源域和部分目标域数据
        n_target_train = int(len(X_target) * 0.5)  # 50%目标域数据用于训练
        X_train = np.concatenate([X_source, X_target[:n_target_train]], axis=0)
        y_train = np.concatenate([y_source, y_target[:n_target_train]], axis=0)
        
        train_dataset = EEGDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # 验证使用剩余目标域数据
        if n_target_train < len(X_target):
            X_val = X_target[n_target_train:]
            y_val = y_target[n_target_train:]
            val_dataset = EEGDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        else:
            val_loader = None
        
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=n_epochs,
        )
    
    # 评估目标域性能
    X_test = X_target[n_target_train:] if config.mode != TransferMode.DANN else X_target
    y_test = y_target[n_target_train:] if config.mode != TransferMode.DANN else y_target
    
    if len(X_test) > 0:
        test_dataset = EEGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
        eval_results = trainer.evaluate(test_loader)
    else:
        eval_results = {'accuracy': 0.0}
    
    return pretrained_model, {
        'history': history,
        'test_accuracy': eval_results['accuracy'],
    }


def compute_domain_distance(
    source_features: np.ndarray,
    target_features: np.ndarray,
    method: str = 'coral',
) -> float:
    """
    计算域间距离
    
    Args:
        source_features: 源域特征
        target_features: 目标域特征
        method: 'coral', 'mmd', 'euclidean'
        
    Returns:
        域距离值
    """
    if method == 'coral':
        # CORAL距离: 协方差矩阵的Frobenius距离
        def covariance(x):
            x = x - x.mean(axis=0)
            return np.cov(x, rowvar=False)
        
        c_source = covariance(source_features)
        c_target = covariance(target_features)
        
        distance = np.sum((c_source - c_target) ** 2)
        return distance / 4
    
    elif method == 'mmd':
        # 最大均值差异 (简化为高斯核)
        def gaussian_kernel(x, y, sigma=1.0):
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            diff = np.sum(diff ** 2, axis=2)
            return np.exp(-diff / (2 * sigma ** 2))
        
        k_ss = gaussian_kernel(source_features, source_features).mean()
        k_tt = gaussian_kernel(target_features, target_features).mean()
        k_st = gaussian_kernel(source_features, target_features).mean()
        
        return k_ss + k_tt - 2 * k_st
    
    elif method == 'euclidean':
        # 均值欧氏距离
        return np.linalg.norm(
            source_features.mean(axis=0) - target_features.mean(axis=0)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


# 导出
__all__ = [
    'TransferConfig',
    'TransferMode',
    'FreezeStrategy',
    'TransferModel',
    'TransferTrainer',
    'DomainClassifier',
    'GradientReversalLayer',
    'create_transfer_model',
    'pretrain_on_subjects',
    'finetune_on_target',
    'compute_domain_distance',
]
