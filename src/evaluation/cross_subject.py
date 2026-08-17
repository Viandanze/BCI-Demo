"""
Cross-Subject Evaluation Module
跨受试者评估框架

主要功能:
1. LOSO (Leave-One-Subject-Out) 评估
2. K-fold 跨受试者评估
3. 域偏移分析
4. 受试者相似度矩阵
5. 自适应权重
6. 统计检验报告

Author: BCI_Projects
"""

import copy
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from ..models.eegnet import EEGNet
from ..training.transfer import (
    TransferModel, TransferConfig, TransferTrainer,
    create_transfer_model, TransferMode, FreezeStrategy,
)
from ..training.trainer import EEGDataset, set_seed

logger = logging.getLogger(__name__)


@dataclass
class SubjectResult:
    """
    单个受试者的评估结果
    
    Attributes:
        subject_id: 受试者ID
        accuracy: 分类准确率
        balanced_accuracy: 平衡准确率
        predictions: 预测标签
        labels: 真实标签
        probabilities: 预测概率
        training_samples: 训练样本数
        test_samples: 测试样本数
        domain_distances: 到其他域的距离
    """
    subject_id: int
    accuracy: float
    balanced_accuracy: float
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    training_samples: int = 0
    test_samples: int = 0
    domain_distances: Optional[Dict[int, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'subject_id': int(self.subject_id),
            'accuracy': float(self.accuracy),
            'balanced_accuracy': float(self.balanced_accuracy),
            'training_samples': int(self.training_samples),
            'test_samples': int(self.test_samples),
        }


@dataclass
class CrossSubjectReport:
    """
    跨受试者评估完整报告
    
    Attributes:
        results: 每个受试者的结果
        mean_accuracy: 平均准确率
        std_accuracy: 准确率标准差
        subject_similarity_matrix: 受试者相似度矩阵
        domain_shift_analysis: 域偏移分析结果
        statistical_tests: 统计检验结果
        training_config: 使用的训练配置
    """
    results: Dict[int, SubjectResult]
    mean_accuracy: float
    std_accuracy: float
    subject_similarity_matrix: Optional[np.ndarray] = None
    domain_shift_analysis: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mean_accuracy': float(self.mean_accuracy),
            'std_accuracy': float(self.std_accuracy),
            'n_subjects': len(self.results),
            'n_success': sum(1 for r in self.results.values() if r.accuracy > 0),
            'subject_results': {k: v.to_dict() for k, v in self.results.items()},
            'subject_similarity_matrix': (
                self.subject_similarity_matrix.tolist() 
                if self.subject_similarity_matrix is not None else None
            ),
            'domain_shift_analysis': self.domain_shift_analysis,
            'statistical_tests': self.statistical_tests,
        }
    
    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            "=" * 60,
            "Cross-Subject Evaluation Report",
            "=" * 60,
            f"Number of subjects: {len(self.results)}",
            f"Mean accuracy: {self.mean_accuracy:.4f} ({self.mean_accuracy*100:.2f}%)",
            f"Std accuracy: {self.std_accuracy:.4f}",
            f"Min accuracy: {min(r.accuracy for r in self.results.values()):.4f}",
            f"Max accuracy: {max(r.accuracy for r in self.results.values()):.4f}",
            "",
            "Per-subject results:",
            "-" * 40,
        ]
        
        for subj_id in sorted(self.results.keys()):
            r = self.results[subj_id]
            lines.append(
                f"  Subject {subj_id:3d}: "
                f"Acc={r.accuracy:.4f} "
                f"(train={r.training_samples}, test={r.test_samples})"
            )
        
        return "\n".join(lines)


class CrossSubjectEvaluator:
    """
    跨受试者评估器
    
    支持多种评估策略:
    1. Leave-One-Subject-Out (LOSO)
    2. K-fold 按受试者分组
    3. 源域选择
    
    Args:
        model_fn: 模型创建函数
        config: 训练配置
        device: 设备
    """
    
    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        config: Optional[TransferConfig] = None,
        device: str = "auto",
    ):
        self.config = config or TransferConfig()
        self.device = device
        
        # 模型创建函数
        if model_fn is None:
            self.model_fn = self._default_model_fn
        else:
            self.model_fn = model_fn
        
        # 存储特征提取器（用于域分析）
        self.feature_extractor = None
        
        # 评估结果
        self.results: Dict[int, SubjectResult] = {}
    
    def _default_model_fn(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
    ) -> TransferModel:
        """默认模型创建函数"""
        return create_transfer_model(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            config=self.config,
        )
    
    def _extract_features(
        self,
        X: np.ndarray,
        model: Optional[TransferModel] = None,
    ) -> np.ndarray:
        """
        提取特征
        
        Args:
            X: 输入数据
            model: 可选的模型（用于获取特征维度）
            
        Returns:
            标准化后的特征
        """
        if self.feature_extractor is None:
            # 使用随机初始化的网络提取特征
            n_channels = X.shape[1]
            n_times = X.shape[2]
            
            dummy_model = self.model_fn(n_channels, n_times, n_classes=2)
            dummy_model.to(self.device)
            dummy_model.eval()
            
            with torch.no_grad():
                if X.ndim == 3:
                    X = X[:, np.newaxis, :, :]  # 添加通道维度
                X_tensor = torch.FloatTensor(X).to(self.device)
                features = dummy_model.get_features(X_tensor)
                return features.cpu().numpy()
        
        # 使用训练好的模型
        model.eval()
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, np.newaxis, :, :]
            X_tensor = torch.FloatTensor(X).to(self.device)
            features = model.get_features(X_tensor)
            return features.cpu().numpy()
    
    def loso_evaluate(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        epochs: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> CrossSubjectReport:
        """
        Leave-One-Subject-Out 评估
        
        每个受试者轮流作为测试集，其余作为训练集
        
        Args:
            subject_data: 字典，key为受试者ID，value为(data, labels)
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印详细信息
            
        Returns:
            CrossSubjectReport
        """
        subject_ids = sorted(subject_data.keys())
        n_subjects = len(subject_ids)
        
        if verbose:
            logger.info(f"LOSO evaluation: {n_subjects} subjects")
        
        self.results = {}
        
        for i, test_subject in enumerate(subject_ids):
            if verbose:
                logger.info(f"[{i+1}/{n_subjects}] Testing on subject {test_subject}")
            
            # 准备训练数据（排除测试受试者）
            train_subjects = [s for s in subject_ids if s != test_subject]
            
            X_train_list = []
            y_train_list = []
            
            for subj_id in train_subjects:
                X, y = subject_data[subj_id]
                X_train_list.append(X)
                y_train_list.append(y)
            
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            
            # 测试数据
            X_test, y_test = subject_data[test_subject]
            
            # 创建模型
            n_channels = X_train.shape[1]
            n_times = X_train.shape[2]
            n_classes = len(np.unique(y_train))
            
            model = self.model_fn(n_channels, n_times, n_classes)
            trainer = TransferTrainer(model, self.config)
            
            # 创建数据加载器
            train_dataset = EEGDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            
            test_dataset = EEGDataset(X_test, y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            
            # 训练
            try:
                trainer.train(train_loader, epochs=epochs)
                
                # 评估
                eval_results = trainer.evaluate(test_loader)
                
                self.results[test_subject] = SubjectResult(
                    subject_id=test_subject,
                    accuracy=eval_results['accuracy'],
                    balanced_accuracy=balanced_accuracy_score(
                        eval_results['labels'],
                        eval_results['predictions']
                    ),
                    predictions=eval_results['predictions'],
                    labels=eval_results['labels'],
                    probabilities=eval_results.get('probabilities'),
                    training_samples=len(X_train),
                    test_samples=len(X_test),
                )
                
                if verbose:
                    logger.info(
                        f"  Subject {test_subject}: "
                        f"Acc={eval_results['accuracy']:.4f}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to evaluate subject {test_subject}: {e}")
                self.results[test_subject] = SubjectResult(
                    subject_id=test_subject,
                    accuracy=0.0,
                    balanced_accuracy=0.0,
                    predictions=np.array([]),
                    labels=y_test,
                    training_samples=len(X_train),
                    test_samples=len(X_test),
                )
        
        # 计算统计信息
        accuracies = [r.accuracy for r in self.results.values()]
        balanced_accs = [r.balanced_accuracy for r in self.results.values()]
        
        report = CrossSubjectReport(
            results=self.results,
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
        )
        
        # 添加域偏移分析
        report.domain_shift_analysis = self._analyze_domain_shift(subject_data)
        
        # 添加统计检验
        report.statistical_tests = self._perform_statistical_tests()
        
        if verbose:
            logger.info(f"\nLOSO Complete: Mean Acc={report.mean_accuracy:.4f} ± {report.std_accuracy:.4f}")
        
        return report
    
    def kfold_subject_evaluate(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        n_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> CrossSubjectReport:
        """
        K-fold 跨受试者评估
        
        按受试者ID分组进行K折交叉验证
        
        Args:
            subject_data: 字典，key为受试者ID，value为(data, labels)
            n_folds: 折数
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印详细信息
            
        Returns:
            CrossSubjectReport
        """
        subject_ids = sorted(subject_data.keys())
        n_subjects = len(subject_ids)
        
        if verbose:
            logger.info(f"K-fold subject evaluation: {n_folds} folds, {n_subjects} subjects")
        
        # 按受试者分组
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        self.results = {}
        fold_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(subject_ids)):
            if verbose:
                logger.info(f"Fold {fold+1}/{n_folds}")
            
            # 获取训练和测试受试者
            train_subjects = [subject_ids[i] for i in train_idx]
            test_subjects = [subject_ids[i] for i in test_idx]
            
            if verbose:
                logger.info(f"  Train subjects: {train_subjects}")
                logger.info(f"  Test subjects: {test_subjects}")
            
            # 收集数据
            X_train_list = []
            y_train_list = []
            
            for subj_id in train_subjects:
                X, y = subject_data[subj_id]
                X_train_list.append(X)
                y_train_list.append(y)
            
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            
            X_test_list = []
            y_test_list = []
            
            for subj_id in test_subjects:
                X, y = subject_data[subj_id]
                X_test_list.append(X)
                y_test_list.append(y)
            
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
            
            # 创建模型
            n_channels = X_train.shape[1]
            n_times = X_train.shape[2]
            n_classes = len(np.unique(y_train))
            
            model = self.model_fn(n_channels, n_times, n_classes)
            trainer = TransferTrainer(model, self.config)
            
            # 创建数据加载器
            train_dataset = EEGDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            
            test_dataset = EEGDataset(X_test, y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            
            # 训练和评估
            try:
                trainer.train(train_loader, epochs=epochs)
                eval_results = trainer.evaluate(test_loader)
                
                fold_acc = eval_results['accuracy']
                fold_accuracies.append(fold_acc)
                
                if verbose:
                    logger.info(f"  Fold {fold+1} accuracy: {fold_acc:.4f}")
                
                # 记录每个测试受试者的结果
                start_idx = 0
                for subj_id in test_subjects:
                    _, y_subj = subject_data[subj_id]
                    n_samples = len(y_subj)
                    
                    self.results[subj_id] = SubjectResult(
                        subject_id=subj_id,
                        accuracy=eval_results['accuracy'],  # 同一fold使用相同模型
                        balanced_accuracy=balanced_accuracy_score(
                            eval_results['labels'],
                            eval_results['predictions']
                        ),
                        predictions=eval_results['predictions'][start_idx:start_idx+n_samples],
                        labels=y_test[start_idx:start_idx+n_samples],
                        probabilities=eval_results.get('probabilities'),
                        training_samples=len(X_train),
                        test_samples=n_samples,
                    )
                    start_idx += n_samples
                    
            except Exception as e:
                logger.error(f"Fold {fold+1} failed: {e}")
                fold_accuracies.append(0.0)
        
        # 计算统计信息
        accuracies = [r.accuracy for r in self.results.values()]
        
        report = CrossSubjectReport(
            results=self.results,
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
        )
        
        if verbose:
            logger.info(
                f"\nK-fold Complete: Mean Acc={report.mean_accuracy:.4f} ± {report.std_accuracy:.4f}"
            )
        
        return report
    
    def compute_subject_similarity(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        method: str = 'coral',
    ) -> Tuple[np.ndarray, List[int]]:
        """
        计算受试者间相似度矩阵
        
        基于特征分布的相似性
        
        Args:
            subject_data: 受试者数据
            method: 相似度计算方法 ('coral', 'mmd', 'euclidean')
            
        Returns:
            (相似度矩阵, 受试者ID列表)
        """
        subject_ids = sorted(subject_data.keys())
        n_subjects = len(subject_ids)
        
        # 提取每个受试者的特征统计量
        subject_stats = {}
        
        for subj_id in subject_ids:
            X, _ = subject_data[subj_id]
            
            # 简化为计算均值和协方差
            mean = X.mean(axis=(0, 2))  # 通道维度的均值
            # 计算通道间协方差
            X_flat = X.reshape(X.shape[0], -1)
            cov = np.cov(X_flat, rowvar=False)
            
            # 采样以减少计算量
            n_samples = min(100, X.shape[0])
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
            X_sample_flat = X_sample.reshape(n_samples, -1)
            
            subject_stats[subj_id] = {
                'mean': X_sample_flat.mean(axis=0),
                'cov': np.cov(X_sample_flat, rowvar=False),
                'features': X_sample_flat,
            }
        
        # 计算距离矩阵
        distance_matrix = np.zeros((n_subjects, n_subjects))
        
        for i, subj_i in enumerate(subject_ids):
            for j, subj_j in enumerate(subject_ids):
                if i == j:
                    distance_matrix[i, j] = 0.0
                elif i < j:
                    feat_i = subject_stats[subj_i]['features']
                    feat_j = subject_stats[subj_j]['features']
                    
                    if method == 'coral':
                        # CORAL距离
                        cov_i = subject_stats[subj_i]['cov']
                        cov_j = subject_stats[subj_j]['cov']
                        dist = np.sum((cov_i - cov_j) ** 2) / 4
                    elif method == 'euclidean':
                        # 均值欧氏距离
                        dist = np.linalg.norm(
                            subject_stats[subj_i]['mean'] - 
                            subject_stats[subj_j]['mean']
                        )
                    elif method == 'mmd':
                        # 最大均值差异
                        dist = self._compute_mmd(
                            feat_i[:50], 
                            feat_j[:50]
                        )
                    else:
                        dist = 0.0
                    
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        # 转换为相似度（距离越小，相似度越高）
        max_dist = distance_matrix.max()
        similarity_matrix = 1 - distance_matrix / (max_dist + 1e-6)
        
        return similarity_matrix, subject_ids
    
    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sigma: float = 1.0,
    ) -> float:
        """
        计算最大均值差异 (Maximum Mean Discrepancy)
        
        Args:
            X: 源域特征
            Y: 目标域特征
            sigma: 高斯核带宽
            
        Returns:
            MMD值
        """
        def gaussian_kernel(x, y, sigma):
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            diff = np.sum(diff ** 2, axis=2)
            return np.exp(-diff / (2 * sigma ** 2))
        
        k_xx = gaussian_kernel(X, X, sigma).mean()
        k_yy = gaussian_kernel(Y, Y, sigma).mean()
        k_xy = gaussian_kernel(X, Y, sigma).mean()
        
        return k_xx + k_yy - 2 * k_xy
    
    def _analyze_domain_shift(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        分析域偏移
        
        计算不同受试者之间的域偏移统计量
        
        Args:
            subject_data: 受试者数据
            
        Returns:
            域偏移分析结果
        """
        subject_ids = sorted(subject_data.keys())
        n_subjects = len(subject_ids)
        
        # 计算每对受试者之间的域距离
        distances = []
        
        for i, subj_i in enumerate(subject_ids):
            for j, subj_j in enumerate(subject_ids):
                if i < j:
                    X_i, _ = subject_data[subj_i]
                    X_j, _ = subject_data[subj_j]
                    
                    # 采样
                    n = min(50, X_i.shape[0], X_j.shape[0])
                    X_i_sample = X_i[:n].reshape(n, -1)
                    X_j_sample = X_j[:n].reshape(n, -1)
                    
                    # 计算CORAL距离
                    dist = self._compute_mmd(X_i_sample, X_j_sample)
                    distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'median_distance': float(np.median(distances)),
        }
    
    def _perform_statistical_tests(
        self,
    ) -> Dict[str, Any]:
        """
        执行统计检验
        
        分析跨受试者准确率的统计显著性
        
        Returns:
            统计检验结果
        """
        if len(self.results) < 2:
            return {}
        
        accuracies = np.array([r.accuracy for r in self.results.values()])
        
        # 基本统计
        result = {
            'n_subjects': len(accuracies),
            'mean': float(accuracies.mean()),
            'std': float(accuracies.std()),
            'median': float(np.median(accuracies)),
            'q25': float(np.percentile(accuracies, 25)),
            'q75': float(np.percentile(accuracies, 75)),
        }
        
        # 单样本t检验：是否显著高于随机水平
        n_classes = 2  # 假设二分类
        random_level = 1.0 / n_classes
        
        if len(accuracies) > 1:
            t_stat, p_value = stats.ttest_1samp(accuracies, random_level)
            result['ttest_vs_random'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
            }
        
        # 置信区间
        if len(accuracies) > 1:
            ci = stats.t.interval(
                0.95, 
                len(accuracies) - 1, 
                loc=accuracies.mean(), 
                scale=stats.sem(accuracies)
            )
            result['confidence_interval_95'] = {
                'lower': float(ci[0]),
                'upper': float(ci[1]),
            }
        
        return result
    
    def compute_adaptive_weights(
        self,
        source_subject: int,
        target_subject: int,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        method: str = 'domain_distance',
    ) -> Dict[int, float]:
        """
        计算自适应样本权重
        
        根据源域样本与目标域的相似度调整权重
        
        Args:
            source_subject: 源域受试者ID
            target_subject: 目标域受试者ID
            subject_data: 所有受试者数据
            method: 加权方法
            
        Returns:
            受试者ID到权重的映射
        """
        if method == 'domain_distance':
            # 基于域距离的权重
            X_target, _ = subject_data[target_subject]
            target_mean = X_target.mean(axis=(0, 2))
            
            weights = {}
            total_weight = 0.0
            
            for subj_id, (X, _) in subject_data.items():
                if subj_id == target_subject:
                    continue
                
                X_mean = X.mean(axis=(0, 2))
                # 距离越小，权重越大
                distance = np.linalg.norm(X_mean - target_mean)
                weight = 1.0 / (distance + 1e-6)
                
                weights[subj_id] = weight
                total_weight += weight
            
            # 归一化
            for subj_id in weights:
                weights[subj_id] /= total_weight
            
            return weights
        
        elif method == 'similarity':
            # 基于相似度矩阵的权重
            similarity, subject_ids = self.compute_subject_similarity(
                subject_data, method='coral'
            )
            
            target_idx = subject_ids.index(target_subject)
            
            weights = {}
            for i, subj_id in enumerate(subject_ids):
                if subj_id != target_subject:
                    weights[subj_id] = max(0, similarity[target_idx, i])
            
            # 归一化
            total = sum(weights.values())
            if total > 0:
                for subj_id in weights:
                    weights[subj_id] /= total
            
            return weights
        
        else:
            # 均匀权重
            return {k: 1.0 for k in subject_data.keys() if k != target_subject}
    
    def generate_comparison_report(
        self,
        baseline_results: CrossSubjectReport,
        transfer_results: CrossSubjectReport,
    ) -> str:
        """
        生成迁移前后对比报告
        
        Args:
            baseline_results: 基线（无迁移）结果
            transfer_results: 迁移学习结果
            
        Returns:
            格式化的对比报告
        """
        lines = [
            "=" * 60,
            "Transfer Learning Comparison Report",
            "=" * 60,
            "",
            "Baseline (No Transfer):",
            f"  Mean Accuracy: {baseline_results.mean_accuracy:.4f}",
            f"  Std: {baseline_results.std_accuracy:.4f}",
            "",
            "With Transfer Learning:",
            f"  Mean Accuracy: {transfer_results.mean_accuracy:.4f}",
            f"  Std: {transfer_results.std_accuracy:.4f}",
            "",
            "Improvement:",
        ]
        
        improvement = (
            transfer_results.mean_accuracy - baseline_results.mean_accuracy
        )
        relative_improvement = (
            improvement / baseline_results.mean_accuracy * 100 
            if baseline_results.mean_accuracy > 0 else 0
        )
        
        lines.append(f"  Absolute: {improvement:+.4f}")
        lines.append(f"  Relative: {relative_improvement:+.2f}%")
        
        # 逐受试者对比
        lines.extend([
            "",
            "Per-Subject Comparison:",
            "-" * 60,
            f"{'Subject':<10} {'Baseline':<12} {'Transfer':<12} {'Delta':<10}",
            "-" * 60,
        ])
        
        for subj_id in sorted(set(
            list(baseline_results.results.keys()) + 
            list(transfer_results.results.keys())
        )):
            baseline_acc = baseline_results.results.get(subj_id)
            transfer_acc = transfer_results.results.get(subj_id)
            
            if baseline_acc is not None and transfer_acc is not None:
                delta = transfer_acc.accuracy - baseline_acc.accuracy
                lines.append(
                    f"{subj_id:<10} "
                    f"{baseline_acc.accuracy:<12.4f} "
                    f"{transfer_acc.accuracy:<12.4f} "
                    f"{delta:<+10.4f}"
                )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DomainShiftAnalyzer:
    """
    域偏移分析器
    
    专门用于分析EEG跨受试者数据中的域偏移问题
    
    Args:
        reference_subject: 参考受试者（通常选择数据质量最好的）
    """
    
    def __init__(self, reference_subject: Optional[int] = None):
        self.reference_subject = reference_subject
        self.analysis_cache = {}
    
    def compute_distribution_stats(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        计算数据分布统计量
        
        Args:
            X: EEG数据 (n_samples, n_channels, n_times)
            
        Returns:
            统计量字典
        """
        # 计算每个通道的统计量
        n_channels = X.shape[1]
        n_times = X.shape[2]
        
        # 均值
        mean_per_channel = X.mean(axis=(0, 2))  # (n_channels,)
        
        # 标准差
        std_per_channel = X.std(axis=(0, 2))  # (n_channels,)
        
        # 功率谱密度（简化）
        # 使用时间维度的方差作为功率代理
        power_per_channel = (X ** 2).mean(axis=(0, 2))
        
        return {
            'mean': mean_per_channel,
            'std': std_per_channel,
            'power': power_per_channel,
            'overall_mean': X.mean(),
            'overall_std': X.std(),
        }
    
    def compare_distributions(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> Dict[str, float]:
        """
        比较两个数据分布的差异
        
        Args:
            X1, X2: 两个EEG数据集
            
        Returns:
            差异度量字典
        """
        # 计算各自的统计量
        stats1 = self.compute_distribution_stats(X1)
        stats2 = self.compute_distribution_stats(X2)
        
        # 均值差异
        mean_diff = np.abs(stats1['mean'] - stats2['mean']).mean()
        
        # 标准差差异
        std_diff = np.abs(stats1['std'] - stats2['std']).mean()
        
        # 功率差异
        power_diff = np.abs(stats1['power'] - stats2['power']).mean()
        
        # 总体分布差异（简化）
        overall_diff = np.abs(stats1['overall_mean'] - stats2['overall_mean'])
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'power_difference': float(power_diff),
            'overall_difference': float(overall_diff),
            'composite_score': float(mean_diff + std_diff + power_diff),
        }
    
    def compute_covariance_distance(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> float:
        """
        计算协方差矩阵距离
        
        使用多种距离度量
        
        Args:
            X1, X2: 两个数据集
            
        Returns:
            CORAL距离
        """
        # 展平数据
        X1_flat = X1.reshape(X1.shape[0], -1)
        X2_flat = X2.reshape(X2.shape[0], -1)
        
        # 采样
        n = min(100, X1_flat.shape[0], X2_flat.shape[0])
        X1_sample = X1_flat[:n]
        X2_sample = X2_flat[:n]
        
        # 协方差矩阵
        cov1 = np.cov(X1_sample, rowvar=False)
        cov2 = np.cov(X2_sample, rowvar=False)
        
        # 添加正则化确保可逆
        reg = 1e-3 * np.eye(cov1.shape[0])
        cov1 += reg
        cov2 += reg
        
        # CORAL距离
        coral_dist = np.sum((cov1 - cov2) ** 2) / 4
        
        return float(coral_dist)
    
    def analyze_all_subjects(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        分析所有受试者的分布特征
        
        Args:
            subject_data: 受试者数据
            
        Returns:
            每个受试者的分析结果
        """
        results = {}
        
        for subj_id, (X, y) in subject_data.items():
            results[subj_id] = {
                'n_samples': len(X),
                'n_channels': X.shape[1],
                'n_times': X.shape[2],
                'class_distribution': {
                    f'class_{i}': int(np.sum(y == i))
                    for i in np.unique(y)
                },
                'distribution_stats': self.compute_distribution_stats(X),
            }
        
        return results
    
    def generate_shift_report(
        self,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> str:
        """
        生成域偏移报告
        
        Args:
            subject_data: 受试者数据
            
        Returns:
            格式化的报告
        """
        subject_ids = sorted(subject_data.keys())
        n_subjects = len(subject_ids)
        
        lines = [
            "=" * 60,
            "Domain Shift Analysis Report",
            "=" * 60,
            f"Number of subjects: {n_subjects}",
            "",
        ]
        
        # 计算所有对的距离
        distances = []
        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                subj_i, subj_j = subject_ids[i], subject_ids[j]
                X_i, _ = subject_data[subj_i]
                X_j, _ = subject_data[subj_j]
                
                dist = self.compute_covariance_distance(X_i, X_j)
                distances.append((subj_i, subj_j, dist))
        
        if distances:
            dist_values = [d[2] for d in distances]
            
            lines.extend([
                "Pairwise Domain Distances (CORAL):",
                "-" * 40,
                f"  Mean: {np.mean(dist_values):.6f}",
                f"  Std:  {np.std(dist_values):.6f}",
                f"  Min:  {np.min(dist_values):.6f}",
                f"  Max:  {np.max(dist_values):.6f}",
                "",
                "Top 5 most similar pairs:",
            ])
            
            # 按距离排序
            distances_sorted = sorted(distances, key=lambda x: x[2])
            for subj_i, subj_j, dist in distances_sorted[:5]:
                lines.append(f"  ({subj_i}, {subj_j}): {dist:.6f}")
            
            lines.extend([
                "",
                "Top 5 most dissimilar pairs:",
            ])
            
            for subj_i, subj_j, dist in distances_sorted[-5:]:
                lines.append(f"  ({subj_i}, {subj_j}): {dist:.6f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# 导入torch（可能在函数中需要）
import torch

# 导出
__all__ = [
    'SubjectResult',
    'CrossSubjectReport',
    'CrossSubjectEvaluator',
    'DomainShiftAnalyzer',
]
