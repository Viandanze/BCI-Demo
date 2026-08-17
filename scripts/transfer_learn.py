#!/usr/bin/env python3
"""
Transfer Learning Script for EEG-based BCI
迁移学习执行脚本

支持多种模式:
- pretrain_finetune: 预训练 + 微调
- dann: 域对抗迁移
- coral: CORAL对齐
- loso: 完整LOSO评估
- compare: 对比所有迁移策略

Usage:
    python scripts/transfer_learn.py --mode loso --data_path ./data/
    python scripts/transfer_learn.py --mode pretrain_finetune --source_subjects 1-50 --target_subjects 51-55
    python scripts/transfer_learn.py --mode compare --output ./results/

Author: BCI_Projects
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.eegnet import EEGNet, EEGNetClassifier
from src.training.transfer import (
    TransferConfig, TransferMode, FreezeStrategy,
    TransferModel, TransferTrainer,
    create_transfer_model,
    pretrain_on_subjects, finetune_on_target,
    compute_domain_distance,
)
from src.evaluation.cross_subject import (
    CrossSubjectEvaluator, CrossSubjectReport,
    DomainShiftAnalyzer, SubjectResult,
)
from src.training.trainer import set_seed, EEGDataset, TrainingConfig
from src.data.loader import (
    load_physionet_data, get_subject_data,
    BINARY_EVENT_ID, create_train_test_split,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='EEG BCI Transfer Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 模式选择
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['pretrain_finetune', 'dann', 'coral', 'loso', 'compare'],
        help='Transfer learning mode',
    )
    
    # 数据相关
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/',
        help='Path to EEG data',
    )
    parser.add_argument(
        '--source_subjects',
        type=str,
        default='1-80',
        help='Source subject range (e.g., 1-80)',
    )
    parser.add_argument(
        '--target_subjects',
        type=str,
        default='81-109',
        help='Target subject range (e.g., 81-109)',
    )
    parser.add_argument(
        '--use_synthetic',
        action='store_true',
        help='Use synthetic data when real data unavailable',
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.001,
        help='Base learning rate',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay',
    )
    parser.add_argument(
        '--dann_lambda',
        type=float,
        default=1.0,
        help='DANN domain loss weight',
    )
    parser.add_argument(
        '--coral_alpha',
        type=float,
        default=1.0,
        help='CORAL alignment weight',
    )
    
    # 冻结策略
    parser.add_argument(
        '--freeze_strategy',
        type=str,
        default='discriminative',
        choices=['none', 'frozen_backbone', 'gradual_unfreeze', 'discriminative'],
        help='Freeze strategy',
    )
    parser.add_argument(
        '--feature_lr_factor',
        type=float,
        default=0.1,
        help='Feature layer learning rate factor',
    )
    parser.add_argument(
        '--classifier_lr_factor',
        type=float,
        default=1.0,
        help='Classifier layer learning rate factor',
    )
    
    # 输出相关
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/',
        help='Output directory',
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained models',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    
    # 设备
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto/cuda/cpu)',
    )
    
    # 随机种子
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    
    return parser.parse_args()


def parse_subject_range(range_str: str) -> List[int]:
    """
    解析受试者范围字符串
    
    Args:
        range_str: 范围字符串，如 '1-80' 或 '1,2,3,5,10'
        
    Returns:
        受试者ID列表
    """
    range_str = range_str.strip()
    
    if '-' in range_str:
        start, end = range_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]


def load_data(
    data_path: str,
    source_range: str,
    target_range: str,
    use_synthetic: bool = False,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    加载数据
    
    Args:
        data_path: 数据路径
        source_range: 源域受试者范围
        target_range: 目标域受试者范围
        use_synthetic: 是否使用合成数据
        
    Returns:
        受试者数据字典
    """
    source_subjects = parse_subject_range(source_range)
    target_subjects = parse_subject_range(target_range)
    all_subjects = source_subjects + target_subjects
    
    logger.info(f"Loading data for {len(all_subjects)} subjects")
    
    try:
        # 尝试加载真实数据
        raw_dict = load_physionet_data(
            data_path=data_path,
            subjects=all_subjects,
            runs=[4, 5, 6],  # Motor imagery runs
        )
        
        # 提取epochs
        subject_data = {}
        for subj_id in all_subjects:
            if subj_id in raw_dict:
                _, X, y = get_subject_data(
                    raw_dict[subj_id],
                    tmin=-1.0,
                    tmax=4.0,
                    event_id=BINARY_EVENT_ID,
                    baseline=(-1.0, 0.0),
                )
                subject_data[subj_id] = (X, y)
                logger.info(f"Subject {subj_id}: {len(X)} epochs")
        
        if len(subject_data) < len(all_subjects) * 0.5 and not use_synthetic:
            logger.warning("Few subjects loaded, consider using --use_synthetic")
        
        return subject_data
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        
        if use_synthetic:
            logger.info("Generating synthetic data...")
            return generate_synthetic_data(all_subjects)
        else:
            raise


def generate_synthetic_data(
    subject_ids: List[int],
    n_channels: int = 48,
    n_times: int = 801,
    n_classes: int = 2,
    samples_per_subject: int = 100,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    生成合成EEG数据用于测试
    
    Args:
        subject_ids: 受试者ID列表
        n_channels: 通道数
        n_times: 时间点数
        n_classes: 类别数
        samples_per_subject: 每个受试者的样本数
        
    Returns:
        合成数据字典
    """
    np.random.seed(42)
    
    subject_data = {}
    
    for subj_id in subject_ids:
        # 使用不同的种子模拟个体差异
        np.random.seed(subj_id * 1000)
        
        X = np.random.randn(samples_per_subject, n_channels, n_times).astype(np.float32)
        y = np.random.randint(0, n_classes, size=samples_per_subject)
        
        # 添加类别相关的模式
        for c in range(n_classes):
            mask = y == c
            # 添加简单的类别区分模式
            pattern = np.sin(np.linspace(0, 4*np.pi, n_times))
            X[mask, :8, :] += pattern * 0.5 * (c + 1)  # 前8通道
        
        # 添加受试者特定的偏移
        subject_offset = np.random.randn(n_channels, n_times) * 0.2
        X += subject_offset
        
        subject_data[subj_id] = (X, y)
        logger.info(f"Subject {subj_id}: {samples_per_subject} synthetic epochs")
    
    return subject_data


def run_pretrain_finetune(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    source_subjects: List[int],
    target_subjects: List[int],
    config: TransferConfig,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    运行预训练-微调流程
    
    Args:
        subject_data: 所有受试者数据
        source_subjects: 源域受试者列表
        target_subjects: 目标域受试者列表
        config: 迁移配置
        epochs: 训练轮数
        batch_size: 批量大小
        
    Returns:
        结果字典
    """
    logger.info("=" * 60)
    logger.info("Running Pretrain-Finetune Transfer Learning")
    logger.info("=" * 60)
    
    # 分离源域和目标域数据
    source_data = {k: subject_data[k] for k in source_subjects if k in subject_data}
    target_data = {k: subject_data[k] for k in target_subjects if k in subject_data}
    
    if len(source_data) == 0 or len(target_data) == 0:
        raise ValueError("Insufficient source or target data")
    
    # 预训练
    logger.info(f"\nPhase 1: Pretraining on {len(source_data)} source subjects")
    pretrained_model = pretrain_on_subjects(
        subject_data=source_data,
        config=config,
        n_epochs=epochs // 2,
    )
    
    # 微调 - 在每个目标受试者上
    logger.info(f"\nPhase 2: Finetuning on {len(target_data)} target subjects")
    
    results = {}
    for target_id in target_subjects:
        if target_id not in target_data:
            continue
        
        logger.info(f"\nFinetuning on target subject {target_id}")
        
        # 获取该目标受试者的数据
        X_target, y_target = target_data[target_id]
        
        # 分割训练和测试
        n_train = int(len(X_target) * 0.7)
        target_train = (X_target[:n_train], y_target[:n_train])
        target_test = (X_target[n_train:], y_target[n_train:])
        
        # 微调
        finetuned_model, finetune_result = finetune_on_target(
            pretrained_model=pretrained_model,
            source_data=(
                np.concatenate([v[0] for v in source_data.values()]),
                np.concatenate([v[1] for v in source_data.values()]),
            ),
            target_data=target_train,
            config=config,
            n_epochs=epochs // 2,
        )
        
        # 评估
        from torch.utils.data import DataLoader
        
        test_dataset = EEGDataset(target_test[0], target_test[1])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        trainer = TransferTrainer(finetuned_model, config)
        eval_result = trainer.evaluate(test_loader)
        
        results[target_id] = {
            'test_accuracy': eval_result['accuracy'],
            'history': finetune_result['history'],
        }
        
        logger.info(f"Subject {target_id}: Test Accuracy = {eval_result['accuracy']:.4f}")
    
    # 汇总
    accuracies = [r['test_accuracy'] for r in results.values()]
    
    return {
        'mode': 'pretrain_finetune',
        'n_source_subjects': len(source_data),
        'n_target_subjects': len(target_data),
        'mean_accuracy': np.mean(accuracies) if accuracies else 0.0,
        'std_accuracy': np.std(accuracies) if accuracies else 0.0,
        'per_subject_results': results,
    }


def run_dann(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    source_subjects: List[int],
    target_subjects: List[int],
    config: TransferConfig,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    运行DANN域对抗迁移
    
    Args:
        subject_data: 所有受试者数据
        source_subjects: 源域受试者列表
        target_subjects: 目标域受试者列表
        config: 迁移配置
        epochs: 训练轮数
        batch_size: 批量大小
        
    Returns:
        结果字典
    """
    logger.info("=" * 60)
    logger.info("Running DANN (Domain Adversarial Neural Network)")
    logger.info("=" * 60)
    
    # 准备数据
    source_data = {k: subject_data[k] for k in source_subjects if k in subject_data}
    target_data = {k: subject_data[k] for k in target_subjects if k in subject_data}
    
    # 合并源域数据
    X_source = np.concatenate([v[0] for v in source_data.values()])
    y_source = np.concatenate([v[1] for v in source_data.values()])
    
    # 获取一个目标受试者用于测试
    test_target_id = target_subjects[0] if target_subjects else None
    X_test, y_test = target_data.get(test_target_id, (None, None))
    
    if X_test is None:
        raise ValueError("No target data available")
    
    # 创建模型
    n_channels = X_source.shape[1]
    n_times = X_source.shape[2]
    n_classes = len(np.unique(y_source))
    
    # 更新配置为DANN模式
    dann_config = TransferConfig(
        mode=TransferMode.DANN,
        freeze_strategy=FreezeStrategy.DISCRIMINATIVE,
        base_lr=config.base_lr,
        dann_lambda=config.dann_lambda,
        device=config.device,
        seed=config.seed,
    )
    
    model = create_transfer_model(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        config=dann_config,
    )
    
    # 创建数据加载器
    source_dataset = EEGDataset(X_source, y_source)
    target_dataset = EEGDataset(X_test, np.zeros(len(X_test)))
    
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    # 训练
    trainer = TransferTrainer(model, dann_config)
    
    # 创建配对加载器
    class PairedDataLoader:
        def __init__(self, source, target):
            self.source = source
            self.target = target
            self.target_iter = iter(target)
        
        def __iter__(self):
            self.target_iter = iter(self.target)
            return self
        
        def __next__(self):
            source_batch = next(self.source)
            try:
                target_batch = next(self.target_iter)
            except StopIteration:
                self.target_iter = iter(self.target)
                target_batch = next(self.target_iter)
            return source_batch, target_batch[0], source_batch[1]
        
        def __len__(self):
            return max(len(self.source), len(self.target))
    
    train_loader = PairedDataLoader(source_loader, target_loader)
    
    logger.info(f"Training DANN on {len(X_source)} source samples")
    history = trainer.train(
        train_loader,
        target_loader=target_loader,
        epochs=epochs,
    )
    
    # 评估
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_result = trainer.evaluate(test_loader)
    
    return {
        'mode': 'dann',
        'test_accuracy': eval_result['accuracy'],
        'history': history,
        'target_subject': test_target_id,
    }


def run_coral(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    source_subjects: List[int],
    target_subjects: List[int],
    config: TransferConfig,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    运行CORAL对齐
    
    Args:
        subject_data: 所有受试者数据
        source_subjects: 源域受试者列表
        target_subjects: 目标域受试者列表
        config: 迁移配置
        epochs: 训练轮数
        batch_size: 批量大小
        
    Returns:
        结果字典
    """
    logger.info("=" * 60)
    logger.info("Running CORAL (Correlation Alignment)")
    logger.info("=" * 60)
    
    # 准备数据
    source_data = {k: subject_data[k] for k in source_subjects if k in subject_data}
    target_data = {k: subject_data[k] for k in target_subjects if k in subject_data}
    
    # 合并源域数据
    X_source = np.concatenate([v[0] for v in source_data.values()])
    y_source = np.concatenate([v[1] for v in source_data.values()])
    
    # 获取目标受试者
    test_target_id = target_subjects[0] if target_subjects else None
    X_test, y_test = target_data.get(test_target_id, (None, None))
    
    if X_test is None:
        raise ValueError("No target data available")
    
    # 创建模型
    n_channels = X_source.shape[1]
    n_times = X_source.shape[2]
    n_classes = len(np.unique(y_source))
    
    # 更新配置为CORAL模式
    coral_config = TransferConfig(
        mode=TransferMode.CORAL,
        freeze_strategy=FreezeStrategy.DISCRIMINATIVE,
        base_lr=config.base_lr,
        coral_alpha=config.coral_alpha,
        device=config.device,
        seed=config.seed,
    )
    
    model = create_transfer_model(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        config=coral_config,
    )
    
    # 创建数据加载器
    source_dataset = EEGDataset(X_source, y_source)
    target_dataset = EEGDataset(X_test, np.zeros(len(X_test)))
    
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    # 训练
    trainer = TransferTrainer(model, coral_config)
    logger.info(f"Training CORAL on {len(X_source)} source samples")
    history = trainer.train(
        source_loader,
        target_loader=target_loader,
        epochs=epochs,
    )
    
    # 评估
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_result = trainer.evaluate(test_loader)
    
    return {
        'mode': 'coral',
        'test_accuracy': eval_result['accuracy'],
        'history': history,
        'target_subject': test_target_id,
    }


def run_loso_evaluation(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    config: TransferConfig,
    epochs: int = 50,
    batch_size: int = 64,
) -> CrossSubjectReport:
    """
    运行LOSO评估
    
    Args:
        subject_data: 所有受试者数据
        config: 迁移配置
        epochs: 训练轮数
        batch_size: 批量大小
        
    Returns:
        CrossSubjectReport
    """
    logger.info("=" * 60)
    logger.info("Running Leave-One-Subject-Out Evaluation")
    logger.info("=" * 60)
    
    # 创建评估器
    evaluator = CrossSubjectEvaluator(
        config=config,
    )
    
    # 运行LOSO
    report = evaluator.loso_evaluate(
        subject_data=subject_data,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    return report


def run_compare_all(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    source_subjects: List[int],
    target_subjects: List[int],
    config: TransferConfig,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    对比所有迁移策略
    
    Args:
        subject_data: 所有受试者数据
        source_subjects: 源域受试者列表
        target_subjects: 目标域受试者列表
        config: 迁移配置
        epochs: 训练轮数
        batch_size: 批量大小
        
    Returns:
        对比结果
    """
    logger.info("=" * 60)
    logger.info("Comparing All Transfer Learning Strategies")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. 基线（无迁移）
    logger.info("\n--- Baseline (No Transfer) ---")
    try:
        # 使用少量源域数据训练，直接在目标上测试
        X_source = np.concatenate([subject_data[k][0] for k in source_subjects if k in subject_data])
        y_source = np.concatenate([subject_data[k][1] for k in source_subjects if k in subject_data])
        
        target_id = target_subjects[0]
        X_target, y_target = subject_data[target_id]
        
        n_train = int(len(X_target) * 0.5)
        X_train = np.concatenate([X_source[:50], X_target[:n_train]])
        y_train = np.concatenate([y_source[:50], y_target[:n_train]])
        
        baseline_config = TransferConfig(
            mode=TransferMode.FINE_TUNE,
            freeze_strategy=FreezeStrategy.NONE,
            device=config.device,
            seed=config.seed,
        )
        
        model = create_transfer_model(
            n_channels=X_train.shape[1],
            n_times=X_train.shape[2],
            n_classes=len(np.unique(y_train)),
            config=baseline_config,
        )
        
        trainer = TransferTrainer(model, baseline_config)
        train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size)
        test_loader = DataLoader(EEGDataset(X_target[n_train:], y_target[n_train:]), batch_size=batch_size)
        
        trainer.train(train_loader, epochs=epochs)
        baseline_result = trainer.evaluate(test_loader)
        
        results['baseline'] = {
            'accuracy': baseline_result['accuracy'],
            'target_subject': target_id,
        }
        logger.info(f"Baseline accuracy: {baseline_result['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Baseline failed: {e}")
        results['baseline'] = {'accuracy': 0.0}
    
    # 2. DANN
    logger.info("\n--- DANN ---")
    try:
        dann_result = run_dann(
            subject_data, source_subjects, target_subjects,
            config, epochs, batch_size
        )
        results['dann'] = dann_result
    except Exception as e:
        logger.error(f"DANN failed: {e}")
        results['dann'] = {'test_accuracy': 0.0}
    
    # 3. CORAL
    logger.info("\n--- CORAL ---")
    try:
        coral_result = run_coral(
            subject_data, source_subjects, target_subjects,
            config, epochs, batch_size
        )
        results['coral'] = coral_result
    except Exception as e:
        logger.error(f"CORAL failed: {e}")
        results['coral'] = {'test_accuracy': 0.0}
    
    # 4. Pretrain-Finetune
    logger.info("\n--- Pretrain-Finetune ---")
    try:
        ptft_result = run_pretrain_finetune(
            subject_data, source_subjects, target_subjects,
            config, epochs, batch_size
        )
        results['pretrain_finetune'] = ptft_result
    except Exception as e:
        logger.error(f"Pretrain-Finetune failed: {e}")
        results['pretrain_finetune'] = {'mean_accuracy': 0.0}
    
    # 5. LOSO
    logger.info("\n--- LOSO Evaluation ---")
    try:
        loso_report = run_loso_evaluation(
            subject_data, config, epochs, batch_size
        )
        results['loso'] = {
            'mean_accuracy': loso_report.mean_accuracy,
            'std_accuracy': loso_report.std_accuracy,
            'n_subjects': len(loso_report.results),
        }
    except Exception as e:
        logger.error(f"LOSO failed: {e}")
        results['loso'] = {'mean_accuracy': 0.0}
    
    # 生成对比表格
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    summary = [
        f"{'Method':<25} {'Accuracy':<15}",
        "-" * 40,
    ]
    
    for method, result in results.items():
        if 'accuracy' in result:
            acc = result['accuracy']
        elif 'test_accuracy' in result:
            acc = result['test_accuracy']
        elif 'mean_accuracy' in result:
            acc = result['mean_accuracy']
        else:
            acc = 0.0
        
        summary.append(f"{method:<25} {acc:.4f}")
    
    for line in summary:
        logger.info(line)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, mode: str) -> None:
    """
    保存结果
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        mode: 运行模式
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON
    result_file = output_path / f"transfer_results_{mode}.json"
    
    # 转换numpy类型为Python类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析受试者范围
    source_subjects = parse_subject_range(args.source_subjects)
    target_subjects = parse_subject_range(args.target_subjects)
    
    logger.info(f"Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Source subjects: {len(source_subjects)} ({source_subjects[0]}-{source_subjects[-1]})")
    logger.info(f"  Target subjects: {len(target_subjects)} ({target_subjects[0] if target_subjects else 'N/A'}-{target_subjects[-1] if target_subjects else 'N/A'})")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Base LR: {args.base_lr}")
    logger.info(f"  Device: {args.device}")
    
    # 创建配置
    config = TransferConfig(
        mode=TransferMode(args.mode) if args.mode in ['dann', 'coral'] else TransferMode.FINE_TUNE,
        freeze_strategy=FreezeStrategy(args.freeze_strategy),
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        dann_lambda=args.dann_lambda,
        coral_alpha=args.coral_alpha,
        feature_lr_factor=args.feature_lr_factor,
        classifier_lr_factor=args.classifier_lr_factor,
        device=args.device,
        seed=args.seed,
    )
    
    # 加载数据
    try:
        subject_data = load_data(
            data_path=args.data_path,
            source_range=args.source_subjects,
            target_range=args.target_subjects,
            use_synthetic=args.use_synthetic,
        )
        
        if len(subject_data) == 0:
            logger.error("No data loaded. Use --use_synthetic to generate synthetic data.")
            return
        
        logger.info(f"\nLoaded data for {len(subject_data)} subjects")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        
        if args.use_synthetic:
            logger.info("Generating synthetic data...")
            all_subjects = source_subjects + target_subjects
            subject_data = generate_synthetic_data(all_subjects)
        else:
            return
    
    # 运行对应模式
    results = None
    
    try:
        if args.mode == 'pretrain_finetune':
            results = run_pretrain_finetune(
                subject_data=subject_data,
                source_subjects=source_subjects,
                target_subjects=target_subjects,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            
        elif args.mode == 'dann':
            results = run_dann(
                subject_data=subject_data,
                source_subjects=source_subjects,
                target_subjects=target_subjects,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            
        elif args.mode == 'coral':
            results = run_coral(
                subject_data=subject_data,
                source_subjects=source_subjects,
                target_subjects=target_subjects,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            
        elif args.mode == 'loso':
            report = run_loso_evaluation(
                subject_data=subject_data,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            
            print("\n" + report.summary())
            results = report.to_dict()
            
            # 额外保存域偏移分析
            if report.domain_shift_analysis:
                logger.info("\nDomain Shift Analysis:")
                for k, v in report.domain_shift_analysis.items():
                    logger.info(f"  {k}: {v}")
            
        elif args.mode == 'compare':
            results = run_compare_all(
                subject_data=subject_data,
                source_subjects=source_subjects,
                target_subjects=target_subjects,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        
        # 保存结果
        if results is not None:
            save_results(results, args.output, args.mode)
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
