"""
Evaluation metrics for BCI classification
Provides comprehensive metrics including accuracy, kappa, confusion matrix, etc.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        accuracy: Classification accuracy (0-1)
        kappa: Cohen's kappa coefficient
        f1_macro: Macro-averaged F1 score
        f1_weighted: Weighted F1 score
        precision_macro: Macro-averaged precision
        recall_macro: Macro-averaged recall
        confusion_matrix: Confusion matrix
        per_class_accuracy: Per-class accuracy
        per_class_f1: Per-class F1 score
    """
    accuracy: float
    kappa: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    confusion_matrix: np.ndarray
    per_class_accuracy: Optional[np.ndarray] = None
    per_class_f1: Optional[np.ndarray] = None
    auc: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'accuracy': self.accuracy,
            'kappa': self.kappa,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
        }
        
        if self.per_class_accuracy is not None:
            result['per_class_accuracy'] = self.per_class_accuracy.tolist()
        
        if self.per_class_f1 is not None:
            result['per_class_f1'] = self.per_class_f1.tolist()
        
        if self.auc is not None:
            result['auc'] = self.auc
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"EvaluationMetrics(\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Kappa: {self.kappa:.4f}\n"
            f"  F1 (macro): {self.f1_macro:.4f}\n"
            f"  F1 (weighted): {self.f1_weighted:.4f}\n"
            f"  Precision (macro): {self.precision_macro:.4f}\n"
            f"  Recall (macro): {self.recall_macro:.4f}\n"
            f")"
        )


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy (0-1)
    """
    return accuracy_score(y_true, y_pred)


def compute_kappa(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    weights: Optional[str] = None
) -> float:
    """
    Compute Cohen's kappa coefficient.
    
    Kappa measures inter-rater agreement, accounting for chance.
    Values > 0.8 indicate almost perfect agreement.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        weights: None, 'linear', or 'quadratic' for weighted kappa
        
    Returns:
        Kappa coefficient (-1 to 1)
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label indices
        normalize: If True, normalize rows to sum to 1
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    return cm


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> str:
    """
    Generate classification report with per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names
        output_dict: If True, return dict instead of string
        
    Returns:
        Classification report (string or dict)
    """
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        class_names: Optional class names
        
    Returns:
        EvaluationMetrics object with all computed metrics
    """
    n_classes = len(np.unique(y_true))
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Precision and recall
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        if cm[i].sum() > 0:
            per_class_acc[i] = cm[i, i] / cm[i].sum()
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # AUC (if probabilities provided)
    auc = None
    if y_proba is not None and n_classes == 2:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            auc = None
    
    return EvaluationMetrics(
        accuracy=accuracy,
        kappa=kappa,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        confusion_matrix=cm,
        per_class_accuracy=per_class_acc,
        per_class_f1=per_class_f1,
        auc=auc,
    )


def format_results_table(
    results: Dict[str, EvaluationMetrics],
    metrics: List[str] = ['accuracy', 'kappa', 'f1_macro'],
) -> str:
    """
    Format multiple model results as a table.
    
    Args:
        results: Dictionary mapping model names to EvaluationMetrics
        metrics: List of metrics to include
        
    Returns:
        Formatted table string
    """
    lines = []
    
    # Header
    header = f"{'Model':<20}"
    for metric in metrics:
        header += f" | {metric:<12}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for model_name, metrics_obj in results.items():
        row = f"{model_name:<20}"
        for metric in metrics:
            value = getattr(metrics_obj, metric, None)
            if value is not None:
                row += f" | {value:<12.4f}"
            else:
                row += f" | {'N/A':<12}"
        lines.append(row)
    
    return "\n".join(lines)


def print_results(
    metrics: EvaluationMetrics,
    class_names: Optional[List[str]] = None,
    title: str = "Evaluation Results",
) -> None:
    """
    Print formatted evaluation results.
    
    Args:
        metrics: EvaluationMetrics object
        class_names: Optional class names
        title: Title for output
    """
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print(f"  Kappa:           {metrics.kappa:.4f}")
    print(f"  F1 (macro):      {metrics.f1_macro:.4f}")
    print(f"  F1 (weighted):   {metrics.f1_weighted:.4f}")
    print(f"  Precision:       {metrics.precision_macro:.4f}")
    print(f"  Recall:          {metrics.recall_macro:.4f}")
    
    if metrics.auc is not None:
        print(f"  AUC-ROC:         {metrics.auc:.4f}")
    
    if class_names and metrics.per_class_f1 is not None:
        print(f"\nPer-Class F1 Scores:")
        for i, (name, f1) in enumerate(zip(class_names, metrics.per_class_f1)):
            print(f"  {name:<15}: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  {metrics.confusion_matrix}")
    
    if metrics.per_class_accuracy is not None:
        print(f"\nPer-Class Accuracy:")
        for i, acc in enumerate(metrics.per_class_accuracy):
            name = class_names[i] if class_names else f"Class {i}"
            print(f"  {name:<15}: {acc:.4f}")


def save_results(
    metrics: EvaluationMetrics,
    save_path: str,
    model_name: str = "model",
    include_confusion_matrix: bool = True,
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        metrics: EvaluationMetrics object
        save_path: Path to save results
        model_name: Name of the model
        include_confusion_matrix: Include confusion matrix in save
    """
    import json
    from pathlib import Path
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_name': model_name,
        'metrics': metrics.to_dict(),
    }
    
    if include_confusion_matrix:
        results['confusion_matrix'] = metrics.confusion_matrix.tolist()
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {save_path}")


# Export
__all__ = [
    "EvaluationMetrics",
    "compute_accuracy",
    "compute_kappa",
    "compute_confusion_matrix",
    "compute_classification_report",
    "compute_all_metrics",
    "format_results_table",
    "print_results",
    "save_results",
]
