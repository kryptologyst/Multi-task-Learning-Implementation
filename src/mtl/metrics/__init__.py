"""
Evaluation Metrics for Multi-task Learning

This module contains various metrics for evaluating multi-task learning models
across different task types (classification, regression, etc.).
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class MultiTaskMetrics:
    """Comprehensive metrics for multi-task learning evaluation."""
    
    def __init__(self, task_names: List[str], task_types: Dict[str, str]):
        self.task_names = task_names
        self.task_types = task_types  # 'classification' or 'regression'
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = {task: [] for task in self.task_names}
        self.targets = {task: [] for task in self.task_names}
        
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]):
        """Update metrics with new predictions and targets."""
        for task_name in self.task_names:
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name].detach().cpu()
                target = targets[task_name].detach().cpu()
                
                if self.task_types[task_name] == "classification":
                    pred = torch.argmax(pred, dim=1)
                    
                self.predictions[task_name].extend(pred.numpy())
                self.targets[task_name].extend(target.numpy())
                
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics for all tasks."""
        metrics = {}
        
        for task_name in self.task_names:
            if not self.predictions[task_name]:
                continue
                
            pred = np.array(self.predictions[task_name])
            target = np.array(self.targets[task_name])
            
            if self.task_types[task_name] == "classification":
                metrics[task_name] = self._compute_classification_metrics(pred, target)
            elif self.task_types[task_name] == "regression":
                metrics[task_name] = self._compute_regression_metrics(pred, target)
                
        return metrics
        
    def _compute_classification_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(target, pred)
        metrics["precision_macro"] = precision_score(target, pred, average='macro', zero_division=0)
        metrics["recall_macro"] = recall_score(target, pred, average='macro', zero_division=0)
        metrics["f1_macro"] = f1_score(target, pred, average='macro', zero_division=0)
        metrics["precision_weighted"] = precision_score(target, pred, average='weighted', zero_division=0)
        metrics["recall_weighted"] = recall_score(target, pred, average='weighted', zero_division=0)
        metrics["f1_weighted"] = f1_score(target, pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(target, pred)
        
        return metrics
        
    def _compute_regression_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        # Basic regression metrics
        metrics["mse"] = mean_squared_error(target, pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(target, pred)
        metrics["r2_score"] = r2_score(target, pred)
        
        # Additional metrics
        metrics["mape"] = np.mean(np.abs((target - pred) / (target + 1e-8))) * 100
        metrics["smape"] = np.mean(2 * np.abs(target - pred) / (np.abs(target) + np.abs(pred) + 1e-8)) * 100
        
        return metrics
        
    def plot_confusion_matrix(self, task_name: str, save_path: Optional[str] = None):
        """Plot confusion matrix for classification task."""
        if self.task_types[task_name] != "classification":
            raise ValueError(f"Task {task_name} is not a classification task")
            
        cm = self.predictions[task_name]
        if not cm:
            return
            
        cm = confusion_matrix(self.targets[task_name], self.predictions[task_name])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_regression_scatter(self, task_name: str, save_path: Optional[str] = None):
        """Plot scatter plot for regression task."""
        if self.task_types[task_name] != "regression":
            raise ValueError(f"Task {task_name} is not a regression task")
            
        pred = np.array(self.predictions[task_name])
        target = np.array(self.targets[task_name])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(target, pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Regression Scatter Plot - {task_name}')
        
        # Add R² score
        r2 = r2_score(target, pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class CalibrationMetrics:
    """Metrics for evaluating model calibration."""
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        
    def compute_ece(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Expected Calibration Error (ECE)."""
        # Convert to probabilities
        probs = torch.softmax(predictions, dim=1)
        confidences, predicted_classes = torch.max(probs, dim=1)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = (predicted_classes[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item()
        
    def compute_mce(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        # Convert to probabilities
        probs = torch.softmax(predictions, dim=1)
        confidences, predicted_classes = torch.max(probs, dim=1)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = (predicted_classes[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Update MCE
                mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin).item())
                
        return mce


def create_metrics(task_names: List[str], task_types: Dict[str, str]) -> MultiTaskMetrics:
    """Create metrics object from configuration."""
    return MultiTaskMetrics(task_names, task_types)
