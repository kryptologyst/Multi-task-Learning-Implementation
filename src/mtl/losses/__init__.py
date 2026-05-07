"""
Loss Functions for Multi-task Learning

This module contains various loss functions and weighting strategies
for multi-task learning experiments.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiTaskLoss(nn.Module):
    """Base class for multi-task loss functions."""
    
    def __init__(self, task_names: list, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_names = task_names
        self.task_weights = task_weights or {name: 1.0 for name in task_names}
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        raise NotImplementedError


class WeightedSumLoss(MultiTaskLoss):
    """Simple weighted sum of individual task losses."""
    
    def __init__(self, task_names: list, task_weights: Optional[Dict[str, float]] = None,
                 loss_functions: Optional[Dict[str, nn.Module]] = None):
        super().__init__(task_names, task_weights)
        
        # Default loss functions
        self.loss_functions = loss_functions or {}
        for task_name in task_names:
            if task_name not in self.loss_functions:
                if "classification" in task_name:
                    self.loss_functions[task_name] = nn.CrossEntropyLoss()
                elif "regression" in task_name:
                    self.loss_functions[task_name] = nn.MSELoss()
                else:
                    self.loss_functions[task_name] = nn.CrossEntropyLoss()
                    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum loss."""
        total_loss = 0.0
        task_losses = {}
        
        for task_name in self.task_names:
            if task_name in predictions and task_name in targets:
                loss = self.loss_functions[task_name](
                    predictions[task_name], targets[task_name]
                )
                weighted_loss = self.task_weights[task_name] * loss
                total_loss += weighted_loss
                task_losses[task_name] = loss.item()
                
        return total_loss


class UncertaintyWeightedLoss(MultiTaskLoss):
    """
    Uncertainty-weighted loss for multi-task learning.
    
    Implements the method from "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018).
    """
    
    def __init__(self, task_names: list, learnable_weights: bool = True):
        super().__init__(task_names)
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable log variance parameters
            self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
        else:
            # Fixed weights
            self.log_vars = torch.zeros(len(task_names))
            
        # Loss functions for each task
        self.loss_functions = {}
        for task_name in task_names:
            if "classification" in task_name:
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif "regression" in task_name:
                self.loss_functions[task_name] = nn.MSELoss()
            else:
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
                
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty-weighted loss."""
        total_loss = 0.0
        
        for i, task_name in enumerate(self.task_names):
            if task_name in predictions and task_name in targets:
                # Compute task loss
                task_loss = self.loss_functions[task_name](
                    predictions[task_name], targets[task_name]
                )
                
                # Uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
                total_loss += weighted_loss
                
        return total_loss


class DynamicWeightedLoss(MultiTaskLoss):
    """
    Dynamic weighted loss based on task difficulty.
    
    Automatically adjusts task weights based on loss magnitudes.
    """
    
    def __init__(self, task_names: list, alpha: float = 0.5):
        super().__init__(task_names)
        self.alpha = alpha
        self.register_buffer('loss_history', torch.zeros(len(task_names)))
        
        # Loss functions for each task
        self.loss_functions = {}
        for task_name in task_names:
            if "classification" in task_name:
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif "regression" in task_name:
                self.loss_functions[task_name] = nn.MSELoss()
            else:
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
                
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute dynamically weighted loss."""
        total_loss = 0.0
        current_losses = []
        
        # Compute individual task losses
        for task_name in self.task_names:
            if task_name in predictions and task_name in targets:
                task_loss = self.loss_functions[task_name](
                    predictions[task_name], targets[task_name]
                )
                current_losses.append(task_loss)
            else:
                current_losses.append(torch.tensor(0.0, device=next(iter(predictions.values())).device))
                
        current_losses = torch.stack(current_losses)
        
        # Update loss history
        self.loss_history = self.alpha * self.loss_history + (1 - self.alpha) * current_losses.detach()
        
        # Compute dynamic weights
        weights = self.loss_history / (self.loss_history.sum() + 1e-8)
        weights = weights / weights.mean()  # Normalize weights
        
        # Compute weighted loss
        for i, task_name in enumerate(self.task_names):
            if task_name in predictions and task_name in targets:
                total_loss += weights[i] * current_losses[i]
                
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Implements the focal loss from "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross-entropy loss."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / inputs.size(1)
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        return loss.mean()


def create_loss_function(loss_config: Dict) -> MultiTaskLoss:
    """Create loss function from configuration."""
    loss_type = loss_config.get("type", "weighted_sum")
    task_names = loss_config.get("task_names", [])
    
    if loss_type == "weighted_sum":
        return WeightedSumLoss(
            task_names=task_names,
            task_weights=loss_config.get("task_weights"),
            loss_functions=loss_config.get("loss_functions")
        )
    elif loss_type == "uncertainty_weighted":
        return UncertaintyWeightedLoss(
            task_names=task_names,
            learnable_weights=loss_config.get("learnable_weights", True)
        )
    elif loss_type == "dynamic_weighted":
        return DynamicWeightedLoss(
            task_names=task_names,
            alpha=loss_config.get("alpha", 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
