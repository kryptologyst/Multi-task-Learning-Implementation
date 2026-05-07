"""
Multi-task Learning Models

This module contains various multi-task learning model architectures
including classical baselines and advanced MTL methods.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class BaseMTLModel(nn.Module):
    """Base class for multi-task learning models."""
    
    def __init__(self, num_tasks: int, task_names: List[str]):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning predictions for all tasks."""
        raise NotImplementedError
        
    def get_task_losses(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses for each task."""
        raise NotImplementedError


class ResNetMTL(BaseMTLModel):
    """
    ResNet-based Multi-task Learning Model
    
    Uses a pre-trained ResNet backbone with task-specific heads for
    classification and regression tasks.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        classification: Optional[Dict] = None,
        age_regression: Optional[Dict] = None,
        gender_classification: Optional[Dict] = None,
        feature_dim: int = 2048,
        shared_layers: List[int] = [512, 256],
    ):
        # Define task names
        task_names = []
        if classification:
            task_names.append("classification")
        if age_regression:
            task_names.append("age_regression")
        if gender_classification:
            task_names.append("gender_classification")
            
        super().__init__(num_tasks=len(task_names), task_names=task_names)
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feature_dim = 2048
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_dim = feature_dim
        for hidden_dim in shared_layers:
            self.shared_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim
            
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        if classification:
            self.task_heads["classification"] = nn.Linear(
                prev_dim, classification["num_classes"]
            )
            
        if age_regression:
            self.task_heads["age_regression"] = nn.Linear(
                prev_dim, 1
            )
            
        if gender_classification:
            self.task_heads["gender_classification"] = nn.Linear(
                prev_dim, gender_classification["num_classes"]
            )
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass through shared layers
        for layer in self.shared_layers:
            features = layer(features)
            
        # Task-specific predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(features)
            
        return predictions


class UncertaintyWeightedMTL(BaseMTLModel):
    """
    Multi-task Learning with Uncertainty Weighting
    
    Implements the method from "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018).
    """
    
    def __init__(self, base_model: BaseMTLModel):
        super().__init__(
            num_tasks=base_model.num_tasks,
            task_names=base_model.task_names
        )
        self.base_model = base_model
        
        # Learnable uncertainty parameters (log variance)
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names)))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty weighting."""
        return self.base_model(x)
        
    def compute_weighted_loss(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute uncertainty-weighted loss."""
        total_loss = 0.0
        
        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * losses[task_name] + self.log_vars[i]
                total_loss += weighted_loss
                
        return total_loss


class GradientSurgeryMTL(BaseMTLModel):
    """
    Multi-task Learning with Gradient Surgery
    
    Implements "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
    """
    
    def __init__(self, base_model: BaseMTLModel):
        super().__init__(
            num_tasks=base_model.num_tasks,
            task_names=base_model.task_names
        )
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.base_model(x)
        
    def apply_gradient_surgery(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply gradient surgery to reduce conflicts."""
        if len(gradients) < 2:
            return gradients
            
        # Compute pairwise conflicts
        conflicts = []
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                conflict = torch.dot(gradients[i].flatten(), gradients[j].flatten())
                conflicts.append(conflict)
                
        # Apply surgery if conflicts are negative
        modified_gradients = []
        for i, grad in enumerate(gradients):
            modified_grad = grad.clone()
            for j, other_grad in enumerate(gradients):
                if i != j:
                    conflict = torch.dot(grad.flatten(), other_grad.flatten())
                    if conflict < 0:
                        # Project out conflicting component
                        projection = conflict / torch.norm(other_grad.flatten()) ** 2
                        modified_grad = modified_grad - projection * other_grad
            modified_gradients.append(modified_grad)
            
        return modified_gradients


class PCGradMTL(BaseMTLModel):
    """
    Multi-task Learning with PCGrad
    
    Implements "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
    """
    
    def __init__(self, base_model: BaseMTLModel):
        super().__init__(
            num_tasks=base_model.num_tasks,
            task_names=base_model.task_names
        )
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.base_model(x)
        
    def apply_pcgrad(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply PCGrad to reduce conflicts."""
        if len(gradients) < 2:
            return gradients
            
        modified_gradients = []
        for i, grad in enumerate(gradients):
            modified_grad = grad.clone()
            for j, other_grad in enumerate(gradients):
                if i != j:
                    # Compute projection
                    dot_product = torch.dot(grad.flatten(), other_grad.flatten())
                    if dot_product < 0:
                        # Project out conflicting component
                        projection = dot_product / torch.norm(other_grad.flatten()) ** 2
                        modified_grad = modified_grad - projection * other_grad
            modified_gradients.append(modified_grad)
            
        return modified_gradients


# Factory function for creating models
def create_model(model_config: Dict) -> BaseMTLModel:
    """Create a model from configuration."""
    model_type = model_config.get("type", "resnet_mtl")
    
    if model_type == "resnet_mtl":
        return ResNetMTL(**model_config)
    elif model_type == "uncertainty_weighted":
        base_model = create_model(model_config["base_model"])
        return UncertaintyWeightedMTL(base_model)
    elif model_type == "gradient_surgery":
        base_model = create_model(model_config["base_model"])
        return GradientSurgeryMTL(base_model)
    elif model_type == "pcgrad":
        base_model = create_model(model_config["base_model"])
        return PCGradMTL(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
