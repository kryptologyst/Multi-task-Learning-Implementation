"""
Tests for Multi-task Learning Implementation

This module contains unit tests for the multi-task learning framework.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mtl.models import ResNetMTL, BaselineModel, UncertaintyWeightedMTL
from mtl.data import CIFAR10AttributesDataset
from mtl.losses import WeightedSumLoss, UncertaintyWeightedLoss
from mtl.metrics import MultiTaskMetrics
from mtl.utils import get_device, set_seed, count_parameters


class TestModels:
    """Test model architectures."""
    
    def test_resnet_mtl_creation(self):
        """Test ResNet MTL model creation."""
        model = ResNetMTL(
            backbone="resnet50",
            pretrained=False,  # Use False for testing
            classification={"num_classes": 10},
            age_regression={},
            gender_classification={"num_classes": 2}
        )
        
        assert model.num_tasks == 3
        assert "classification" in model.task_names
        assert "age_regression" in model.task_names
        assert "gender_classification" in model.task_names
        
    def test_resnet_mtl_forward(self):
        """Test ResNet MTL forward pass."""
        model = ResNetMTL(
            backbone="resnet50",
            pretrained=False,
            classification={"num_classes": 10},
            age_regression={},
            gender_classification={"num_classes": 2}
        )
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        predictions = model(x)
        
        assert "classification" in predictions
        assert "age_regression" in predictions
        assert "gender_classification" in predictions
        
        assert predictions["classification"].shape == (2, 10)
        assert predictions["age_regression"].shape == (2, 1)
        assert predictions["gender_classification"].shape == (2, 2)
        
    def test_baseline_model(self):
        """Test baseline model."""
        model = BaselineModel()
        
        x = torch.randn(2, 3, 32, 32)
        predictions = model(x)
        
        assert "classification" in predictions
        assert "age_regression" in predictions
        assert "gender_classification" in predictions
        
    def test_uncertainty_weighted_mtl(self):
        """Test uncertainty weighted MTL model."""
        base_model = ResNetMTL(
            backbone="resnet50",
            pretrained=False,
            classification={"num_classes": 10},
            age_regression={},
            gender_classification={"num_classes": 2}
        )
        
        model = UncertaintyWeightedMTL(base_model)
        
        x = torch.randn(2, 3, 224, 224)
        predictions = model(x)
        
        assert "classification" in predictions
        assert "age_regression" in predictions
        assert "gender_classification" in predictions


class TestData:
    """Test data loading functionality."""
    
    def test_cifar10_attributes_dataset(self):
        """Test CIFAR-10 attributes dataset."""
        # Mock the CIFAR-10 dataset to avoid downloading
        with patch('mtl.data.datasets.CIFAR10') as mock_dataset:
            # Create mock dataset
            mock_dataset.return_value = Mock()
            mock_dataset.return_value.__len__ = Mock(return_value=100)
            mock_dataset.return_value.__getitem__ = Mock(return_value=(torch.randn(3, 32, 32), 0))
            mock_dataset.return_value.targets = [0] * 100
            
            dataset = CIFAR10AttributesDataset(
                root="test_data",
                train=True,
                download=False,
                seed=42
            )
            
            assert len(dataset) == 100
            assert len(dataset.age_labels) == 100
            assert len(dataset.gender_labels) == 100
            
            # Test getting an item
            image, targets = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert isinstance(targets, dict)
            assert "classification" in targets
            assert "age_regression" in targets
            assert "gender_classification" in targets


class TestLosses:
    """Test loss functions."""
    
    def test_weighted_sum_loss(self):
        """Test weighted sum loss."""
        loss_fn = WeightedSumLoss(
            task_names=["classification", "age_regression", "gender_classification"],
            task_weights={"classification": 1.0, "age_regression": 0.5, "gender_classification": 1.0}
        )
        
        predictions = {
            "classification": torch.randn(2, 10),
            "age_regression": torch.randn(2, 1),
            "gender_classification": torch.randn(2, 2)
        }
        
        targets = {
            "classification": torch.randint(0, 10, (2,)),
            "age_regression": torch.randn(2, 1),
            "gender_classification": torch.randint(0, 2, (2,))
        }
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
    def test_uncertainty_weighted_loss(self):
        """Test uncertainty weighted loss."""
        loss_fn = UncertaintyWeightedLoss(
            task_names=["classification", "age_regression", "gender_classification"]
        )
        
        predictions = {
            "classification": torch.randn(2, 10),
            "age_regression": torch.randn(2, 1),
            "gender_classification": torch.randn(2, 2)
        }
        
        targets = {
            "classification": torch.randint(0, 10, (2,)),
            "age_regression": torch.randn(2, 1),
            "gender_classification": torch.randint(0, 2, (2,))
        }
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestMetrics:
    """Test metrics computation."""
    
    def test_multi_task_metrics(self):
        """Test multi-task metrics."""
        metrics = MultiTaskMetrics(
            task_names=["classification", "age_regression", "gender_classification"],
            task_types={
                "classification": "classification",
                "age_regression": "regression",
                "gender_classification": "classification"
            }
        )
        
        # Update with dummy data
        predictions = {
            "classification": torch.randint(0, 10, (10,)),
            "age_regression": torch.randn(10, 1),
            "gender_classification": torch.randint(0, 2, (10,))
        }
        
        targets = {
            "classification": torch.randint(0, 10, (10,)),
            "age_regression": torch.randn(10, 1),
            "gender_classification": torch.randint(0, 2, (10,))
        }
        
        metrics.update(predictions, targets)
        results = metrics.compute_metrics()
        
        assert "classification" in results
        assert "age_regression" in results
        assert "gender_classification" in results


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can check it doesn't raise an error
        assert True
        
    def test_count_parameters(self):
        """Test parameter counting."""
        model = BaselineModel()
        param_info = count_parameters(model)
        
        assert "total" in param_info
        assert "trainable" in param_info
        assert "non_trainable" in param_info
        assert param_info["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
