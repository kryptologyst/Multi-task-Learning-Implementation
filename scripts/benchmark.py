#!/usr/bin/env python3
"""
Benchmark Script for Multi-task Learning

This script runs a quick benchmark to test the framework performance.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import time
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mtl.models import ResNetMTL, BaselineModel, UncertaintyWeightedMTL
from mtl.data import CIFAR10AttributesDataset
from mtl.losses import WeightedSumLoss, UncertaintyWeightedLoss
from mtl.metrics import MultiTaskMetrics
from mtl.utils import get_device, set_seed, count_parameters


def benchmark_model(model, data_loader, device, num_batches=10):
    """Benchmark model inference speed."""
    model.eval()
    model.to(device)
    
    times = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            
            # Time inference
            start_time = time.time()
            predictions = model(images)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput': images.size(0) / np.mean(times)
    }


def main():
    """Run benchmark tests."""
    print("🚀 Multi-task Learning Benchmark")
    print("=" * 50)
    
    # Set up
    device = get_device("auto")
    set_seed(42)
    
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Create dummy dataset for benchmarking
    print("📊 Creating benchmark dataset...")
    
    # Mock dataset for benchmarking
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            image = torch.randn(3, 32, 32)
            targets = {
                "classification": torch.randint(0, 10, (1,)).item(),
                "age_regression": torch.randn(1).item(),
                "gender_classification": torch.randint(0, 2, (1,)).item(),
            }
            return image, targets
    
    from torch.utils.data import DataLoader
    dataset = MockDataset(1000)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Test models
    models_to_test = [
        ("Baseline CNN", BaselineModel()),
        ("ResNet MTL", ResNetMTL(
            backbone="resnet50",
            pretrained=False,  # Use False for faster benchmarking
            classification={"num_classes": 10},
            age_regression={},
            gender_classification={"num_classes": 2}
        )),
    ]
    
    results = {}
    
    for model_name, model in models_to_test:
        print(f"🧪 Testing {model_name}...")
        
        # Count parameters
        param_info = count_parameters(model)
        
        # Benchmark inference
        benchmark_results = benchmark_model(model, data_loader, device)
        
        results[model_name] = {
            'parameters': param_info,
            'benchmark': benchmark_results
        }
        
        print(f"  Parameters: {param_info['total']:,}")
        print(f"  Trainable: {param_info['trainable']:,}")
        print(f"  Avg inference time: {benchmark_results['avg_time']:.4f}s")
        print(f"  Throughput: {benchmark_results['throughput']:.2f} samples/s")
        print()
    
    # Print summary
    print("📈 Benchmark Summary")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Parameters: {result['parameters']['total']:,}")
        print(f"  Inference: {result['benchmark']['avg_time']:.4f}s ± {result['benchmark']['std_time']:.4f}s")
        print(f"  Throughput: {result['benchmark']['throughput']:.2f} samples/s")
        print()
    
    # Test loss functions
    print("🔧 Testing Loss Functions...")
    
    # Create dummy predictions and targets
    predictions = {
        "classification": torch.randn(32, 10),
        "age_regression": torch.randn(32, 1),
        "gender_classification": torch.randn(32, 2)
    }
    
    targets = {
        "classification": torch.randint(0, 10, (32,)),
        "age_regression": torch.randn(32, 1),
        "gender_classification": torch.randint(0, 2, (32,))
    }
    
    # Test weighted sum loss
    loss_fn = WeightedSumLoss(
        task_names=["classification", "age_regression", "gender_classification"]
    )
    
    start_time = time.time()
    loss = loss_fn(predictions, targets)
    loss_time = time.time() - start_time
    
    print(f"  WeightedSumLoss: {loss.item():.4f} (computed in {loss_time:.6f}s)")
    
    # Test uncertainty weighted loss
    loss_fn_unc = UncertaintyWeightedLoss(
        task_names=["classification", "age_regression", "gender_classification"]
    )
    
    start_time = time.time()
    loss_unc = loss_fn_unc(predictions, targets)
    loss_time_unc = time.time() - start_time
    
    print(f"  UncertaintyWeightedLoss: {loss_unc.item():.4f} (computed in {loss_time_unc:.6f}s)")
    
    print()
    
    # Test metrics
    print("📊 Testing Metrics...")
    
    metrics = MultiTaskMetrics(
        task_names=["classification", "age_regression", "gender_classification"],
        task_types={
            "classification": "classification",
            "age_regression": "regression",
            "gender_classification": "classification"
        }
    )
    
    # Update metrics
    start_time = time.time()
    metrics.update(predictions, targets)
    computed_metrics = metrics.compute_metrics()
    metrics_time = time.time() - start_time
    
    print(f"  Metrics computed in {metrics_time:.6f}s")
    print(f"  Tasks evaluated: {len(computed_metrics)}")
    
    print()
    print("✅ Benchmark completed successfully!")
    print()
    print("Author: kryptologyst")
    print("GitHub: https://github.com/kryptologyst")


if __name__ == "__main__":
    main()
