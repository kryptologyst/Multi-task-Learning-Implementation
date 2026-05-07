#!/usr/bin/env python3
"""
Main Training Script for Multi-task Learning

This script provides a comprehensive training pipeline for multi-task learning
experiments with various baselines and advanced methods.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mtl.models import ResNetMTL, UncertaintyWeightedMTL, GradientSurgeryMTL, PCGradMTL
from mtl.data import create_data_loader
from mtl.losses import WeightedSumLoss, UncertaintyWeightedLoss, DynamicWeightedLoss
from mtl.train import Trainer
from mtl.eval import Evaluator
from mtl.viz import MTLVisualizer
from mtl.utils import (
    get_device, set_seed, save_config, create_experiment_dir,
    print_model_summary, create_optimizer, create_scheduler
)


class BaselineModel(nn.Module):
    """Simple baseline model for comparison."""
    
    def __init__(self, num_classes: int = 10, num_age_classes: int = 1, num_gender_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)
        self.age_predictor = nn.Linear(128 * 4 * 4, num_age_classes)
        self.gender_predictor = nn.Linear(128 * 4 * 4, num_gender_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        return {
            "classification": self.classifier(features),
            "age_regression": self.age_predictor(features),
            "gender_classification": self.gender_predictor(features)
        }


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_type = config.get("type", "resnet_mtl")
    
    if model_type == "baseline":
        model = BaselineModel()
    elif model_type == "resnet_mtl":
        model = ResNetMTL(
            backbone=config.get("backbone", "resnet50"),
            pretrained=config.get("pretrained", True),
            freeze_backbone=config.get("freeze_backbone", False),
            dropout_rate=config.get("dropout_rate", 0.5),
            classification=config.get("classification"),
            age_regression=config.get("age_regression"),
            gender_classification=config.get("gender_classification"),
            feature_dim=config.get("feature_dim", 2048),
            shared_layers=config.get("shared_layers", [512, 256])
        )
    elif model_type == "uncertainty_weighted":
        base_model = ResNetMTL(**config.get("base_model", {}))
        model = UncertaintyWeightedMTL(base_model)
    elif model_type == "gradient_surgery":
        base_model = ResNetMTL(**config.get("base_model", {}))
        model = GradientSurgeryMTL(base_model)
    elif model_type == "pcgrad":
        base_model = ResNetMTL(**config.get("base_model", {}))
        model = PCGradMTL(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Create loss function based on configuration."""
    loss_type = config.get("type", "weighted_sum")
    task_names = ["classification", "age_regression", "gender_classification"]
    
    if loss_type == "weighted_sum":
        return WeightedSumLoss(
            task_names=task_names,
            task_weights=config.get("task_weights"),
            loss_functions=config.get("loss_functions")
        )
    elif loss_type == "uncertainty_weighted":
        return UncertaintyWeightedLoss(
            task_names=task_names,
            learnable_weights=config.get("learnable_weights", True)
        )
    elif loss_type == "dynamic_weighted":
        return DynamicWeightedLoss(
            task_names=task_names,
            alpha=config.get("alpha", 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Multi-task Learning Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="mtl_experiment",
                       help="Name of the experiment")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.device:
        config.device = args.device
    if args.use_wandb:
        config.use_wandb = True
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config.output_dir, config.experiment_name)
    config.output_dir = exp_dir
    
    # Save configuration
    save_config(config, os.path.join(exp_dir, "config.yaml"))
    
    print("="*60)
    print("MULTI-TASK LEARNING EXPERIMENT")
    print("="*60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"Output directory: {exp_dir}")
    print("="*60)
    
    # Create data loaders
    print("Loading data...")
    data_loader = create_data_loader(
        dataset_config=config.data,
        batch_size=config.batch_size,
        num_workers=config.get("num_workers", 4),
        seed=config.seed
    )
    
    print(f"Train samples: {len(data_loader.train_dataset)}")
    print(f"Val samples: {len(data_loader.val_dataset)}")
    print(f"Test samples: {len(data_loader.test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config.model, device)
    print_model_summary(model)
    
    # Create loss function
    print("Creating loss function...")
    loss_function = create_loss_function(config.training.losses)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(model, config.training.optimizer)
    
    # Create scheduler
    scheduler = None
    if "scheduler" in config.training:
        print("Creating scheduler...")
        scheduler = create_scheduler(optimizer, config.training.scheduler)
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=config.get("use_wandb", False)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['loss']
    
    # Train model
    print("Starting training...")
    training_results = trainer.train(
        num_epochs=config.num_epochs,
        save_dir=os.path.join(exp_dir, "checkpoints")
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = Evaluator(
        model=model,
        test_loader=data_loader.test_loader,
        device=device,
        save_dir=os.path.join(exp_dir, "outputs")
    )
    
    evaluation_metrics = evaluator.evaluate()
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = MTLVisualizer(save_dir=os.path.join(exp_dir, "assets"))
    
    # Plot training curves
    visualizer.plot_training_curves(
        train_losses=training_results["train_losses"],
        val_losses=training_results["val_losses"],
        learning_rates=training_results["learning_rates"]
    )
    
    # Plot metrics comparison
    visualizer.plot_metrics_comparison(evaluation_metrics)
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard(
        training_history=training_results,
        evaluation_metrics=evaluation_metrics
    )
    
    print("="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Results saved to: {exp_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
