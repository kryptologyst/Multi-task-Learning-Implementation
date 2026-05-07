"""
Training Module for Multi-task Learning

This module handles training loops, optimization, and advanced MTL techniques
like gradient surgery and uncertainty weighting.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import DictConfig

from ..models import BaseMTLModel
from ..losses import MultiTaskLoss
from ..metrics import MultiTaskMetrics, CalibrationMetrics
from ..utils import get_device, set_seed, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer for multi-task learning models."""
    
    def __init__(
        self,
        model: BaseMTLModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_function: MultiTaskLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        config: Optional[DictConfig] = None,
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device(device)
        self.config = config or {}
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics
        task_types = {
            "classification": "classification",
            "age_regression": "regression", 
            "gender_classification": "classification"
        }
        self.metrics = MultiTaskMetrics(
            task_names=list(task_types.keys()),
            task_types=task_types
        )
        self.calibration_metrics = CalibrationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="multi-task-learning",
                config=self.config,
                name=f"mtl_experiment_{int(time.time())}"
            )
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {task: 0.0 for task in self.model.task_names}
        epoch_losses["total"] = 0.0
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(images)
                
                # Compute loss
                loss = self.loss_function(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if specified
                if self.config.get("gradient_clip_val"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                
                # Apply advanced MTL techniques
                if hasattr(self.model, 'apply_gradient_surgery'):
                    self.model.apply_gradient_surgery()
                elif hasattr(self.model, 'apply_pcgrad'):
                    self.model.apply_pcgrad()
                
                self.optimizer.step()
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Track losses
                epoch_losses["total"] += loss.item()
                for task_name in self.model.task_names:
                    if task_name in predictions and task_name in targets:
                        task_loss = self._compute_task_loss(predictions[task_name], targets[task_name], task_name)
                        epoch_losses[task_name] += task_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to wandb
                if self.use_wandb and batch_idx % self.config.get("log_frequency", 10) == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": self.current_epoch,
                        "batch": batch_idx
                    })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {task: 0.0 for task in self.model.task_names}
        epoch_losses["total"] = 0.0
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                loss = self.loss_function(predictions, targets)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Track losses
                epoch_losses["total"] += loss.item()
                for task_name in self.model.task_names:
                    if task_name in predictions and task_name in targets:
                        task_loss = self._compute_task_loss(predictions[task_name], targets[task_name], task_name)
                        epoch_losses[task_name] += task_loss
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def _compute_task_loss(self, prediction: torch.Tensor, target: torch.Tensor, task_name: str) -> float:
        """Compute loss for a specific task."""
        if "classification" in task_name:
            loss_fn = nn.CrossEntropyLoss()
        elif "regression" in task_name:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        return loss_fn(prediction, target).item()
        
    def train(self, num_epochs: int, save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate_epoch()
            
            # Compute metrics
            metrics = self.metrics.compute_metrics()
            self.metrics.reset()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Track losses
            self.train_losses.append(train_losses["total"])
            self.val_losses.append(val_losses["total"])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            
            for task_name in self.model.task_names:
                if task_name in train_losses:
                    print(f"Train {task_name}: {train_losses[task_name]:.4f}")
                if task_name in val_losses:
                    print(f"Val {task_name}: {val_losses[task_name]:.4f}")
                    
            # Print metrics
            for task_name, task_metrics in metrics.items():
                print(f"\n{task_name} Metrics:")
                for metric_name, value in task_metrics.items():
                    if metric_name != "confusion_matrix":
                        print(f"  {metric_name}: {value:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_losses["total"],
                    "val/loss": val_losses["total"],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
                
                # Add task-specific losses
                for task_name in self.model.task_names:
                    if task_name in train_losses:
                        log_dict[f"train/{task_name}_loss"] = train_losses[task_name]
                    if task_name in val_losses:
                        log_dict[f"val/{task_name}_loss"] = val_losses[task_name]
                
                # Add metrics
                for task_name, task_metrics in metrics.items():
                    for metric_name, value in task_metrics.items():
                        if metric_name != "confusion_matrix":
                            log_dict[f"val/{task_name}_{metric_name}"] = value
                
                wandb.log(log_dict)
            
            # Save checkpoint
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_losses["total"],
                    metrics=metrics,
                    path=os.path.join(save_dir, "best_model.pth")
                )
                
            # Save regular checkpoint
            if (epoch + 1) % self.config.get("save_frequency", 10) == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_losses["total"],
                    metrics=metrics,
                    path=os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                )
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        if self.use_wandb:
            wandb.finish()
            
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "final_metrics": final_metrics
        }
        
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate the model on validation set."""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Evaluation"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(images)
                self.metrics.update(predictions, targets)
        
        metrics = self.metrics.compute_metrics()
        return metrics
