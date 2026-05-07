"""
Utility Functions for Multi-task Learning

This module contains various utility functions for device management,
seeding, checkpointing, and other common operations.
"""

import os
import random
import json
from typing import Any, Dict, Optional, Union
import torch
import numpy as np
from omegaconf import DictConfig


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            print("Using CPU device")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("MPS not available, falling back to CPU")
            device = "cpu"
            
    return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Random seed set to {seed}")


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, Any],
                   path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The scheduler state (optional)
        epoch: Current epoch
        loss: Current loss value
        metrics: Evaluation metrics
        path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   path: str = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        path: Path to the checkpoint file
        
    Returns:
        Dict containing checkpoint information
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Dict containing total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict containing size information in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create experiment directory structure.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to the experiment directory
    """
    import time
    timestamp = int(time.time())
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    subdirs = ['checkpoints', 'logs', 'outputs', 'assets']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
        
    return exp_dir


def save_config(config: DictConfig, path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        path: Path to save the configuration
    """
    with open(path, 'w') as f:
        json.dump(dict(config), f, indent=2)
    print(f"Configuration saved to {path}")


def load_config(path: str) -> DictConfig:
    """
    Load configuration from file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return DictConfig(config_dict)


def format_time(seconds: float) -> str:
    """
    Format time in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(num: Union[int, float]) -> str:
    """
    Format large numbers in a human-readable way.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: The model to summarize
        input_size: Input size for the model (batch_size, channels, height, width)
    """
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Parameter count
    param_info = count_parameters(model)
    print(f"Total parameters: {format_number(param_info['total'])}")
    print(f"Trainable parameters: {format_number(param_info['trainable'])}")
    print(f"Non-trainable parameters: {format_number(param_info['non_trainable'])}")
    
    # Model size
    size_info = get_model_size(model)
    print(f"Model size: {size_info['total_size_mb']:.2f} MB")
    
    # Task information
    if hasattr(model, 'task_names'):
        print(f"Number of tasks: {model.num_tasks}")
        print(f"Tasks: {', '.join(model.task_names)}")
    
    print("="*60)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: The optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer,
                       warmup_iters: int,
                       warmup_factor: float = 0.1) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a warmup learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        warmup_iters: Number of warmup iterations
        warmup_factor: Warmup factor
        
    Returns:
        Warmup scheduler
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def cosine_annealing_lr_scheduler(optimizer: torch.optim.Optimizer,
                                 T_max: int,
                                 eta_min: float = 0) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a cosine annealing learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        
    Returns:
        Cosine annealing scheduler
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)


def reduce_lr_on_plateau_scheduler(optimizer: torch.optim.Optimizer,
                                  mode: str = 'min',
                                  factor: float = 0.1,
                                  patience: int = 10,
                                  threshold: float = 1e-4) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a reduce on plateau learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        mode: 'min' or 'max'
        factor: Factor by which to reduce learning rate
        patience: Number of epochs to wait before reducing
        threshold: Threshold for measuring improvement
        
    Returns:
        Reduce on plateau scheduler
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
    )


def create_optimizer(model: torch.nn.Module, 
                    optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: The model
        optimizer_config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_config.get('type', 'adam')
    
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.01),
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0)
        )
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: The optimizer
        scheduler_config: Scheduler configuration
        
    Returns:
        Configured scheduler
    """
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
