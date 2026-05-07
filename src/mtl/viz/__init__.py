"""
Visualization Module for Multi-task Learning

This module provides comprehensive visualization capabilities for
multi-task learning experiments including training curves, metrics,
and model interpretability.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from ..models import BaseMTLModel
from ..utils import get_device


class MTLVisualizer:
    """Comprehensive visualizer for multi-task learning experiments."""
    
    def __init__(self, save_dir: str = "assets", style: str = "seaborn-v0_8"):
        self.save_dir = save_dir
        self.style = style
        
        # Set matplotlib style
        plt.style.use(style)
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set default colors
        self.colors = px.colors.qualitative.Set3
        
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           learning_rates: List[float],
                           task_losses: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> None:
        """Plot training curves including loss and learning rate."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Overall loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color=self.colors[0])
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color=self.colors[1])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[0, 1].plot(epochs, learning_rates, color=self.colors[2])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Task-specific losses
        if task_losses:
            for i, (task_name, losses) in enumerate(task_losses.items()):
                axes[1, 0].plot(epochs, losses, label=task_name, color=self.colors[i % len(self.colors)])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Task-Specific Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio (val/train)
        loss_ratio = [v/t if t > 0 else 0 for v, t in zip(val_losses, train_losses)]
        axes[1, 1].plot(epochs, loss_ratio, color=self.colors[3])
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val/Train Loss Ratio')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_comparison(self, 
                              metrics: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None) -> None:
        """Plot comparison of metrics across tasks."""
        # Prepare data
        tasks = list(metrics.keys())
        
        # Classification metrics
        cls_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cls_tasks = [task for task in tasks if 'classification' in task]
        
        # Regression metrics  
        reg_metrics = ['rmse', 'mae', 'r2_score']
        reg_tasks = [task for task in tasks if 'regression' in task]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification metrics
        if cls_tasks and any(m in str(metrics.values()) for m in cls_metrics):
            available_metrics = [m for m in cls_metrics if any(m in task_metrics for task_metrics in metrics.values())]
            x = np.arange(len(cls_tasks))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = [metrics[task].get(metric, 0) for task in cls_tasks]
                axes[0].bar(x + i * width, values, width, label=metric, color=self.colors[i])
            
            axes[0].set_xlabel('Tasks')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Classification Metrics')
            axes[0].set_xticks(x + width * (len(available_metrics) - 1) / 2)
            axes[0].set_xticklabels(cls_tasks, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Regression metrics
        if reg_tasks and any(m in str(metrics.values()) for m in reg_metrics):
            available_metrics = [m for m in reg_metrics if any(m in task_metrics for task_metrics in metrics.values())]
            x = np.arange(len(reg_tasks))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = [metrics[task].get(metric, 0) for task in reg_tasks]
                axes[1].bar(x + i * width, values, width, label=metric, color=self.colors[i])
            
            axes[1].set_xlabel('Tasks')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Regression Metrics')
            axes[1].set_xticks(x + width * (len(available_metrics) - 1) / 2)
            axes[1].set_xticklabels(reg_tasks, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrices(self, 
                              confusion_matrices: Dict[str, np.ndarray],
                              class_names: Optional[Dict[str, List[str]]] = None,
                              save_path: Optional[str] = None) -> None:
        """Plot confusion matrices for classification tasks."""
        n_tasks = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))
        
        if n_tasks == 1:
            axes = [axes]
        
        for i, (task_name, cm) in enumerate(confusion_matrices.items()):
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {task_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Add class names if provided
            if class_names and task_name in class_names:
                axes[i].set_xticklabels(class_names[task_name], rotation=45)
                axes[i].set_yticklabels(class_names[task_name], rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_regression_scatter(self, 
                              predictions: Dict[str, np.ndarray],
                              targets: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> None:
        """Plot scatter plots for regression tasks."""
        reg_tasks = [task for task in predictions.keys() if 'regression' in task]
        
        if not reg_tasks:
            return
            
        n_tasks = len(reg_tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))
        
        if n_tasks == 1:
            axes = [axes]
        
        for i, task_name in enumerate(reg_tasks):
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Create scatter plot
            axes[i].scatter(target, pred, alpha=0.6, color=self.colors[i])
            
            # Perfect prediction line
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'Regression Plot - {task_name}')
            
            # Add R² score
            r2 = np.corrcoef(target, pred)[0, 1] ** 2
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'regression_scatter.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_comparison(self, 
                            model_results: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None) -> None:
        """Plot comparison between different models."""
        models = list(model_results.keys())
        tasks = list(model_results[models[0]].keys())
        
        # Create subplots for different metric types
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Classification metrics
        cls_metrics = ['accuracy', 'f1_macro']
        cls_tasks = [task for task in tasks if 'classification' in task]
        
        if cls_tasks:
            for metric in cls_metrics:
                if any(metric in model_results[model][task] for model in models for task in cls_tasks):
                    x = np.arange(len(models))
                    width = 0.8 / len(cls_tasks)
                    
                    for i, task in enumerate(cls_tasks):
                        values = [model_results[model][task].get(metric, 0) for model in models]
                        axes[0, 0].bar(x + i * width, values, width, label=task, color=self.colors[i])
                    
                    axes[0, 0].set_xlabel('Models')
                    axes[0, 0].set_ylabel(metric.title())
                    axes[0, 0].set_title(f'{metric.title()} Comparison')
                    axes[0, 0].set_xticks(x + width * (len(cls_tasks) - 1) / 2)
                    axes[0, 0].set_xticklabels(models, rotation=45)
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    break
        
        # Regression metrics
        reg_metrics = ['rmse', 'r2_score']
        reg_tasks = [task for task in tasks if 'regression' in task]
        
        if reg_tasks:
            for metric in reg_metrics:
                if any(metric in model_results[model][task] for model in models for task in reg_tasks):
                    x = np.arange(len(models))
                    width = 0.8 / len(reg_tasks)
                    
                    for i, task in enumerate(reg_tasks):
                        values = [model_results[model][task].get(metric, 0) for model in models]
                        axes[0, 1].bar(x + i * width, values, width, label=task, color=self.colors[i])
                    
                    axes[0, 1].set_xlabel('Models')
                    axes[0, 1].set_ylabel(metric.title())
                    axes[0, 1].set_title(f'{metric.title()} Comparison')
                    axes[0, 1].set_xticks(x + width * (len(reg_tasks) - 1) / 2)
                    axes[0, 1].set_xticklabels(models, rotation=45)
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    break
        
        # Overall performance radar chart
        self._create_radar_chart(axes[1, 0], model_results)
        
        # Performance summary table
        self._create_performance_table(axes[1, 1], model_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'model_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_radar_chart(self, ax, model_results: Dict[str, Dict[str, float]]):
        """Create radar chart for model comparison."""
        models = list(model_results.keys())
        
        # Calculate average scores for each model
        model_scores = {}
        for model in models:
            scores = []
            for task_metrics in model_results[model].values():
                for metric, value in task_metrics.items():
                    if metric in ['accuracy', 'f1_macro', 'r2_score']:
                        scores.append(value)
            model_scores[model] = np.mean(scores) if scores else 0
        
        # Create simple bar chart instead of radar for now
        x = np.arange(len(models))
        values = [model_scores[model] for model in models]
        
        ax.bar(x, values, color=self.colors[:len(models)])
        ax.set_xlabel('Models')
        ax.set_ylabel('Average Score')
        ax.set_title('Overall Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
        
    def _create_performance_table(self, ax, model_results: Dict[str, Dict[str, float]]):
        """Create performance summary table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        models = list(model_results.keys())
        tasks = list(model_results[models[0]].keys())
        
        table_data = []
        for model in models:
            row = [model]
            for task in tasks:
                if 'accuracy' in model_results[model][task]:
                    row.append(f"{model_results[model][task]['accuracy']:.3f}")
                elif 'rmse' in model_results[model][task]:
                    row.append(f"{model_results[model][task]['rmse']:.3f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        # Create table
        headers = ['Model'] + tasks
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Performance Summary', pad=20)
        
    def create_interactive_dashboard(self, 
                                  training_history: Dict,
                                  evaluation_metrics: Dict,
                                  save_path: Optional[str] = None) -> None:
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Learning Rate', 'Task Metrics', 'Model Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        epochs = range(1, len(training_history['train_losses']) + 1)
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['train_losses'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['val_losses'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history['learning_rates'], 
                      name='Learning Rate', line=dict(color='green')),
            row=1, col=2
        )
        
        # Task metrics
        for task_name, metrics in evaluation_metrics.items():
            if 'accuracy' in metrics:
                fig.add_trace(
                    go.Bar(name=f'{task_name} Accuracy', 
                          x=[task_name], y=[metrics['accuracy']]),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="Multi-task Learning Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(os.path.join(self.save_dir, 'interactive_dashboard.html'))
            
    def visualize_feature_importance(self, 
                                   model: BaseMTLModel,
                                   sample_input: torch.Tensor,
                                   task_name: str,
                                   save_path: Optional[str] = None) -> None:
        """Visualize feature importance using gradient-based methods."""
        model.eval()
        sample_input = sample_input.to(next(model.parameters()).device)
        sample_input.requires_grad_(True)
        
        # Forward pass
        predictions = model(sample_input)
        
        if task_name not in predictions:
            raise ValueError(f"Task {task_name} not found in model predictions")
        
        # Compute gradients
        loss = predictions[task_name].sum()
        loss.backward()
        
        # Get gradients
        gradients = sample_input.grad.data.abs().mean(dim=0)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(gradients.cpu().numpy().transpose(1, 2, 0))
        plt.colorbar()
        plt.title(f'Feature Importance - {task_name}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'feature_importance_{task_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
