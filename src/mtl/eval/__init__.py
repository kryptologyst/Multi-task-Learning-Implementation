"""
Evaluation Module for Multi-task Learning

This module provides comprehensive evaluation capabilities including
metrics computation, visualization, and model comparison.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import BaseMTLModel
from ..metrics import MultiTaskMetrics, CalibrationMetrics
from ..utils import get_device


class Evaluator:
    """Comprehensive evaluator for multi-task learning models."""
    
    def __init__(
        self,
        model: BaseMTLModel,
        test_loader: DataLoader,
        device: str = "auto",
        save_dir: str = "outputs",
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = get_device(device)
        self.save_dir = save_dir
        
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
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate(self, save_results: bool = True) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Comprehensive evaluation of the model."""
        print("Starting comprehensive evaluation...")
        
        # Set model to evaluation mode
        self.model.eval()
        self.metrics.reset()
        
        # Collect all predictions and targets
        all_predictions = {task: [] for task in self.model.task_names}
        all_targets = {task: [] for task in self.model.task_names}
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Get predictions
                predictions = self.model(images)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Store for detailed analysis
                for task_name in self.model.task_names:
                    if task_name in predictions and task_name in targets:
                        pred = predictions[task_name].detach().cpu()
                        target = targets[task_name].detach().cpu()
                        
                        if "classification" in task_name:
                            pred = torch.argmax(pred, dim=1)
                            
                        all_predictions[task_name].extend(pred.numpy())
                        all_targets[task_name].extend(target.numpy())
        
        # Compute metrics
        metrics = self.metrics.compute_metrics()
        
        # Add calibration metrics for classification tasks
        for task_name in self.model.task_names:
            if "classification" in task_name and task_name in all_predictions:
                # Get original predictions (before argmax)
                pred_probs = []
                targets_cls = []
                
                with torch.no_grad():
                    for images, targets in self.test_loader:
                        images = images.to(self.device)
                        targets = {k: v.to(self.device) for k, v in targets.items()}
                        
                        predictions = self.model(images)
                        if task_name in predictions and task_name in targets:
                            pred_probs.append(predictions[task_name].detach().cpu())
                            targets_cls.append(targets[task_name].detach().cpu())
                
                if pred_probs:
                    pred_probs = torch.cat(pred_probs)
                    targets_cls = torch.cat(targets_cls)
                    
                    ece = self.calibration_metrics.compute_ece(pred_probs, targets_cls)
                    mce = self.calibration_metrics.compute_mce(pred_probs, targets_cls)
                    
                    if task_name not in metrics:
                        metrics[task_name] = {}
                    metrics[task_name]["ece"] = ece
                    metrics[task_name]["mce"] = mce
        
        # Print results
        self._print_results(metrics)
        
        # Create visualizations
        if self.save_dir:
            self._create_visualizations(metrics, all_predictions, all_targets)
            
        # Save results
        if save_results:
            self._save_results(metrics, all_predictions, all_targets)
            
        return metrics
        
    def _print_results(self, metrics: Dict[str, Dict[str, Union[float, np.ndarray]]]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for task_name, task_metrics in metrics.items():
            print(f"\n{task_name.upper()}:")
            print("-" * 40)
            
            for metric_name, value in task_metrics.items():
                if metric_name == "confusion_matrix":
                    print(f"{metric_name}:")
                    print(value)
                else:
                    if isinstance(value, float):
                        print(f"{metric_name}: {value:.4f}")
                    else:
                        print(f"{metric_name}: {value}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Classification tasks
        cls_tasks = [task for task in metrics.keys() if "classification" in task]
        if cls_tasks:
            print(f"\nClassification Tasks ({len(cls_tasks)}):")
            for task in cls_tasks:
                if "accuracy" in metrics[task]:
                    print(f"  {task}: Accuracy = {metrics[task]['accuracy']:.4f}")
        
        # Regression tasks
        reg_tasks = [task for task in metrics.keys() if "regression" in task]
        if reg_tasks:
            print(f"\nRegression Tasks ({len(reg_tasks)}):")
            for task in reg_tasks:
                if "rmse" in metrics[task]:
                    print(f"  {task}: RMSE = {metrics[task]['rmse']:.4f}")
                if "r2_score" in metrics[task]:
                    print(f"  {task}: R² = {metrics[task]['r2_score']:.4f}")
        
    def _create_visualizations(self, metrics: Dict, predictions: Dict, targets: Dict):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion matrices for classification tasks
        cls_tasks = [task for task in metrics.keys() if "classification" in task]
        if cls_tasks:
            fig, axes = plt.subplots(1, len(cls_tasks), figsize=(6*len(cls_tasks), 5))
            if len(cls_tasks) == 1:
                axes = [axes]
                
            for i, task in enumerate(cls_tasks):
                if "confusion_matrix" in metrics[task]:
                    cm = metrics[task]["confusion_matrix"]
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'Confusion Matrix - {task}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Regression scatter plots
        reg_tasks = [task for task in metrics.keys() if "regression" in task]
        if reg_tasks:
            fig, axes = plt.subplots(1, len(reg_tasks), figsize=(6*len(reg_tasks), 5))
            if len(reg_tasks) == 1:
                axes = [axes]
                
            for i, task in enumerate(reg_tasks):
                pred = np.array(predictions[task])
                target = np.array(targets[task])
                
                axes[i].scatter(target, pred, alpha=0.6)
                
                # Perfect prediction line
                min_val = min(target.min(), pred.min())
                max_val = max(target.max(), pred.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[i].set_xlabel('True Values')
                axes[i].set_ylabel('Predicted Values')
                axes[i].set_title(f'Regression Plot - {task}')
                
                # Add R² score
                if "r2_score" in metrics[task]:
                    r2 = metrics[task]["r2_score"]
                    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                               transform=axes[i].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'regression_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Metrics comparison bar chart
        self._create_metrics_comparison_chart(metrics)
        
        # 4. Task performance radar chart
        self._create_performance_radar_chart(metrics)
        
    def _create_metrics_comparison_chart(self, metrics: Dict):
        """Create bar chart comparing metrics across tasks."""
        # Prepare data
        task_names = list(metrics.keys())
        metric_names = set()
        for task_metrics in metrics.values():
            metric_names.update(task_metrics.keys())
        
        # Remove non-numeric metrics
        metric_names = [m for m in metric_names if m not in ['confusion_matrix']]
        
        # Create subplots for different metric types
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Classification metrics
        cls_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cls_tasks = [task for task in task_names if "classification" in task]
        
        if cls_tasks and any(m in metric_names for m in cls_metrics):
            available_metrics = [m for m in cls_metrics if m in metric_names]
            x = np.arange(len(cls_tasks))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = [metrics[task].get(metric, 0) for task in cls_tasks]
                axes[0].bar(x + i * width, values, width, label=metric)
            
            axes[0].set_xlabel('Tasks')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Classification Metrics')
            axes[0].set_xticks(x + width * (len(available_metrics) - 1) / 2)
            axes[0].set_xticklabels(cls_tasks, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Regression metrics
        reg_metrics = ['rmse', 'mae', 'r2_score']
        reg_tasks = [task for task in task_names if "regression" in task]
        
        if reg_tasks and any(m in metric_names for m in reg_metrics):
            available_metrics = [m for m in reg_metrics if m in metric_names]
            x = np.arange(len(reg_tasks))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = [metrics[task].get(metric, 0) for task in reg_tasks]
                axes[1].bar(x + i * width, values, width, label=metric)
            
            axes[1].set_xlabel('Tasks')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Regression Metrics')
            axes[1].set_xticks(x + width * (len(available_metrics) - 1) / 2)
            axes[1].set_xticklabels(reg_tasks, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(2, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_performance_radar_chart(self, metrics: Dict):
        """Create radar chart showing overall performance."""
        # Define performance categories
        categories = {
            'Classification': ['accuracy', 'f1_macro'],
            'Regression': ['r2_score'],
            'Calibration': ['ece', 'mce']
        }
        
        # Calculate average scores for each category
        category_scores = {}
        for category, metric_list in categories.items():
            scores = []
            for metric in metric_list:
                for task_metrics in metrics.values():
                    if metric in task_metrics:
                        scores.append(task_metrics[metric])
            if scores:
                category_scores[category] = np.mean(scores)
            else:
                category_scores[category] = 0.0
        
        # Create radar chart
        if len(category_scores) >= 3:
            categories_list = list(category_scores.keys())
            scores_list = list(category_scores.values())
            
            # Normalize scores to 0-1 range
            scores_list = [max(0, min(1, score)) for score in scores_list]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories_list), endpoint=False).tolist()
            scores_list += scores_list[:1]  # Complete the circle
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, scores_list, 'o-', linewidth=2, label='Performance')
            ax.fill(angles, scores_list, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories_list)
            ax.set_ylim(0, 1)
            ax.set_title('Overall Performance Radar Chart', size=16, pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'performance_radar.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
    def _save_results(self, metrics: Dict, predictions: Dict, targets: Dict):
        """Save evaluation results to files."""
        # Save metrics as JSON
        metrics_serializable = {}
        for task_name, task_metrics in metrics.items():
            metrics_serializable[task_name] = {}
            for metric_name, value in task_metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_serializable[task_name][metric_name] = value.tolist()
                else:
                    metrics_serializable[task_name][metric_name] = value
        
        with open(os.path.join(self.save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Save predictions and targets
        results = {
            'predictions': predictions,
            'targets': targets
        }
        
        with open(os.path.join(self.save_dir, 'predictions_targets.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.save_dir}")
        
    def compare_models(self, other_evaluator: 'Evaluator', 
                      model_names: List[str] = None) -> Dict:
        """Compare this model with another model."""
        if model_names is None:
            model_names = ['Model 1', 'Model 2']
            
        # Evaluate both models
        metrics1 = self.evaluate(save_results=False)
        metrics2 = other_evaluator.evaluate(save_results=False)
        
        # Create comparison
        comparison = {}
        for task_name in metrics1.keys():
            if task_name in metrics2:
                comparison[task_name] = {}
                for metric_name in metrics1[task_name].keys():
                    if metric_name in metrics2[task_name] and metric_name != 'confusion_matrix':
                        comparison[task_name][metric_name] = {
                            model_names[0]: metrics1[task_name][metric_name],
                            model_names[1]: metrics2[task_name][metric_name]
                        }
        
        return comparison
