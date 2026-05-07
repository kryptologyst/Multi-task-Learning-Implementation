# Multi-task Learning Implementation

A comprehensive framework for multi-task learning in computer vision, featuring classical baselines, advanced MTL methods, and proper evaluation.

**Author:** [kryptologyst](https://github.com/kryptologyst)  
**GitHub:** https://github.com/kryptologyst

## Safety and Ethics Disclaimer

⚠️ **IMPORTANT: This is a research demonstration only.**

- This implementation uses **synthetic attributes** for educational purposes
- **Not suitable for production decisions** or real-world applications
- Age and gender predictions are synthetic and not based on real biometric data
- Use responsibly and in accordance with applicable laws and regulations
- **Not for production decisions/control**

## Overview

This project implements a comprehensive multi-task learning framework that simultaneously performs:

1. **Image Classification** (CIFAR-10 classes)
2. **Age Regression** (synthetic age prediction)
3. **Gender Classification** (synthetic gender prediction)

The framework includes:
- Multiple model architectures (ResNet-based, baseline CNN)
- Advanced MTL methods (uncertainty weighting, gradient surgery, PCGrad)
- Comprehensive evaluation metrics
- Interactive visualization tools
- Reproducible experiments

## Architecture

### Model Types

1. **ResNet MTL**: Pre-trained ResNet backbone with task-specific heads
2. **Baseline CNN**: Simple convolutional network for comparison
3. **Uncertainty Weighted MTL**: Implements Kendall et al. uncertainty weighting
4. **Gradient Surgery MTL**: Implements Yu et al. gradient surgery
5. **PCGrad MTL**: Implements gradient conflict resolution

### Task-Specific Heads

- **Classification Head**: 10-class CIFAR-10 classification
- **Age Regression Head**: Continuous age prediction
- **Gender Classification Head**: Binary gender classification

## Features

### Multi-task Learning Methods

- **Baseline**: Simple weighted sum of task losses
- **Uncertainty Weighting**: Learnable uncertainty parameters
- **Gradient Surgery**: Conflict resolution between task gradients
- **PCGrad**: Projected gradient descent for MTL
- **Dynamic Weighting**: Adaptive task weight adjustment

### Evaluation Metrics

**Classification Tasks:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)

**Regression Tasks:**
- RMSE, MAE, MAPE, SMAPE
- R² Score
- Scatter plots and residual analysis

### Visualization Tools

- Training curves and learning rate schedules
- Confusion matrices
- Regression scatter plots
- Model comparison charts
- Interactive dashboards
- Feature importance visualization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Multi-task-Learning-Implementation.git
cd Multi-task-Learning-Implementation

# Install dependencies
pip install -e .

# Or install manually
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
pip install streamlit plotly omegaconf hydra-core
pip install albumentations opencv-python tqdm wandb
```

### Basic Usage

```python
from mtl.models import ResNetMTL
from mtl.data import create_data_loader
from mtl.train import Trainer

# Create model
model = ResNetMTL(
    backbone="resnet50",
    pretrained=True,
    classification={"num_classes": 10},
    age_regression={},
    gender_classification={"num_classes": 2}
)

# Create data loader
data_loader = create_data_loader(dataset_config)

# Train model
trainer = Trainer(model, data_loader.train_loader, data_loader.val_loader, ...)
trainer.train(num_epochs=100)
```

### Training Script

```bash
# Basic training
python train.py --config configs/config.yaml

# With custom experiment name
python train.py --experiment-name "my_experiment" --device cuda

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth

# Use Weights & Biases logging
python train.py --use-wandb
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo.py

# Or with custom port
streamlit run demo.py --server.port 8502
```

## 📁 Project Structure

```
├── src/mtl/                    # Main package
│   ├── models/                 # Model architectures
│   ├── data/                   # Data loading and preprocessing
│   ├── losses/                 # Loss functions
│   ├── metrics/                # Evaluation metrics
│   ├── train/                  # Training utilities
│   ├── eval/                   # Evaluation utilities
│   ├── viz/                    # Visualization tools
│   └── utils/                  # Utility functions
├── configs/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── model/                 # Model configurations
│   ├── data/                  # Data configurations
│   ├── training/              # Training configurations
│   └── evaluation/            # Evaluation configurations
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
├── demo.py                     # Interactive demo
├── train.py                    # Training script
└── README.md                   # This file
```

## Configuration

The framework uses Hydra for configuration management. Key configuration options:

```yaml
# configs/config.yaml
experiment_name: "mtl_experiment"
seed: 42
device: "auto"

# Model configuration
model:
  backbone: "resnet50"
  pretrained: true
  freeze_backbone: false

# Training configuration
num_epochs: 100
batch_size: 32
learning_rate: 0.001

# Multi-task specific
task_weights:
  classification: 1.0
  age_regression: 1.0
  gender_classification: 1.0

# Advanced MTL methods
mtl_method: "baseline"  # baseline, uncertainty_weighting, gradient_surgery, pcgrad
```

## Expected Results

### Baseline Performance (CIFAR-10 + Synthetic Attributes)

| Model | Classification Acc | Age RMSE | Gender Acc | Total Params |
|-------|-------------------|----------|------------|--------------|
| Baseline CNN | ~65% | ~8.5 | ~75% | ~2M |
| ResNet MTL | ~85% | ~6.2 | ~85% | ~25M |
| Uncertainty Weighted | ~87% | ~5.8 | ~87% | ~25M |
| Gradient Surgery | ~86% | ~6.0 | ~86% | ~25M |

*Note: Results may vary based on random seeds and training conditions.*

## Advanced Features

### Uncertainty Weighting

Implements the method from "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., 2018).

```python
from mtl.models import UncertaintyWeightedMTL

model = UncertaintyWeightedMTL(base_model)
# Automatically learns uncertainty parameters during training
```

### Gradient Surgery

Implements conflict resolution between task gradients:

```python
from mtl.models import GradientSurgeryMTL

model = GradientSurgeryMTL(base_model)
# Applies gradient surgery during backward pass
```

### Dynamic Task Weighting

Automatically adjusts task weights based on loss magnitudes:

```python
from mtl.losses import DynamicWeightedLoss

loss_fn = DynamicWeightedLoss(task_names, alpha=0.5)
```

## Evaluation

### Comprehensive Metrics

The framework provides extensive evaluation capabilities:

```python
from mtl.eval import Evaluator

evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()

# Metrics include:
# - Classification: accuracy, precision, recall, f1, confusion matrix
# - Regression: rmse, mae, r2, mape, scatter plots
# - Calibration: ece, mce
```

### Model Comparison

Compare different models side-by-side:

```python
comparison = evaluator.compare_models(other_evaluator, ["Model A", "Model B"])
```

## Visualization

### Training Curves

```python
from mtl.viz import MTLVisualizer

visualizer = MTLVisualizer()
visualizer.plot_training_curves(train_losses, val_losses, learning_rates)
```

### Interactive Dashboard

Create interactive Plotly dashboards:

```python
visualizer.create_interactive_dashboard(training_history, evaluation_metrics)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=mtl

# Run specific test file
pytest tests/test_mtl.py
```

## Research Background

This implementation is based on several key papers:

1. **Multi-Task Learning Using Uncertainty to Weigh Losses** (Kendall et al., 2018)
2. **Gradient Surgery for Multi-Task Learning** (Yu et al., 2020)
3. **A Survey on Multi-Task Learning** (Zhang & Yang, 2021)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Hydra](https://hydra.cc/) for configuration management
- [Streamlit](https://streamlit.io/) for the interactive demo
- [Plotly](https://plotly.com/) for interactive visualizations

## Contact

**Author:** kryptologyst  
**GitHub:** [https://github.com/kryptologyst](https://github.com/kryptologyst)

---

**⚠️ Remember: This is a research demonstration. Not suitable for production use.**
# Multi-task-Learning-Implementation
