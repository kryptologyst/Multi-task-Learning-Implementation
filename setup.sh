#!/bin/bash

# Multi-task Learning Setup Script
# Author: kryptologyst
# GitHub: https://github.com/kryptologyst

set -e

echo "🚀 Setting up Multi-task Learning Environment"
echo "============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest pytest-cov pre-commit

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed}
mkdir -p outputs/{checkpoints,logs,assets}
mkdir -p assets/{plots,models,results}

# Download sample data (CIFAR-10 will be downloaded automatically)
echo "📥 Data will be downloaded automatically on first run"

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run training: python train.py"
echo "3. Launch demo: streamlit run demo.py"
echo ""
echo "Author: kryptologyst"
echo "GitHub: https://github.com/kryptologyst"
