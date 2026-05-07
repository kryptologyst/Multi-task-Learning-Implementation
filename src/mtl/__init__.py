"""
Multi-task Learning Implementation

A comprehensive framework for multi-task learning in computer vision,
featuring classical baselines, advanced MTL methods, and proper evaluation.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

__version__ = "0.1.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from . import data, models, losses, metrics, train, eval, viz, utils

__all__ = ["data", "models", "losses", "metrics", "train", "eval", "viz", "utils"]
