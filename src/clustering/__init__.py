# src/clustering/__init__.py
"""
Character Clustering Module

This module provides functionality for clustering detected characters
from mathematical expressions using various algorithms and evaluation metrics.
"""

from .feature_extraction import YOLOFeatureExtractor, extract_features_from_dataset_yolo
from .clustering_methods import CharacterClusterer
from .cluster_evaluation import ClusterEvaluator
from .visualization import ClusterVisualizer

__all__ = [
    'YOLOFeatureExtractor',
    'extract_features_from_dataset_yolo',
    'CharacterClusterer',
    'ClusterEvaluator',
    'ClusterVisualizer'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Ahmadreza Farvardin'