"""
Spectra Vis (Tensor Edition)
Neural Network Tensor Activation Visualization

A tool for observing and visualizing tensor activations during
Large Language Model inference using GGUF models.

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

__version__ = "1.0.0-beta"
__author__ = "3rdEyeVisuals"

from .collector import TensorCollector, create_callback
from .model_profiles import get_model_profile, SUPPORTED_MODELS

__all__ = [
    "TensorCollector",
    "create_callback",
    "get_model_profile",
    "SUPPORTED_MODELS",
]
