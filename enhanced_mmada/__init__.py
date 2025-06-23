"""
Multimodal Language Model with Memory
A practical multimodal language model with memory systems and robust training.
"""

__version__ = "1.0.0"
__author__ = "Multimodal Model Team"

from .config import ModelConfig
from .models import MultimodalModel
from .training import ModelTrainer

__all__ = [
    'ModelConfig',
    'MultimodalModel', 
    'ModelTrainer',
] 