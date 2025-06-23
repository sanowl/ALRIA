"""
Enhanced MMaDA: Multimodal Attention and Domain Adaptation Framework
A comprehensive AI system with advanced reasoning, memory, and adaptation capabilities.
"""

__version__ = "1.0.0"
__author__ = "Enhanced MMaDA Team"

from .config import EnhancedMMaDAConfig
from .models import EnhancedMMaDAModel
from .training import EnhancedMMaDATrainer

__all__ = [
    'EnhancedMMaDAConfig',
    'EnhancedMMaDAModel', 
    'EnhancedMMaDATrainer',
] 