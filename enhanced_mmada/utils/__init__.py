"""
Utility modules for the multimodal model.
Contains common utilities, decorators, and helper functions.
"""

from .decorators import timing_decorator
from .text_embeddings import TextEmbedding

__all__ = [
    'timing_decorator',
    'TextEmbedding',
] 