"""
Text embedding utilities for Enhanced MMaDA.
Provides various methods for generating text embeddings.
"""

import torch
import numpy as np
import logging
from typing import List
from collections import Counter
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import EnhancedMMaDAConfig
from .decorators import timing_decorator, error_handler

logger = logging.getLogger(__name__)


class AdvancedTextEmbedding:
    """Advanced text embedding with multiple methods and fallbacks."""
    
    def __init__(self, config: EnhancedMMaDAConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model storage
        self.bert_tokenizer = None
        self.bert_model = None
        self.tfidf_vectorizer = None
        
        # Initialize models with error handling
        self._initialize_models()
    
    @error_handler(log_error=True)
    def _initialize_models(self):
        """Initialize embedding models with proper error handling."""
        try:
            # Initialize BERT
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.bert_model.eval()
            logger.info("BERT model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            self.bert_model = None
    
    @timing_decorator
    def get_embeddings(self, texts: List[str], method: str = 'bert') -> np.ndarray:
        """Get embeddings using specified method with fallbacks."""
        if method == 'bert' and self.bert_model is not None:
            return self._get_bert_embeddings(texts)
        elif method == 'tfidf':
            return self._get_tfidf_embeddings(texts)
        else:
            logger.warning(f"Method '{method}' not available, falling back to simple embeddings")
            return self._get_simple_embeddings(texts)
    
    @error_handler(default_return=np.array([]), log_error=True)
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings with proper error handling."""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Get embeddings
                outputs = self.bert_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding.squeeze())
        
        return np.array(embeddings)
    
    @error_handler(default_return=np.array([]), log_error=True)
    def _get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fit and transform
        embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
        return embeddings
    
    def _get_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple word-based embeddings as fallback."""
        # Create simple bag-of-words embeddings
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        word_to_idx = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        embeddings = []
        for text in texts:
            embedding = np.zeros(len(word_to_idx))
            words = Counter(text.lower().split())
            
            for word, count in words.items():
                if word in word_to_idx:
                    embedding[word_to_idx[word]] = count
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings) 