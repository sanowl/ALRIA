"""
Working Memory Buffer for the multimodal model.
Manages short-term working memory for reasoning processes.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from collections import deque

from ..config import ModelConfig
from ..utils.decorators import timing_decorator, error_handler

import logging
logger = logging.getLogger(__name__)


class WorkingMemoryBuffer:
    """Working memory buffer with context management."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_size = config.working_memory_size
        self.memory_buffer = deque(maxlen=self.max_size)
        
        # Current context tracking
        self.current_context = {}
        self.context_history = []
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_additions': 0,
            'total_retrievals': 0,
            'context_switches': 0
        }
    
    @timing_decorator
    @error_handler(default_return=False, log_error=True)
    def add_item(self, 
                item_type: str, 
                content: Any, 
                importance: float = 1.0,
                metadata: Optional[Dict] = None) -> bool:
        """Add item to working memory with importance weighting."""
        
        with self.lock:
            # Create memory item
            memory_item = {
                'id': self.stats['total_additions'],
                'type': item_type,
                'content': content,
                'importance': importance,
                'timestamp': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'metadata': metadata or {}
            }
            
            # Add to buffer (automatically evicts oldest if full)
            self.memory_buffer.append(memory_item)
            
            self.stats['total_additions'] += 1
            
            logger.debug(f"Added {item_type} item to working memory")
            return True
    
    @timing_decorator
    @error_handler(default_return=[], log_error=True)
    def retrieve_items(self, 
                      item_type: Optional[str] = None,
                      min_importance: float = 0.0,
                      max_items: Optional[int] = None) -> List[Dict]:
        """Retrieve items from working memory with filtering."""
        
        with self.lock:
            if not self.memory_buffer:
                return []
            
            # Filter items
            filtered_items = []
            for item in self.memory_buffer:
                if item_type and item['type'] != item_type:
                    continue
                if item['importance'] < min_importance:
                    continue
                
                # Update access statistics
                item['access_count'] += 1
                item['last_accessed'] = time.time()
                
                filtered_items.append(item.copy())
            
            # Sort by importance and recency
            def sort_key(item):
                recency_score = 1.0 / (1.0 + (time.time() - item['timestamp']) / 3600)
                return item['importance'] * recency_score
            
            filtered_items.sort(key=sort_key, reverse=True)
            
            # Limit results
            if max_items:
                filtered_items = filtered_items[:max_items]
            
            self.stats['total_retrievals'] += 1
            
            logger.debug(f"Retrieved {len(filtered_items)} items from working memory")
            return filtered_items
    
    @timing_decorator
    @error_handler(log_error=True)
    def update_context(self, new_context: Dict[str, Any]):
        """Update current working context."""
        with self.lock:
            # Store previous context in history
            if self.current_context:
                self.context_history.append({
                    'context': self.current_context.copy(),
                    'timestamp': time.time()
                })
                
                # Limit history size
                if len(self.context_history) > 10:
                    self.context_history.pop(0)
            
            # Update current context
            self.current_context = new_context.copy()
            self.stats['context_switches'] += 1
            
            logger.debug("Updated working memory context")
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current working context."""
        with self.lock:
            return self.current_context.copy()
    
    def get_context_history(self) -> List[Dict]:
        """Get context history."""
        with self.lock:
            return self.context_history.copy()
    
    def clear_memory(self):
        """Clear all working memory."""
        with self.lock:
            self.memory_buffer.clear()
            self.current_context.clear()
            self.context_history.clear()
            
            logger.info("Cleared working memory")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self.lock:
            return {
                'current_size': len(self.memory_buffer),
                'max_size': self.max_size,
                'utilization': len(self.memory_buffer) / self.max_size,
                'statistics': self.stats.copy(),
                'context_active': bool(self.current_context),
                'context_history_size': len(self.context_history)
            } 