"""
Episodic Memory Bank for the multimodal model.
Memory system with indexing and retrieval capabilities.
"""

import time
import pickle
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..config import ModelConfig
from ..utils.decorators import timing_decorator, error_handler
from ..utils.text_embeddings import TextEmbedding

import logging
logger = logging.getLogger(__name__)


class EpisodicMemoryBank:
    """Episodic memory system with indexing and retrieval."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_size = config.episodic_memory_size
        self.embedding_dim = config.memory_embedding_dim
        self.decay_factor = config.memory_decay_factor
        
        # Memory storage
        self.episodes = []
        self.episode_embeddings = []
        self.episode_index = 0
        
        # Indexing structures
        self.task_type_index = defaultdict(list)
        self.difficulty_index = defaultdict(list)
        self.success_index = defaultdict(list)
        
        # Text embedding
        self.text_embedder = TextEmbedding(config)
        self.similarity_threshold = 0.3
        
        # Performance tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'avg_similarity': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Load existing memory
        self._load_memory()
    
    @timing_decorator
    @error_handler(default_return=False, log_error=True)
    def store_episode(self, 
                     context: str, 
                     reasoning: str, 
                     outcome: str, 
                     success_rate: float,
                     task_type: str,
                     difficulty: float,
                     metadata: Optional[Dict] = None) -> bool:
        """Store a reasoning episode with indexing."""
        
        with self.lock:
            # Create episode
            episode = {
                'id': self.episode_index,
                'context': context,
                'reasoning': reasoning,
                'outcome': outcome,
                'success_rate': success_rate,
                'task_type': task_type,
                'difficulty': difficulty,
                'timestamp': time.time(),
                'usage_count': 0,
                'last_accessed': time.time(),
                'metadata': metadata or {}
            }
            
            # Get embedding
            combined_text = f"{context} {reasoning} {outcome}"
            embedding = self._get_episode_embedding(combined_text)
            
            # Check capacity and evict if necessary
            if len(self.episodes) >= self.max_size:
                self._evict_episode()
            
            # Store episode
            self.episodes.append(episode)
            self.episode_embeddings.append(embedding)
            
            # Update indices
            self._update_indices(episode, len(self.episodes) - 1)
            
            self.episode_index += 1
            
            logger.debug(f"Stored episode {episode['id']} for task type {task_type}")
            return True
    
    @timing_decorator
    @error_handler(default_return=[], log_error=True)
    def retrieve_similar_episodes(self, 
                                context: str, 
                                task_type: str,
                                top_k: Optional[int] = None,
                                min_similarity: float = None) -> List[Dict]:
        """Retrieve similar episodes with scoring."""
        
        if top_k is None:
            top_k = self.config.memory_retrieval_top_k
        
        if min_similarity is None:
            min_similarity = self.similarity_threshold
        
        with self.lock:
            if not self.episodes:
                return []
            
            # Get query embedding
            query_embedding = self._get_episode_embedding(context)
            
            # Filter by task type first
            candidate_indices = self.task_type_index.get(task_type, [])
            if not candidate_indices:
                # Fall back to all episodes
                candidate_indices = list(range(len(self.episodes)))
            
            # Compute similarities
            scored_episodes = []
            
            for idx in candidate_indices:
                if idx >= len(self.episodes):
                    continue
                
                episode = self.episodes[idx]
                episode_embedding = self.episode_embeddings[idx]
                
                # Compute similarity
                similarity = self._compute_similarity(query_embedding, episode_embedding)
                
                if similarity < min_similarity:
                    continue
                
                # Compute composite score
                score = self._compute_episode_score(episode, similarity)
                
                scored_episodes.append((score, similarity, episode))
            
            # Sort by score and select top-k
            scored_episodes.sort(key=lambda x: x[0], reverse=True)
            retrieved_episodes = []
            
            for score, similarity, episode in scored_episodes[:top_k]:
                # Update usage statistics
                episode['usage_count'] += 1
                episode['last_accessed'] = time.time()
                
                # Add retrieval metadata
                retrieved_episode = episode.copy()
                retrieved_episode['retrieval_score'] = score
                retrieved_episode['retrieval_similarity'] = similarity
                
                retrieved_episodes.append(retrieved_episode)
            
            # Update retrieval statistics
            self._update_retrieval_stats(len(retrieved_episodes), scored_episodes)
            
            logger.debug(f"Retrieved {len(retrieved_episodes)} episodes for {task_type}")
            return retrieved_episodes
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            if not self.episodes:
                return {'total_episodes': 0}
            
            # Basic stats
            total_episodes = len(self.episodes)
            task_type_counts = {}
            for ep in self.episodes:
                task_type = ep['task_type']
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            # Success rate distribution
            success_rates = [ep['success_rate'] for ep in self.episodes]
            avg_success_rate = np.mean(success_rates) if success_rates else 0.0
            
            # Usage statistics
            usage_counts = [ep['usage_count'] for ep in self.episodes]
            avg_usage = np.mean(usage_counts) if usage_counts else 0.0
            
            return {
                'total_episodes': total_episodes,
                'task_type_distribution': task_type_counts,
                'average_success_rate': avg_success_rate,
                'average_usage_count': avg_usage,
                'retrieval_stats': self.retrieval_stats.copy(),
                'memory_utilization': total_episodes / self.max_size
            }
    
    def _get_episode_embedding(self, text: str) -> np.ndarray:
        """Get embedding for episode text."""
        return self.text_embedder.get_embeddings([text], method='bert')[0]
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _compute_episode_score(self, episode: Dict, similarity: float) -> float:
        """Compute composite score for episode ranking."""
        # Base similarity score
        score = similarity
        
        # Recency bonus (exponential decay)
        age_hours = (time.time() - episode['timestamp']) / 3600
        recency_bonus = np.exp(-age_hours / 24) * 0.2  # 24-hour half-life
        
        # Success rate bonus
        success_bonus = episode['success_rate'] * 0.3
        
        # Usage frequency penalty (avoid over-used episodes)
        usage_penalty = min(episode['usage_count'] * 0.05, 0.2)
        
        return score + recency_bonus + success_bonus - usage_penalty
    
    def _update_indices(self, episode: Dict, episode_idx: int):
        """Update all indexing structures."""
        # Task type index
        self.task_type_index[episode['task_type']].append(episode_idx)
        
        # Difficulty index (binned)
        difficulty_bin = int(episode['difficulty'] * 10)  # 0-9 bins
        self.difficulty_index[difficulty_bin].append(episode_idx)
        
        # Success index (binned)
        success_bin = int(episode['success_rate'] * 10)  # 0-10 bins
        self.success_index[success_bin].append(episode_idx)
    
    def _evict_episode(self):
        """Evict least useful episode to make space."""
        if not self.episodes:
            return
        
        # Score episodes for eviction (lower score = more likely to evict)
        eviction_scores = []
        current_time = time.time()
        
        for i, episode in enumerate(self.episodes):
            # Age penalty (older episodes more likely to be evicted)
            age_penalty = (current_time - episode['timestamp']) / 3600
            
            # Usage bonus (frequently used episodes less likely to be evicted)
            usage_bonus = episode['usage_count'] * 10
            
            # Success bonus
            success_bonus = episode['success_rate'] * 20
            
            # Recent access bonus
            last_access_bonus = max(0, 24 - (current_time - episode['last_accessed']) / 3600)
            
            score = usage_bonus + success_bonus + last_access_bonus - age_penalty
            eviction_scores.append((score, i))
        
        # Sort by score and evict lowest
        eviction_scores.sort()
        evict_idx = eviction_scores[0][1]
        
        # Remove from all structures
        evicted_episode = self.episodes.pop(evict_idx)
        self.episode_embeddings.pop(evict_idx)
        
        # Rebuild indices (expensive but necessary)
        self._rebuild_indices()
        
        logger.debug(f"Evicted episode {evicted_episode['id']} from memory")
    
    def _rebuild_indices(self):
        """Rebuild all indices after eviction."""
        self.task_type_index.clear()
        self.difficulty_index.clear()
        self.success_index.clear()
        
        for i, episode in enumerate(self.episodes):
            self._update_indices(episode, i)
    
    def _update_retrieval_stats(self, num_retrieved: int, all_scored: List):
        """Update retrieval statistics."""
        self.retrieval_stats['total_retrievals'] += 1
        
        if num_retrieved > 0:
            self.retrieval_stats['successful_retrievals'] += 1
            
            # Update average similarity
            similarities = [score[1] for score in all_scored[:num_retrieved]]
            if similarities:
                current_avg = self.retrieval_stats['avg_similarity']
                new_avg = np.mean(similarities)
                
                # Exponential moving average
                alpha = 0.1
                self.retrieval_stats['avg_similarity'] = alpha * new_avg + (1 - alpha) * current_avg
    
    @error_handler(log_error=True)
    def _save_memory(self):
        """Save memory to disk."""
        save_path = Path(self.config.memory_cache_dir) / "episodic_memory.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        memory_data = {
            'episodes': self.episodes,
            'episode_embeddings': self.episode_embeddings,
            'episode_index': self.episode_index,
            'retrieval_stats': self.retrieval_stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(memory_data, f)
        
        logger.info(f"Saved {len(self.episodes)} episodes to {save_path}")
    
    @error_handler(log_error=True)
    def _load_memory(self):
        """Load memory from disk."""
        save_path = Path(self.config.memory_cache_dir) / "episodic_memory.pkl"
        
        if not save_path.exists():
            logger.info("No existing episodic memory found")
            return
        
        with open(save_path, 'rb') as f:
            memory_data = pickle.load(f)
        
        self.episodes = memory_data.get('episodes', [])
        self.episode_embeddings = memory_data.get('episode_embeddings', [])
        self.episode_index = memory_data.get('episode_index', 0)
        self.retrieval_stats = memory_data.get('retrieval_stats', {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'avg_similarity': 0.0
        })
        
        # Rebuild indices
        self._rebuild_indices()
        
        logger.info(f"Loaded {len(self.episodes)} episodes from {save_path}") 