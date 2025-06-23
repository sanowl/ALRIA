"""
Configuration module for the multimodal model.
Contains model, training, and inference settings.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Main model configuration."""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.1
    
    # Vision settings
    image_token_start: int = 32002
    image_vocab_size: int = 8192
    image_resolution: int = 512
    patch_size: int = 16
    vision_hidden_size: int = 1024
    
    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Generation settings
    generation_max_length: int = 1024
    generation_temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Memory settings
    episodic_memory_size: int = 10000
    working_memory_size: int = 100
    memory_retrieval_top_k: int = 5
    memory_embedding_dim: int = 768
    memory_decay_factor: float = 0.95
    
    # Paths
    text_tokenizer_path: str = "microsoft/DialoGPT-medium"
    vision_model_path: str = "openai/clip-vit-large-patch14"
    save_dir: str = "./model_checkpoints"
    memory_cache_dir: str = "./memory_cache"
    logs_dir: str = "./training_logs"


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    
    # Basic training settings
    num_epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    fp16: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Validation
    eval_every_n_steps: int = 500
    eval_batch_size: int = 16
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "multimodal_model"


@dataclass 
class InferenceConfig:
    """Inference-specific configuration."""
    
    # Generation settings
    max_length: int = 1024
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Quality settings
    quality_threshold: float = 0.8
    max_attempts: int = 3
    
    # Performance settings
    batch_size: int = 1
    use_cache: bool = True
    low_memory_mode: bool = False 