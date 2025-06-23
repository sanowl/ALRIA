"""
Configuration module for Enhanced MMaDA.
Contains all configuration classes and default settings.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnhancedMMaDAConfig:
    """Comprehensive configuration with all features enabled."""
    
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
    
    # Diffusion and multimodal
    num_diffusion_steps: int = 1000
    mask_token_id: int = 32001
    image_token_start: int = 32002
    image_vocab_size: int = 8192
    image_resolution: int = 512
    patch_size: int = 16
    vision_hidden_size: int = 1024
    
    # Training configuration
    mixed_cot_prob: float = 0.8
    unigrpo_clip_eps: float = 0.2
    unigrpo_kl_beta: float = 0.01
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Reward and objective weights
    correctness_reward: float = 2.0
    format_reward: float = 0.5
    clip_reward_scale: float = 0.1
    image_reward_scale: float = 0.1
    accuracy_weight: float = 0.7
    speed_weight: float = 0.2
    safety_weight: float = 0.1
    
    # Generation parameters
    generation_max_length: int = 1024
    generation_temperature: float = 1.0
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Training stages
    stage1_epochs: int = 3  # Basic training
    stage2_epochs: int = 2  # Advanced features
    stage3_epochs: int = 1  # Fine-tuning
    
    # Feature enables
    enable_adaptive_reasoning: bool = True
    enable_episodic_memory: bool = True
    enable_uncertainty_estimation: bool = True
    enable_cross_modal_verification: bool = True
    enable_speculative_decoding: bool = True
    enable_modular_generation: bool = True
    enable_meta_cognition: bool = True
    enable_domain_adaptation: bool = True
    enable_performance_monitoring: bool = True
    enable_online_learning: bool = True
    
    # Adaptive reasoning
    reasoning_depth_threshold_high: float = 0.8
    reasoning_depth_threshold_low: float = 0.3
    confidence_threshold: float = 0.6
    complexity_estimation_samples: int = 100
    
    # Memory systems
    episodic_memory_size: int = 10000
    working_memory_size: int = 100
    memory_retrieval_top_k: int = 5
    memory_embedding_dim: int = 768
    memory_decay_factor: float = 0.95
    
    # Uncertainty estimation
    uncertainty_num_samples: int = 10
    confidence_calibration_temp: float = 1.5
    abstention_threshold: float = 0.7
    monte_carlo_samples: int = 5
    
    # Cross-modal verification
    clip_similarity_threshold: float = 0.7
    verification_confidence_threshold: float = 0.8
    max_verification_attempts: int = 3
    
    # Speculative decoding
    draft_model_layers: int = 12
    speculation_lookahead: int = 4
    acceptance_threshold: float = 0.8
    
    # Modular generation
    max_subproblems: int = 10
    synthesis_temperature: float = 0.8
    component_timeout: int = 30
    
    # Meta-cognition
    self_reflection_threshold: float = 0.6
    improvement_tracking_window: int = 50
    meta_learning_rate: float = 1e-4
    
    # Domain adaptation
    num_domains: int = 8
    domain_adapter_rank: int = 16
    domain_detection_threshold: float = 0.5
    adaptation_strength: float = 0.1
    
    # Performance monitoring
    performance_window_size: int = 100
    performance_alert_threshold: float = 0.3
    resource_check_interval: int = 10
    
    # Model paths
    text_tokenizer_path: str = "microsoft/DialoGPT-medium"
    vision_model_path: str = "openai/clip-vit-large-patch14"
    nli_model_path: str = "microsoft/DialoGPT-medium"
    
    # Storage paths
    save_dir: str = "./enhanced_mmada_checkpoints"
    memory_cache_dir: str = "./memory_cache"
    logs_dir: str = "./training_logs"
    
    # Advanced features
    enable_curriculum_learning: bool = True
    enable_few_shot_learning: bool = True
    enable_active_learning: bool = True
    curriculum_difficulty_steps: int = 5
    few_shot_examples: int = 3
    active_learning_threshold: float = 0.4


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
    wandb_project: str = "enhanced_mmada"


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
    
    # Quality requirements
    quality_threshold: float = 0.8
    max_attempts: int = 3
    
    # Performance settings
    batch_size: int = 1
    use_cache: bool = True
    low_memory_mode: bool = False 