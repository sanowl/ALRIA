"""
Multimodal Model - Main model class.
Multimodal model with memory systems and robust architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from ..config import ModelConfig
from ..memory import EpisodicMemoryBank, WorkingMemoryBuffer
from ..utils.decorators import timing_decorator, error_handler

import logging
logger = logging.getLogger(__name__)


class MultimodalModel(nn.Module):
    """
    Multimodal model with memory systems.
    
    This is the main model class that includes:
    - Text and vision processing
    - Memory systems
    - Generation capabilities
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory systems
        self.episodic_memory = EpisodicMemoryBank(config)
        self.working_memory = WorkingMemoryBuffer(config)
        
        # Model components will be initialized based on configuration
        self._initialize_model_components()
        
        # Training state
        self.training_step = 0
        self.evaluation_mode = False
        
        logger.info("Multimodal model initialized successfully")
    
    @error_handler(log_error=True)
    def _initialize_model_components(self):
        """Initialize all model components based on configuration."""
        
        # Core language model components
        self.embeddings = nn.Embedding(
            self.config.vocab_size, 
            self.config.hidden_size
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.config) 
            for _ in range(self.config.num_hidden_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Vision components (if enabled)
        if hasattr(self.config, 'enable_vision') and self.config.enable_vision:
            self.vision_encoder = VisionEncoder(self.config)
            self.vision_projection = nn.Linear(
                self.config.vision_hidden_size, 
                self.config.hidden_size
            )
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    @timing_decorator
    def forward(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               token_type_ids: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.Tensor] = None,
               images: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None,
               return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multimodal model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs  
            position_ids: Position IDs
            images: Image tensors
            labels: Labels for training
            return_dict: Whether to return dictionary
            
        Returns:
            Dictionary with model outputs
        """
        
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        token_embeddings = self.embeddings(input_ids)
        
        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Process images if provided
        if images is not None and hasattr(self, 'vision_encoder'):
            image_features = self.vision_encoder(images)
            image_features = self.vision_projection(image_features)
            # Concatenate with text embeddings (simplified)
            token_embeddings = torch.cat([token_embeddings, image_features], dim=1)
        
        # Apply dropout
        hidden_states = self.dropout(token_embeddings)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                attention_mask=attention_mask,
                use_cache=False
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'past_key_values': None,  # Not implemented in this simplified version
        }
    
    @torch.no_grad()
    def generate(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                do_sample: bool = True,
                **kwargs) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to sample
            
        Returns:
            Generated token IDs
        """
        
        if max_length is None:
            max_length = self.config.generation_max_length
        
        # Set model to eval mode
        self.eval()
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get next token logits
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=-1)
        
        return generated_ids
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1, None]
            logits = torch.where(logits < min_values, 
                               torch.full_like(logits, float('-inf')), 
                               logits)
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory system usage statistics."""
        return {
            'episodic_memory': self.episodic_memory.get_memory_statistics(),
            'working_memory': self.working_memory.get_memory_stats()
        }


class TransformerLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               use_cache: bool = False) -> torch.Tensor:
        """Forward pass through transformer layer."""
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + self.dropout(attn_output)
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, hidden_states: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through multi-head attention."""
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Get Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=2)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            # Convert attention mask to attention bias
            attention_mask = attention_mask[:, None, None, :].to(dtype=attn_weights.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        attn_output = self.c_proj(attn_output)
        
        return attn_output


class MLP(nn.Module):
    """MLP layer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class VisionEncoder(nn.Module):
    """Vision encoder for image processing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        # Simplified vision encoder - in practice would use CLIP or similar
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.vision_hidden_size)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder."""
        batch_size = images.shape[0]
        features = self.conv_layers(images)
        return features.unsqueeze(1)  # Add sequence dimension 