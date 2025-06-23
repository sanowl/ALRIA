"""
Model Trainer class.
Handles training logic with robust error handling and monitoring.
"""

import os
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from ..config import ModelConfig, TrainingConfig
from ..models import MultimodalModel
from ..utils.decorators import timing_decorator, error_handler

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for the multimodal model.
    
    Features:
    - Robust error handling
    - Automatic checkpointing
    - Performance monitoring
    - Mixed precision training
    - Early stopping
    """
    
    def __init__(self, 
                 model: MultimodalModel,
                 config: ModelConfig,
                 training_config: Optional[TrainingConfig] = None):
        
        self.model = model
        self.config = config
        self.training_config = training_config or TrainingConfig()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.training_config.use_mixed_precision else None
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'step_losses': [],
            'learning_rates': [],
            'training_times': []
        }
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Model Trainer initialized successfully")
    
    @error_handler(log_error=True)
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters for different components
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.training_config.warmup_steps
        )
    
    @error_handler(log_error=True)
    def _setup_logging(self):
        """Setup training logging."""
        log_dir = Path(self.config.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = log_dir / f"training_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Training logs will be saved to {log_file}")
    
    @timing_decorator
    def train(self, 
              train_dataloader: DataLoader,
              eval_dataloader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop with error handling.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            num_epochs: Number of epochs (optional, uses config default)
            
        Returns:
            Training statistics and results
        """
        
        if num_epochs is None:
            num_epochs = self.training_config.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            try:
                # Training phase
                train_loss = self._train_epoch(train_dataloader)
                
                # Evaluation phase
                eval_loss = None
                if eval_dataloader is not None:
                    eval_loss = self._evaluate_epoch(eval_dataloader)
                
                # Learning rate scheduling
                if self.global_step < self.training_config.warmup_steps:
                    self.scheduler.step()
                
                # Record statistics
                epoch_time = time.time() - epoch_start_time
                self.training_stats['epoch_losses'].append(train_loss)
                self.training_stats['training_times'].append(epoch_time)
                self.training_stats['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Eval Loss: {eval_loss:.4f if eval_loss else 'N/A'} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Checkpointing
                if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, train_loss, eval_loss)
                
                # Early stopping
                if eval_loss is not None:
                    if eval_loss < self.best_loss - self.training_config.early_stopping_threshold:
                        self.best_loss = eval_loss
                        self.early_stopping_counter = 0
                        # Save best model
                        self._save_checkpoint(epoch, train_loss, eval_loss, is_best=True)
                    else:
                        self.early_stopping_counter += 1
                        
                    if self.early_stopping_counter >= self.training_config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
            except Exception as e:
                logger.error(f"Error during epoch {epoch+1}: {e}")
                # Save emergency checkpoint
                self._save_checkpoint(epoch, float('inf'), float('inf'), is_emergency=True)
                raise
        
        # Final cleanup
        self._finalize_training()
        
        return {
            'training_stats': self.training_stats,
            'final_epoch': self.current_epoch,
            'final_step': self.global_step,
            'best_loss': self.best_loss
        }
    
    @timing_decorator
    @error_handler(default_return=float('inf'), log_error=True)
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.training_config.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                # Handle gradient accumulation
                loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                if self.training_config.use_mixed_precision and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.training_config.use_mixed_precision and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Record loss
                total_loss += loss.item() * self.training_config.gradient_accumulation_steps
                num_batches += 1
                
                # Logging
                if (batch_idx + 1) % self.training_config.log_every_n_steps == 0:
                    current_loss = total_loss / num_batches
                    logger.debug(
                        f"Step {self.global_step} - "
                        f"Batch {batch_idx+1}/{len(dataloader)} - "
                        f"Loss: {current_loss:.4f}"
                    )
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    @timing_decorator
    @error_handler(default_return=float('inf'), log_error=True)
    def _evaluate_epoch(self, dataloader: DataLoader) -> float:
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.training_config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    @error_handler(log_error=True)
    def _save_checkpoint(self, 
                        epoch: int, 
                        train_loss: float, 
                        eval_loss: Optional[float],
                        is_best: bool = False,
                        is_emergency: bool = False):
        """Save model checkpoint."""
        
        checkpoint_dir = Path(self.config.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
            'config': self.config,
            'training_config': self.training_config
        }
        
        # Save checkpoint
        if is_emergency:
            checkpoint_path = checkpoint_dir / f"emergency_checkpoint_epoch_{epoch}.pt"
        elif is_best:
            checkpoint_path = checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Clean up old checkpoints
        if not is_best and not is_emergency:
            self._cleanup_old_checkpoints(checkpoint_dir)
    
    @error_handler(log_error=True)
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints to save space."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > self.training_config.keep_last_n_checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.training_config.keep_last_n_checkpoints]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    @error_handler(log_error=True)
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model from checkpoint."""
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            self.training_stats = checkpoint['training_stats']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def _finalize_training(self):
        """Finalize training process."""
        logger.info("Training completed successfully")
        
        # Save final statistics
        stats_path = Path(self.config.save_dir) / "training_stats.json"
        try:
            import json
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            logger.info(f"Training statistics saved to {stats_path}")
        except Exception as e:
            logger.error(f"Failed to save training statistics: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'device': str(self.device),
            'memory_usage': self.model.get_memory_usage()
        } 