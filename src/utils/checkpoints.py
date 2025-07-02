"""
Checkpoint management utilities for SerenaNet.

This module provides utilities for saving and loading model checkpoints
with support for best model tracking and resuming training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import shutil
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic saving and loading.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        model (nn.Module): Model to checkpoint
        optimizer (optim.Optimizer, optional): Optimizer to checkpoint
        scheduler (optim.lr_scheduler.LRScheduler, optional): Scheduler to checkpoint
        max_checkpoints (int): Maximum number of checkpoints to keep
        save_best (bool): Whether to save best model separately
        monitor_metric (str): Metric to monitor for best model
        mode (str): 'min' or 'max' for best model monitoring
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Best model tracking
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        # Checkpoint tracking
        self.checkpoint_history = []
        
        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        filename: Optional[str] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint_data (Dict[str, Any]): Additional data to save
            filename (str, optional): Custom filename
            is_best (bool): Whether this is the best checkpoint
            
        Returns:
            str: Path to saved checkpoint
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': getattr(self.model, 'config', {}),
            **checkpoint_data
        }
        
        # Add optimizer state if available
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add metadata
        checkpoint['timestamp'] = datetime.now().isoformat()
        checkpoint['model_class'] = self.model.__class__.__name__
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'timestamp': checkpoint['timestamp'],
                'metrics': checkpoint_data.get('metrics', {}),
                'is_best': is_best
            })
            
            # Save best checkpoint separately
            if is_best and self.save_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy2(checkpoint_path, best_path)
                self.best_checkpoint_path = str(best_path)
                logger.info(f"Saved best model: {best_path}")
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            # Save checkpoint history
            self._save_checkpoint_history()
            
            return str(checkpoint_path)
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (Union[str, Path]): Path to checkpoint
            load_optimizer (bool): Whether to load optimizer state
            load_scheduler (bool): Whether to load scheduler state
            strict (bool): Whether to strictly enforce state dict loading
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint data or None
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return None
        
        try:
            # Load checkpoint to the same device it was saved from
            checkpoint_data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            
            # Load model state dict
            if self.model is not None:
                self.model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
                logger.info("Loaded model state from checkpoint")
            else:
                logger.warning("No model state found in checkpoint")
            
            # Load optimizer state
            if (load_optimizer and self.optimizer is not None and 
                'optimizer_state_dict' in checkpoint_data):
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
            
            # Load scheduler state
            if (load_scheduler and self.scheduler is not None and 
                'scheduler_state_dict' in checkpoint_data):
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info("Loaded scheduler state from checkpoint")
            
            logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_best_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """
        Load the best saved checkpoint.
        
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError("No best model checkpoint found")
        
        return self.load_checkpoint(best_path, **kwargs)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Optional[str]: Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest)
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            list: List of checkpoint information
        """
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path (Union[str, Path]): Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            
            # Update history
            self.checkpoint_history = [
                ckpt for ckpt in self.checkpoint_history
                if ckpt['path'] != str(checkpoint_path)
            ]
            self._save_checkpoint_history()
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        # Get non-best checkpoints
        regular_checkpoints = [
            ckpt for ckpt in self.checkpoint_history
            if not ckpt.get('is_best', False)
        ]
        
        # Sort by timestamp (oldest first)
        regular_checkpoints.sort(key=lambda x: x['timestamp'])
        
        # Remove excess checkpoints
        while len(regular_checkpoints) > self.max_checkpoints:
            old_checkpoint = regular_checkpoints.pop(0)
            old_path = Path(old_checkpoint['path'])
            
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Cleaned up old checkpoint: {old_path}")
        
        # Update history
        best_checkpoints = [
            ckpt for ckpt in self.checkpoint_history
            if ckpt.get('is_best', False)
        ]
        self.checkpoint_history = regular_checkpoints + best_checkpoints
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to JSON file."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.checkpoint_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint history: {e}")
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from JSON file."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r') as f:
                self.checkpoint_history = json.load(f)
            logger.debug("Loaded checkpoint history")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint history: {e}")
    
    def is_better(self, current_metric: float) -> bool:
        """
        Check if current metric is better than best metric.
        
        Args:
            current_metric (float): Current metric value
            
        Returns:
            bool: True if current metric is better
        """
        if self.mode == 'min':
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def update_best(self, current_metric: float):
        """
        Update best metric value.
        
        Args:
            current_metric (float): Current metric value
        """
        if self.is_better(current_metric):
            self.best_metric = current_metric
            logger.info(f"New best {self.monitor_metric}: {current_metric:.4f}")


def save_model_for_inference(
    model: nn.Module,
    save_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    phoneme_mapper: Optional[Any] = None,
    processor: Optional[Any] = None
):
    """
    Save model for inference with all necessary components.
    
    Args:
        model (nn.Module): Trained model
        save_path (Union[str, Path]): Path to save model
        config (Dict[str, Any], optional): Model configuration
        phoneme_mapper (optional): Phoneme mapper
        processor (optional): Audio processor
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if config is not None:
        save_dict['config'] = config
    
    if phoneme_mapper is not None:
        save_dict['phoneme_mapper'] = {
            'phonemes': phoneme_mapper.phonemes,
            'language': phoneme_mapper.language,
            'vocab_size': phoneme_mapper.vocab_size
        }
    
    if processor is not None:
        save_dict['processor_config'] = processor.get_config()
    
    # Save
    torch.save(save_dict, save_path)
    logger.info(f"Saved model for inference: {save_path}")


def load_model_for_inference(
    model_class: type,
    model_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> tuple:
    """
    Load model for inference.
    
    Args:
        model_class (type): Model class to instantiate
        model_path (Union[str, Path]): Path to saved model
        device (torch.device, optional): Device to load model on
        
    Returns:
        tuple: (model, config, phoneme_mapper, processor_config)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load saved data
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    saved_data = torch.load(model_path, map_location=device)
    
    # Create model
    config = saved_data.get('config', {})
    model = model_class(config=config)
    model.load_state_dict(saved_data['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract other components
    phoneme_mapper_data = saved_data.get('phoneme_mapper')
    processor_config = saved_data.get('processor_config')
    
    logger.info(f"Loaded model for inference from {model_path}")
    
    return model, config, phoneme_mapper_data, processor_config
