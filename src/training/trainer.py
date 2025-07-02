"""
Main training script for SerenaNet.

This script handles both pre-training and fine-tuning phases of SerenaNet
with comprehensive logging, checkpointing, and evaluation.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm
import time
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.serenanet import SerenaNet
from data.preprocessing import SpectrogramProcessor
from data.augmentation import create_training_augmentation, create_light_augmentation
from data.datasets import create_datasets, create_dataloaders
from data.phonemes import PhonemeMapper
from evaluation.metrics import compute_wer_cer, decode_ctc
from utils.config import load_config
from utils.logger import setup_logging
from utils.checkpoints import CheckpointManager
from models.decoder import GreedyDecoder, BeamSearchDecoder

logger = logging.getLogger(__name__)


class SerenaTrainer:
    """
    Main trainer class for SerenaNet.
    
    Handles both pre-training and fine-tuning phases with comprehensive
    logging, checkpointing, and evaluation capabilities.
    
    Args:
        config (Dict[str, Any]): Training configuration
        device (torch.device): Training device
        experiment_name (str): Name for logging experiments
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        experiment_name: str = "serenanet"
    ):
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Initialize components in the correct order
        self._setup_data_first()
        self._setup_model()
        self._setup_dataloaders()
        self._setup_training()
        self._setup_logging()
        self._setup_checkpointing()
        
        # Log model summary after everything is set up
        self._log_model_summary()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Setup the decoder
        self._setup_decoder()
        
    def _setup_data_first(self):
        """Setup data processing components that must exist before the model."""
        logger.info("Setting up data processing (pre-model)...")
        # Phoneme mapper
        phoneme_config = self.config.get('phoneme_mapping', {})
        self.phoneme_mapper = PhonemeMapper(**phoneme_config)
        
        # Audio processor
        self.processor = SpectrogramProcessor(
            **self.config.get('data', {}).get('preprocessing', {})
        )
        
        # Augmentation
        if self.config['training'].get('mode') == 'pretrain':
            self.augmentation = create_training_augmentation(**self.config.get('augmentation', {}))
        else:
            self.augmentation = create_light_augmentation()

    def _setup_model(self):
        """Initialize model and move to device."""
        logger.info("Initializing SerenaNet model...")
        
        self.model = SerenaNet(
            config=self.config.get('model', {}),
            phoneme_vocab_size=self.phoneme_mapper.get_vocab_size()
        )
        
        self.model.to(self.device)
        
        # Log model info
        param_counts = self.model.get_num_parameters()
        logger.info(f"Model parameters: {param_counts}")
        
        # Set training mode
        training_mode = self.config['training'].get('mode', 'pretrain')
        self.model.set_training_mode(training_mode)
        logger.info(f"Set model to {training_mode} mode")
    
    def _setup_decoder(self):
        """Initialize the CTC decoder based on config."""
        decoder_config = self.config.get('decoder', {})
        decoder_type = decoder_config.get('type', 'greedy').lower()
        logger.info(f"Setting up '{decoder_type}' decoder.")

        if decoder_type == 'greedy':
            self.decoder = GreedyDecoder(
                phoneme_mapper=self.phoneme_mapper,
                blank_id=self.phoneme_mapper.blank_idx
            )
        elif decoder_type == 'beam':
            self.decoder = BeamSearchDecoder(
                phoneme_mapper=self.phoneme_mapper,
                beam_width=decoder_config.get('beam_width', 100),
                lm_type=decoder_config.get('lm_type', None),
                lm_path=decoder_config.get('lm_path', None),
                alpha=decoder_config.get('alpha', 0.8),
                beta=decoder_config.get('beta', 1.2),
                blank_id=self.phoneme_mapper.blank_idx
            )
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
    
    def _log_model_summary(self):
        """Log model architecture and parameter counts to TensorBoard."""
        model_summary = str(self.model)
        
        param_counts = self.model.get_num_parameters()
        total_params = param_counts.pop('total')
        
        summary_text = (
            f"Model Architecture:\n\n{model_summary}\n\n"
            f"Total Parameters: {total_params:,}\n\n"
            "Parameter Breakdown:\n"
        )
        
        for name, count in param_counts.items():
            summary_text += f"- {name}: {count:,}\n"
            
        self.writer.add_text("model/summary", summary_text, 0)
        logger.info("Logged model summary to TensorBoard.")
    
    def _setup_dataloaders(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        # Datasets and dataloaders
        data_cfg = self.config.get('data', {})
        if 'datasets' not in data_cfg:
            # backwards compatibility for older configs
            data_cfg = {
                'datasets': {
                    'train': {
                        'type': 'common_voice',
                        'root_dir': '.',
                        'manifest_file': data_cfg.get('train_manifest', 'dummy_train.json')
                    },
                    'val': {
                        'type': 'common_voice',
                        'root_dir': '.',
                        'manifest_file': data_cfg.get('val_manifest', 'dummy_val.json')
                    }
                }
            }

        self.datasets = create_datasets(
            config=data_cfg,
            processor=self.processor,
            phoneme_mapper=self.phoneme_mapper,
            augmentation=self.augmentation
        )
        
        self.dataloaders = create_dataloaders(
            datasets=self.datasets,
            config=self.config['training']
        )
        
        logger.info(f"Created datasets: {list(self.datasets.keys())}")
        for name, dataset in self.datasets.items():
            logger.info(f"  {name}: {len(dataset)} samples")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss functions."""
        logger.info("Setting up training components...")
        
        # Gradient accumulation steps
        self.accum_steps = self.config['training'].get('accum_steps', 1)

        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        if optimizer_config['type'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config['training']['learning_rate']),
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=float(optimizer_config.get('eps', 1e-8)),
                weight_decay=float(self.config['training'].get('weight_decay', 1e-4))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
        
        # AMP
        if self.device.type == 'cuda':
            self.scaler: Optional[torch.cuda.amp.GradScaler] = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        # Scheduler
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler: Optional[_LRScheduler] = None
        if scheduler_config.get('type') == 'linear_warmup':
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR( # type: ignore
                self.optimizer,
                start_factor=scheduler_config.get('start_factor', 0.1),
                total_iters=scheduler_config.get('warmup_steps', 10)
            )
        elif scheduler_config.get('type') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR( # type: ignore
                self.optimizer,
                T_max=self.config['training'].get('num_epochs', 10) * len(self.dataloaders['train']),
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
            )
        self.scheduler = scheduler
        
        # Loss weights
        self.lambda_pessl = self.config['training'].get('lambda_pessl', 0.1)
        self.lambda_car = self.config['training'].get('lambda_car', 0.1)
        
        # Gradient clipping
        self.gradient_clip = self.config['training'].get('gradient_clip', 1.0)
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        # TensorBoard
        log_dir = Path(self.config['logging']['log_dir']) / self.experiment_name
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Weights & Biases
        if self.config['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['logging'].get('project_name', 'serenanet'),
                name=self.experiment_name,
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        logger.info(f"Logging to {log_dir}")
    
    def _setup_checkpointing(self):
        """Setup checkpoint management."""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
    
    def fit_pessl_kmeans(self):
        """Fit k-means clustering for PESSL if in pre-training mode."""
        if (self.model.training_mode == 'pretrain' and 
            self.model.use_pessl and 
            not self.model.pessl.is_fitted):
            
            logger.info("Fitting k-means for PESSL...")
            self.model.fit_pessl_kmeans(
                dataloader=self.dataloaders['train'],
                max_batches=10
            )
            logger.info("K-means fitting completed!")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'ctc_loss': 0.0,
            'pessl_loss': 0.0,
            'car_loss': 0.0,
            'l2_loss': 0.0
        }
        
        num_batches = len(self.dataloaders['train'])
        progress_bar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.epoch+1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            spectrograms, targets, input_lengths, target_lengths, metadata = batch
            spectrograms = spectrograms.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                outputs = self.model(
                    x=spectrograms,
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths
                )

                loss_dict = self.model.compute_total_loss(
                    outputs=outputs,
                    lambda_pessl=self.lambda_pessl,
                    lambda_car=self.lambda_car
                )
                total_loss = loss_dict['total_loss'] / self.accum_steps

            # Backward pass with scaler
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            with torch.no_grad():
                epoch_metrics['loss'] += total_loss.item()
                epoch_metrics['ctc_loss'] += loss_dict.get('ctc_loss', torch.tensor(0.0)).item()
                epoch_metrics['pessl_loss'] += loss_dict.get('pessl_loss', torch.tensor(0.0)).item()
                epoch_metrics['car_loss'] += loss_dict.get('car_loss', torch.tensor(0.0)).item()
                epoch_metrics['l2_loss'] += (
                    outputs.get('athm_l2_loss', torch.tensor(0.0)).item() +
                    outputs.get('car_l2_loss', torch.tensor(0.0)).item()
                )
            
            # Log batch metrics
            self.global_step += 1
            save_interval = self.config['logging'].get(
                'save_interval',
                self.config.get('training', {}).get('save_every_n_steps', 100)
            )
            if save_interval and self.global_step % save_interval == 0:
                self._log_metrics({
                    'batch_loss': total_loss.item(), # type: ignore
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step, prefix='train')
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}", # type: ignore
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        val_metrics = {'loss': 0.0, 'wer': 0.0, 'cer': 0.0}
        all_predictions, all_targets = [], []
        
        with torch.no_grad():
            for batch_idx, (spectrograms, targets, input_lengths, target_lengths, metadata) in enumerate(
                tqdm(self.dataloaders['val'], desc="Validating")
            ):
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    x=spectrograms,
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths
                )
                
                # Compute loss
                loss_dict = self.model.compute_total_loss(
                    outputs=outputs,
                    lambda_pessl=self.lambda_pessl,
                    lambda_car=self.lambda_car
                )
                total_loss = loss_dict['total_loss']
                
                val_metrics['loss'] += total_loss.item()
                
                # Collect predictions for WER/CER calculation
                if 'decoder_logits' in outputs:
                    
                    # Use the configured decoder to get transcriptions
                    decoded_transcriptions = self.decoder.decode(outputs['decoder_logits'])
                    
                    # --- BATCH-LEVEL DEBUG (prints for first batch only) ---
                    if batch_idx == 0:
                        print("\n" + "="*80)
                        print(f"VALIDATION DEBUG: Batch {batch_idx}")
                        print("="*80)

                    # Process batch
                    for i, pred_text in enumerate(decoded_transcriptions):
                        target_seq = targets[i, :target_lengths[i]].cpu().numpy()
                        target_text = self.phoneme_mapper.indices_to_text(target_seq.tolist())
                        
                        # --- SAMPLE-LEVEL DEBUG (prints first 4 samples of the first batch) ---
                        if batch_idx == 0 and i < 4:
                            original_text = metadata[i].get('text', '[NO ORIGINAL TEXT]')
                            print(f"\n--- Sample {i} ---")
                            print(f"  GROUND TRUTH: {repr(original_text)}")
                            print(f"  TARGET (PHONEMES): {repr(target_text)}")
                            print(f"  PREDICTED: {repr(pred_text)}")

                        all_predictions.append(pred_text)
                        all_targets.append(target_text)

                    if batch_idx == 0:
                        print("="*80 + "\n")
        
        # Calculate WER and CER
        if all_predictions and all_targets:
            # Debug logging to see what we're actually calculating WER on
            logger.info(f"WER calculation with {len(all_predictions)} predictions and {len(all_targets)} targets")
            logger.debug(f"Sample predictions: {all_predictions[:3]}")
            logger.debug(f"Sample targets: {all_targets[:3]}")
            
            wer, cer = compute_wer_cer(all_predictions, all_targets)
            val_metrics['wer'] = wer
            val_metrics['cer'] = cer
            
            logger.info(f"Computed WER: {wer:.4f}, CER: {cer:.4f}")
        else:
            logger.warning(f"No predictions/targets for WER: preds={len(all_predictions)}, targets={len(all_targets)}")
            val_metrics['wer'] = 0.0
            val_metrics['cer'] = 0.0
        
        # Average loss
        val_metrics['loss'] /= len(self.dataloaders['val'])
        
        return val_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to TensorBoard and W&B."""
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            
            # TensorBoard
            self.writer.add_scalar(metric_name, value, step)
            
            # Weights & Biases
            if self.use_wandb:
                wandb.log({metric_name: value}, step=step)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save training state."""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config,
            'metrics': metrics
        }
        self.checkpoint_manager.save_checkpoint(state, is_best=is_best)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training state."""
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        if state:
            self.epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_metric = state['best_metric']
            logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {self.epoch}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Fit k-means if needed
        self.fit_pessl_kmeans()
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            all_metrics = {**{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_metrics:
                all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            self._log_metrics(all_metrics, step=self.global_step, prefix='epoch')
            
            # Check if best model
            monitor_metric = self.config.get('checkpoint', {}).get('monitor', 'val_loss')
            current_metric = all_metrics.get(monitor_metric, float('inf'))
            is_best = current_metric < self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                logger.info(f"New best {monitor_metric}: {current_metric:.4f}")
            
            # Save checkpoint
            if epoch % self.config['logging'].get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(all_metrics, is_best=is_best)
            
            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics.get('loss', 0):.4f} - "
                f"Val WER: {val_metrics.get('wer', 0):.4f} - "
                f"Time: {elapsed_time/3600:.1f}h"
            )
        
        logger.info("Training completed!")
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()

    # ------------------------------------------------------------------
    # Helper to proxy training mode to underlying model (used by tests)
    # ------------------------------------------------------------------
    def set_training_mode(self, mode: str):
        """Proxy helper so tests can switch between pretrain / finetune."""
        self.model.set_training_mode(mode)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train SerenaNet")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"serenanet_{timestamp}"
    
    # Create trainer
    trainer = SerenaTrainer(
        config=config,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
