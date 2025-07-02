"""
SerenaNet: Main model implementation.

This module combines all components (ATHM, Transformer, CAR, Decoder, PESSL)
into the complete SerenaNet architecture for speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union

from .athm import ATHM
from .transformer import TransformerEncoder
from .car import CAR
from .decoder import Decoder
from .pessl import PESSL


class SerenaNet(nn.Module):
    """
    Complete SerenaNet architecture for speech recognition.
    
    SerenaNet combines:
    - ATHM: Multi-resolution temporal hierarchy
    - Transformer: Self-attention encoder
    - CAR: Mamba-based CTC alignment refinement
    - Decoder: Phoneme classification
    - PESSL: Self-supervised pre-training
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters
        phoneme_vocab_size (int): Size of phoneme vocabulary
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        phoneme_vocab_size: int = 41
    ):
        super(SerenaNet, self).__init__()
        
        # Set default config if not provided
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self.phoneme_vocab_size = phoneme_vocab_size
        
        # Extract component configurations
        athm_config = config.get('athm', {})
        transformer_config = config.get('transformer', {})
        car_config = config.get('car', {})
        pessl_config = config.get('pessl', {})
        
        # Initialize components based on ablation settings
        self.use_athm = config.get('use_athm', True)
        self.use_pessl = config.get('use_pessl', True)
        self.use_car = config.get('use_car', True)
        
        # ATHM module
        if self.use_athm:
            self.athm = ATHM(
                in_channels=athm_config.get('in_channels', 128),
                out_channels=athm_config.get('out_channels', 512),
                kernel_sizes=athm_config.get('kernel_sizes', [1, 2, 4]),
                strides=athm_config.get('strides', [1, 2, 4]),
                l2_lambda=athm_config.get('l2_lambda', 0.01)
            )
            transformer_input_dim = athm_config.get('out_channels', 512)
        else:
            # Direct input to transformer without ATHM
            self.input_proj = nn.Linear(128, transformer_config.get('d_model', 512))
            transformer_input_dim = transformer_config.get('d_model', 512)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_input_dim,
            nhead=transformer_config.get('nhead', 8),
            num_layers=transformer_config.get('num_layers', 6),
            dim_feedforward=transformer_config.get('dim_feedforward', 2048),
            dropout=transformer_config.get('dropout', 0.1),
            max_len=transformer_config.get('max_len', 1000)
        )
        
        # CAR module
        if self.use_car:
            self.car = CAR(
                input_dim=transformer_input_dim,
                phoneme_vocab_size=phoneme_vocab_size,
                mamba_hidden_dim=car_config.get('mamba_hidden_dim', 256),
                mamba_state_dim=car_config.get('mamba_state_dim', 16),
                l2_lambda=car_config.get('l2_lambda', 0.01)
            )
        
        # Decoder
        self.decoder = Decoder(
            input_dim=transformer_input_dim,
            output_dim=phoneme_vocab_size,
            num_layers=2,  # Use 2-layer decoder for better performance
            dropout=0.1
        )
        
        # PESSL module
        if self.use_pessl:
            self.pessl = PESSL(
                input_dim=transformer_input_dim,
                num_clusters=pessl_config.get('num_clusters', 100),
                proj_dim=pessl_config.get('proj_dim', 256),
                mask_prob=pessl_config.get('mask_prob', 0.15),
                mask_length=10,
                temperature=0.1
            )
        
        # Training mode tracking
        self.training_mode = 'pretrain'  # 'pretrain' or 'finetune'
        
        self._init_weights()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'athm': {
                'in_channels': 128,
                'out_channels': 512,
                'kernel_sizes': [1, 2, 4],
                'strides': [1, 2, 4],
                'l2_lambda': 0.01
            },
            'transformer': {
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'max_len': 1000
            },
            'car': {
                'mamba_hidden_dim': 256,
                'mamba_state_dim': 16,
                'l2_lambda': 0.01
            },
            'pessl': {
                'num_clusters': 100,
                'proj_dim': 256,
                'mask_prob': 0.15
            },
            'use_athm': True,
            'use_pessl': True,
            'use_car': True,
        }
    
    def _init_weights(self):
        """Initialize weights for any additional parameters."""
        if hasattr(self, 'input_proj'):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
    
    def set_training_mode(self, mode: str):
        """
        Set training mode.
        
        Args:
            mode (str): 'pretrain' or 'finetune'
        """
        if mode not in ['pretrain', 'finetune']:
            raise ValueError("Mode must be 'pretrain' or 'finetune'")
        self.training_mode = mode
    
    def forward(
        self, 
        x: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SerenaNet.
        
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, T, 128)
            targets (torch.Tensor, optional): Target phoneme sequences for CTC loss
            input_lengths (torch.Tensor, optional): Input sequence lengths
            target_lengths (torch.Tensor, optional): Target sequence lengths
            src_key_padding_mask (torch.Tensor, optional): Padding mask for transformer
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing outputs and losses
        """
        outputs = {}
        
        # ATHM processing
        if self.use_athm:
            athm_features, athm_l2_loss = self.athm(x)  # (B, T, 512)
            outputs['athm_l2_loss'] = athm_l2_loss
        else:
            athm_features = self.input_proj(x)  # (B, T, 512)
            outputs['athm_l2_loss'] = torch.tensor(0.0, device=x.device)
        
        # Transformer encoding
        transformer_features = self.transformer(
            athm_features, 
            src_key_padding_mask=src_key_padding_mask
        )  # (B, T, 512)
        
        outputs['athm_features'] = athm_features
        outputs['transformer_features'] = transformer_features
        
        # PESSL pre-training
        if self.use_pessl and self.training:
            if not self.pessl.is_fitted:
                outputs['pessl_loss'] = torch.tensor(0.0, device=x.device)
            else:
                masks = self.pessl.create_masks(x.size(0), x.size(1), x.device)
                masked_features = self.pessl.apply_masks(athm_features, masks)
                pessl_loss, perplexity = self.pessl(athm_features, masked_features, masks)
                outputs['pessl_loss'] = pessl_loss
                outputs['pessl_perplexity'] = perplexity
        
        # Main decoder branch (for fine-tuning and inference)
        decoder_logits = self.decoder(transformer_features)
        outputs['decoder_logits'] = decoder_logits
        
        # CAR branch (if used)
        if self.use_car:
            car_logits, car_l2_loss = self.car(transformer_features)
            outputs['car_logits'] = car_logits
            outputs['car_l2_loss'] = car_l2_loss
        else:
            outputs['car_l2_loss'] = torch.tensor(0.0, device=x.device)
        
        # --- Loss Computation ---
        if self.training and targets is not None:
            # CTC loss on main decoder branch
            if input_lengths is not None and target_lengths is not None:
                ctc_loss = F.ctc_loss(
                    log_probs=F.log_softmax(decoder_logits, dim=-1).transpose(0, 1),
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    blank=0,
                    reduction='mean'
                )
                outputs['ctc_loss'] = ctc_loss

            # CTC loss on CAR branch
            if self.use_car and 'car_logits' in outputs and input_lengths is not None and target_lengths is not None:
                car_ctc_loss = self.car.compute_ctc_loss(
                    outputs['car_logits'], targets, input_lengths, target_lengths
                )
                outputs['car_ctc_loss'] = car_ctc_loss
        
        return outputs
    
    def compute_total_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        lambda_pessl: float = 0.1,
        lambda_car: float = 0.1,
        lambda_ctc: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from output dictionary.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Outputs from forward pass
            lambda_pessl (float): Weight for PESSL loss
            lambda_car (float): Weight for CAR loss
            lambda_ctc (float): Weight for main CTC loss
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of all loss components
        """
        loss_dict = {}

        # Use a zero tensor that requires grad as the base for the total loss
        total_loss = torch.tensor(0.0, device=outputs['transformer_features'].device)
        if self.training:
            total_loss.requires_grad_()

        # Pre-training loss
        if 'pessl_loss' in outputs and self.training_mode == 'pretrain' and self.use_pessl:
            pessl_loss = outputs['pessl_loss']
            total_loss = total_loss + lambda_pessl * pessl_loss
            loss_dict['pessl_loss'] = pessl_loss
            
        # Fine-tuning losses
        if self.training_mode == 'finetune':
            # Main CTC loss
            if 'ctc_loss' in outputs:
                ctc_loss = outputs['ctc_loss']
                total_loss = total_loss + lambda_ctc * ctc_loss
                loss_dict['ctc_loss'] = ctc_loss
            
            # CAR CTC loss
            if 'car_ctc_loss' in outputs and self.use_car:
                car_loss = outputs['car_ctc_loss']
                total_loss = total_loss + lambda_car * car_loss
                loss_dict['car_loss'] = car_loss
            
            # L2 regularization losses
            if 'athm_l2_loss' in outputs and self.use_athm:
                total_loss = total_loss + outputs['athm_l2_loss']
            if 'car_l2_loss' in outputs and self.use_car:
                total_loss = total_loss + outputs['car_l2_loss']
        
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def decode(
        self,
        x: torch.Tensor,
        method: str = 'greedy',
        beam_width: int = 10
    ) -> Union[list, Tuple[list, list], torch.Tensor]:
        """
        Decode output probabilities to phoneme sequences.
        
        Args:
            x (torch.Tensor): Input spectrogram
            method (str): 'greedy' or 'beam'
            beam_width (int): Beam width for beam search
            
        Returns:
            Decoded phoneme sequences
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['decoder_logits']

        if method == 'greedy':
            return self.decoder.decode_greedy(logits)
        elif method == 'beam':
            return self.decoder.decode_beam(logits, beam_width=beam_width)
        else:
            raise ValueError(f"Unsupported decoding method: {method}")

    def fit_pessl_kmeans(self, dataloader: torch.utils.data.DataLoader, max_batches: int = 10):
        """
        Fit PESSL k-means clustering on a subset of the training data.
        
        Args:
            dataloader: Data loader for fitting
            max_batches: Maximum number of batches to use
        """
        if not self.use_pessl:
            return
        
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                # Unpack batch (assuming audio is first element)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(next(self.parameters()).device)
                
                # Get ATHM features
                if self.use_athm:
                    athm_features, _ = self.athm(x)
                else:
                    athm_features = self.input_proj(x)
                
                # Collect features
                all_features.append(athm_features.view(-1, athm_features.size(-1)))
        
        # Concatenate all features and fit k-means
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            self.pessl.fit_kmeans(all_features)
        
        self.train()
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters for each component."""
        param_counts = {}
        
        if self.use_athm:
            param_counts['athm'] = sum(p.numel() for p in self.athm.parameters())
        else:
            param_counts['athm'] = 0
        
        param_counts['transformer'] = sum(p.numel() for p in self.transformer.parameters())
        
        if self.use_car:
            param_counts['car'] = sum(p.numel() for p in self.car.parameters())
        else:
            param_counts['car'] = 0
        
        param_counts['decoder'] = sum(p.numel() for p in self.decoder.parameters())
        
        if self.use_pessl:
            param_counts['pessl'] = sum(p.numel() for p in self.pessl.parameters())
        else:
            param_counts['pessl'] = 0
        
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts
