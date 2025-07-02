"""
CTC Alignment Refinement (CAR) module for SerenaNet.

This module implements the CAR component using Mamba SSM for improved
CTC alignment and phoneme probability estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .mamba_official import MambaOfficialBlock


class CAR(nn.Module):
    """
    CTC Alignment Refinement module using Mamba SSM.
    
    This module takes transformer outputs and refines them using a state-space
    model to produce better phoneme probability distributions for CTC alignment.
    
    Args:
        input_dim (int): Input feature dimension from transformer
        phoneme_vocab_size (int): Output vocabulary size (number of phonemes)
        mamba_hidden_dim (int): Hidden dimension for Mamba SSM
        mamba_state_dim (int): State space dimension for SSM
        num_layers (int): Number of Mamba blocks
        l2_lambda (float): L2 regularization weight
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        phoneme_vocab_size: int,
        mamba_hidden_dim: int = 256,
        mamba_state_dim: int = 16,
        num_layers: int = 2,
        l2_lambda: float = 0.01,
        dropout: float = 0.1
    ):
        super(CAR, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = mamba_hidden_dim
        self.output_dim = phoneme_vocab_size
        self.l2_lambda = l2_lambda
        self.num_layers = num_layers
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        
        # Stack of Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaOfficialBlock(
                dim=self.hidden_dim,
                state_dim=mamba_state_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection and classification head
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CAR module.
        
        Args:
            x (torch.Tensor): Input tensor from transformer of shape (B, T, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Raw logits of shape (B, T, output_dim)
                - L2 regularization loss
        """
        # Input projection and normalization
        h = self.input_proj(x)  # (B, T, hidden_dim)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Pass through Mamba blocks
        for mamba_block in self.mamba_blocks:
            h = mamba_block(h)  # (B, T, hidden_dim)
        
        # Output projection
        h = self.output_proj(h)  # (B, T, hidden_dim)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Classification logits
        logits = self.classifier(h)
        
        # Compute L2 regularization loss on hidden states
        l2_loss = self.l2_lambda * torch.norm(h, p=2, dim=-1).mean()
        
        return logits, l2_loss
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits without softmax for CTC loss computation.
        
        Args:
            x (torch.Tensor): Input tensor from transformer
            
        Returns:
            torch.Tensor: Raw logits of shape (B, T, output_dim)
        """
        # Input projection and normalization
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Pass through Mamba blocks
        for mamba_block in self.mamba_blocks:
            h = mamba_block(h)
        
        # Output projection
        h = self.output_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Classification logits
        logits = self.classifier(h)
        
        return logits
    
    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits (torch.Tensor): Logits from CAR of shape (B, T, V)
            targets (torch.Tensor): Target phoneme sequences of shape (B, L)
            input_lengths (torch.Tensor): Lengths of input sequences of shape (B,)
            target_lengths (torch.Tensor): Lengths of target sequences of shape (B,)
            
        Returns:
            torch.Tensor: CTC loss
        """
        # CTC loss requires log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC loss: (T, B, V)
        log_probs = log_probs.transpose(0, 1)
        
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,  # Assuming blank token is at index 0
            reduction='mean'
        )
        
        return loss
    
    def compute_ctc_alignment_loss(
        self, 
        phoneme_probs: torch.Tensor, 
        target_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss between CAR probabilities and target probabilities.
        
        Args:
            phoneme_probs (torch.Tensor): CAR phoneme probabilities (B, T, V)
            target_probs (torch.Tensor): Target probabilities from decoder (B, T, V)
            
        Returns:
            torch.Tensor: KL divergence loss
        """
        # Use KL divergence to align distributions
        kl_loss = F.kl_div(
            F.log_softmax(phoneme_probs, dim=-1),
            F.softmax(target_probs.detach(), dim=-1),
            reduction='batchmean'
        )
        
        return kl_loss
    
    def get_state_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate state representations for analysis.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Hidden state representations
        """
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.activation(h)
        
        for mamba_block in self.mamba_blocks:
            h = mamba_block(h)
            
        return h
    
    def get_output_dim(self) -> int:
        """Get output vocabulary size."""
        return self.output_dim
