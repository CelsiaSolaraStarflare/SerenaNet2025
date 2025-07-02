"""
Transformer Encoder module for SerenaNet.

This module implements a multi-layer transformer encoder with positional encoding
for sequence modeling in speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Optional fast attention / feed-forward kernels
_flash_attn_available = False
try:
    from flash_attn.layers.rotary import apply_rotary  # type: ignore
    from flash_attn.modules.mha import MHA  # type: ignore
    from flash_attn.modules.ff import FF  # type: ignore

    _flash_attn_available = True
except ModuleNotFoundError:
    # Flash-Attn not installed; we will fall back to standard nn.TransformerEncoder
    pass


class PositionalEncoding(nn.Module):
    """
    Positional encoding module using learnable embeddings.
    
    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.d_model = d_model
        
        # Initialize with sinusoidal encoding
        self._init_sinusoidal()
    
    def _init_sinusoidal(self):
        """Initialize with sinusoidal positional encoding."""
        position = torch.arange(0, self.pos_embedding.size(1)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           -(math.log(10000.0) / self.d_model))
        
        with torch.no_grad():
            self.pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
            self.pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D)
            
        Returns:
            torch.Tensor: Output with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for SerenaNet.
    
    Args:
        d_model (int): Model dimension (default: 512)
        nhead (int): Number of attention heads (default: 8)
        num_layers (int): Number of transformer layers (default: 6)
        dim_feedforward (int): Feedforward dimension (default: 2048)
        dropout (float): Dropout probability (default: 0.1)
        max_len (int): Maximum sequence length (default: 1000)
        activation (str): Activation function (default: 'relu')
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1000,
        activation: str = 'relu'
    ):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Input layer norm
        self.input_norm = nn.LayerNorm(d_model)
        
        # Choose implementation automatically
        self.use_flash = _flash_attn_available

        if self.use_flash:
            # Build stack of FlashAttention-based blocks
            self.flash_layers = nn.ModuleList([
                _FlashEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:
            # Fallback to standard nn.TransformerEncoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D)
            src_key_padding_mask (torch.Tensor, optional): Padding mask of shape (B, T)
            
        Returns:
            torch.Tensor: Encoded features of shape (B, T, D)
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if self.use_flash:
            if src_key_padding_mask is not None:
                # Flash-Attn MHA currently doesn't support key padding mask; fall back
                encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # type: ignore
            else:
                h = x
                for blk in self.flash_layers:
                    h = blk(h)
                encoded = h
        else:
            encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output normalization
        encoded = self.output_norm(encoded)
        
        return encoded
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        layer_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """
        Extract attention weights from a specific layer.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int): Layer index to extract weights from (-1 for last layer)
            
        Returns:
            torch.Tensor: Attention weights
        """
        # This is a simplified version - full implementation would require
        # modifying the transformer layers to return attention weights
        if layer_idx < 0:
            layer_idx = self.num_layers + layer_idx
            
        # For now, return None - would need custom implementation
        return None
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.d_model

if _flash_attn_available:
    class _FlashEncoderBlock(nn.Module):
        """Single encoder block using flash-attn fused MHA and FF."""

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()

            from flash_attn.modules.mha import MHA  # type: ignore
            from flash_attn.modules.ff import FF  # type: ignore

            self.norm1 = nn.LayerNorm(d_model)
            self.attn = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = FF(d_model, dim_feedforward, dropout=dropout)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,D)
            h = self.norm1(x)
            h = self.attn(h)
            x = x + self.dropout(h)

            h2 = self.norm2(x)
            h2 = self.ff(h2)
            return x + self.dropout(h2)
