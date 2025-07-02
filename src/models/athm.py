"""
Adaptive Temporal Hierarchy Module (ATHM) for SerenaNet.

This module implements multi-resolution Conv1D processing with residual connections
and MLP gating for temporal feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ATHM(nn.Module):
    """
    Adaptive Temporal Hierarchy Module with multi-resolution Conv1D layers.
    
    Args:
        in_channels (int): Input feature dimension (default: 128)
        out_channels (int): Output feature dimension (default: 512)
        kernel_sizes (list): Convolution kernel sizes (default: [3, 3, 3])
        dilations (list): Dilation rates to control receptive field (default: [1, 2, 4])
        strides (list): Convolution strides (default: [1, 1, 1])
        l2_lambda (float): L2 regularization weight (default: 0.01)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 512,
        kernel_sizes: Optional[list] = None,
        dilations: Optional[list] = None,
        strides: Optional[list] = None,
        l2_lambda: float = 0.01
    ):
        super(ATHM, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if dilations is None:
            dilations = [1, 2, 4]
        if strides is None:
            strides = [1, 1, 1]
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l2_lambda = l2_lambda
        
        # Multi-resolution Conv1D branches
        self.conv_branches = nn.ModuleList()
        for k, s, d in zip(kernel_sizes, strides, dilations):
            pad = (k // 2) * d
            self.conv_branches.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    padding=pad,
                )
            )
        
        # Residual projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(out_channels, out_channels)
            for _ in kernel_sizes
        ])
        
        # Gated Linear Unit (better than simple MLP for gating large channels)
        self.glu_gate = nn.Sequential(
            nn.Linear(len(kernel_sizes) * out_channels, 2 * out_channels),
            nn.GLU(dim=-1)  # outputs (B,T,out_channels)
        )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(out_channels)
            for _ in kernel_sizes
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ATHM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, 128)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Output features of shape (B, T, 512)
                - L2 regularization loss
        """
        # x: (B, T, 128) -> (B, 128, T)
        x = x.transpose(1, 2)
        batch_size, _, seq_len = x.shape
        
        # Process through conv branches
        branch_outputs = []
        for i, (conv, proj, bn) in enumerate(zip(self.conv_branches, self.proj_layers, self.batch_norms)):
            # Convolution
            conv_out = conv(x)  # (B, 512, T_i)
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            
            # Residual connection with projection
            conv_out_transposed = conv_out.transpose(1, 2)  # (B, T_i, 512)
            proj_out = proj(conv_out_transposed)  # (B, T_i, 512)
            residual_out = conv_out_transposed + proj_out
            residual_out = residual_out.transpose(1, 2)  # (B, 512, T_i)
            
            # Interpolate to original sequence length
            if residual_out.size(-1) != seq_len:
                residual_out = F.interpolate(
                    residual_out, 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                )
            
            branch_outputs.append(residual_out)
        
        # Concatenate all branch outputs
        F_cat = torch.cat(branch_outputs, dim=1)  # (B, 3*512, T)
        F_cat = F_cat.transpose(1, 2)  # (B, T, 3*512)
        
        # MLP gating
        gate_weights = torch.sigmoid(self.glu_gate(F_cat))  # (B, T, 512)
        
        # Apply gating to first branch (could be enhanced to gate all branches)
        gated_output = gate_weights * F_cat[:, :, :self.out_channels]  # (B, T, 512)
        
        # L2 regularization loss
        l2_loss = self.l2_lambda * torch.norm(gate_weights, p=2, dim=-1).mean()
        
        return gated_output, l2_loss
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.out_channels
