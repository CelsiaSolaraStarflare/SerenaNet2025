"""
Official Mamba SSM wrapper.

This adapter allows SerenaNet to leverage the highly-optimized implementation
from the `mamba-ssm` package when it is installed.  It preserves the exact
interface of our local `MambaBlock` so that `CAR` and any other callers do
not need to change their logic.

If the external dependency is missing or if the user is on a platform where
CUDA kernels cannot be compiled (e.g. Apple M-series without Conda-Forge
wheel), importing this module will gracefully fall back to the local
light-weight Python implementation in `src/models/mamba.py`.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Try to import the reference implementation.  If that fails we will alias the
# existing Python fallback so that the rest of the codebase keeps working.
# -----------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaSSM  # type: ignore
    _mamba_ssm_available = True
    logger.info("Using official mamba-ssm implementation.")
except ImportError:
    logger.warning(
        "mamba-ssm is not installed. Falling back to local Mamba implementation."
    )
    from .mamba import MambaBlock as MambaSSM  # noqa: E402
    _mamba_ssm_available = False

# Local Mamba for comparison/fallback
# from .mamba import MambaBlock as _LocalMambaBlock  # noqa: E402


class MambaOfficialBlock(nn.Module):
    """Thin adapter around the official Mamba Block.

    Args:
        dim (int): hidden dimension (input = output = *dim*)
        state_dim (int): state space dimension `N` (called *d_state* in
            the paper / reference implementation).
        dt_rank (int): rank of Δt parameterisation (called *d_conv*).
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        dim: int = 256,
        state_dim: int = 16,
        dt_rank: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = MambaSSM(
            d_model=dim,
            d_state=state_dim,
            d_conv=dt_rank,
            expand=2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return self.block(x) 