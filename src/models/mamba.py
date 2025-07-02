import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Any = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        device: str = "cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        self.dim = d_model
        self.state_dim = d_state
        self.conv_dim = d_conv
        self.expand = expand
        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
        self.d_inner = int(self.expand * self.dim)

        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias to 0
        nn.init.constant_(self.dt_proj.bias, 0.0)

        # Initialize dt_proj weights
        A = torch.arange(1, self.state_dim + 1, dtype=torch.float32).view(self.state_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)

        # S4D real initialization
        A = torch.arange(1, self.state_dim + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore
        
        # Initialize dt proj
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias = nn.Parameter(inv_dt)
        self.dt_proj.bias._no_weight_decay = True  # type: ignore


    def forward(self, x):
        b, l, d = x.shape
        
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        return self.out_proj(y)

    def ssm(self, x):
        (d_inner, n) = self.A_log.shape

        # Compute ∆, A, B, C, D through linear projections
        A = -torch.exp(self.A_log.float())  # (d_inner, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # (b, l, dt_rank), (b, l, n), (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_inner)
        
        y = self.selective_scan(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_inner) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # A is being broadcasted from (d_inner, n) to (b, l, d_inner, n)
        # B is being broadcasted from (b, l, n) to (b, l, d_inner, n)
        delta_A = torch.exp(delta.unsqueeze(-1) * A) # (b, l, d_inner, n)
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1) # (b, l, d_inner, n)

        # Perform parallel scan (algorithm 1 in paper)
        x = torch.zeros((b, d_inner, n), device=u.device)
        ys = []
        for i in range(l):
            x = delta_A[:, i] * x + delta_B_u[:, i]
            y = (x @ C[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b, l, d_inner)

        y = y + u * D
    
        return y 