
"""
BurgerPreCondition.burger_models
--------------------------------
Models and feature maps for 1D viscous Burgers on (x,t).
- We reuse the Shared MLP.
- Provide 2D Fourier features with operator-aware scaling:
    alpha_{k,m} = 1 / sqrt( (m*omega0)^2 + (nu*k^2)^2 )
This flattens the spectrum of the linearized operator (∂t - nu ∂xx).
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from ..Shared.pinn_models import MLP, MLPConfig

torch.set_default_dtype(torch.float64)

@dataclass
class FeatureConfig:
    Kx: int = 24   # spatial modes
    Mt: int = 24   # temporal modes
    T: float = 1.0
    nu: float = 1e-2
    hard_periodic: bool = True  # if True and boundary is periodic, features enforce periodicity


def fourier_features_2d(x: torch.Tensor, t: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """
    Build scaled sin/cos features over (x,t).
    x: (N,1), t: (N,1)
    returns: (N, 2*Kx*Mt) tensor
    """
    device, dtype = x.device, x.dtype
    Kx, Mt = cfg.Kx, cfg.Mt
    omega0 = 2.0 * math.pi / cfg.T

    ks = torch.arange(1, Kx + 1, device=device, dtype=dtype).view(1, -1)
    ms = torch.arange(0, Mt, device=device, dtype=dtype).view(1, -1)  # include m=0

    # (N,Kx) and (N,Mt)
    ax = x @ ks
    at = t @ (ms * omega0)

    # Broadcast to (N,Kx,Mt)
    ax = ax.unsqueeze(-1).expand(-1, -1, Mt)
    at = at.unsqueeze(1).expand(-1, Kx, -1)

    # Scaling per (k,m)
    k2 = ks**2  # (1,Kx)
    w = ms * omega0  # (1,Mt)
    # form sqrt(w^2 + (nu*k^2)^2)
    scal = torch.sqrt(w.view(1, 1, Mt) ** 2 + (cfg.nu * k2.view(1, Kx, 1)) ** 2)
    scal = torch.clamp(scal, min=1e-8)
    inv = 1.0 / scal  # (1,Kx,Mt)

    sin_part = torch.sin(ax + at) * inv
    cos_part = torch.cos(ax + at) * inv

    z = torch.cat([sin_part, cos_part], dim=1)  # (N, 2*Kx, Mt)
    z = z.reshape(x.shape[0], -1)
    return z


def build_u_of_xt(model: nn.Module, featurizer=None):
    """
    Return callable u(x,t) that handles feature mapping if provided.
    Inputs must be tensors with shape (N,1) each.
    Output shape: (N,1).
    """
    def u_of_xt(x, t):
        if featurizer is None:
            inp = torch.cat([x, t], dim=1)
            return model(inp)
        else:
            z = featurizer(x, t)
            return model(z)
    return u_of_xt
