
"""
Shared.pinn_models
==================
Neural network models and feature maps used for PINN experiments.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

@dataclass
class MLPConfig:
    in_dim: int = 1
    out_dim: int = 1
    width: int = 64
    depth: int = 3
    act: str = "tanh"  # "tanh" or "gelu"

def _make_act(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    """
    Simple fully-connected network.
    """
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        act = _make_act(cfg.act)
        layers = [nn.Linear(cfg.in_dim, cfg.width), act]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width), act]
        layers += [nn.Linear(cfg.width, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def fourier_features(x: torch.Tensor, K: int = 20) -> torch.Tensor:
    """
    z(x) = concat_{k=1..K} [ (1/k^2) sin(kx), (1/k^2) cos(kx) ]

    This scaling (1/k^2) implements the operator-aware preconditioning used in C.4.
    x: (N,1) tensor
    returns: (N, 2K) tensor
    """
    ks = torch.arange(1, K + 1, device=x.device, dtype=x.dtype).view(1, -1)  # (1,K)
    ax = x @ ks
    scale = 1.0 / (ks ** 2)
    z = torch.cat([torch.sin(ax) * scale, torch.cos(ax) * scale], dim=1)
    return z
