# models.py
from __future__ import annotations
import torch
import torch.nn as nn

def mlp(sizes, activation=nn.Tanh, last_layer_activation: bool = False):
    """
    Build a simple MLP with given layer sizes.
    sizes: [in, h1, h2, ..., out]
    """
    layers = []
    for i in range(len(sizes) - 1):
        in_d, out_d = sizes[i], sizes[i + 1]
        layers.append(nn.Linear(in_d, out_d))
        is_last = (i == len(sizes) - 2)
        if (not is_last) or last_layer_activation:
            layers.append(activation())
    return nn.Sequential(*layers)

class DeepONet(nn.Module):
    """
    Classic DeepONet: s(u, y) = <B(u), T(y)>
      - Branch: vector of sensor values u(x_i), i=1..m  -> R^F
      - Trunk : coordinate y (dim = in_dim)            -> R^F
    """
    def __init__(self, m: int, width: int = 50, depth: int = 5,
                 feat_dim: int = 50, act = nn.Tanh, in_dim: int = 1):
        super().__init__()
        hidden = [width] * (depth - 1)
        self.in_dim = in_dim
        self.branch = mlp([m] + hidden + [feat_dim], activation=act)
        self.trunk  = mlp([in_dim] + hidden + [feat_dim], activation=act)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        u: (B, m)
        y: (B, Q, in_dim)   e.g. in_dim=1 for ODE, in_dim=2 for (x,t) PDE
        returns s_hat: (B, Q)
        """
        B, Q, _ = y.shape
        b = self.branch(u)                              # (B, F)
        t = self.trunk(y.reshape(B * Q, self.in_dim))   # (B*Q, F)
        s = (b[:, None, :] * t.view(B, Q, -1)).sum(dim=-1)  # (B, Q)
        return s
