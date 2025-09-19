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
    Classic DeepONet: s(u, x) = <B(u), T(x)>
    - Branch net takes vector of sensor values u(x_i), i=1..m
    - Trunk net takes coordinate x (scalar)
    """
    def __init__(self, m: int, width: int = 50, depth: int = 5, feat_dim: int = 50, act = nn.Tanh):
        super().__init__()
        hidden = [width] * (depth - 1)
        self.branch = mlp([m] + hidden + [feat_dim], activation=act)
        self.trunk  = mlp([1] + hidden + [feat_dim], activation=act)

    def forward(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        u: (B, m)
        x: (B, Q, 1)
        returns s_hat: (B, Q)
        """
        B, Q, _ = x.shape
        b = self.branch(u)                        # (B, F)
        t = self.trunk(x.reshape(B * Q, 1))       # (B*Q, F)
        s = (b[:, None, :] * t.view(B, Q, -1)).sum(dim=-1)  # (B, Q)
        return s
