
"""
Shared.pinn_ops
===============
Core operators for PINN training and A-matrix construction (C.4).
"""

from dataclasses import dataclass
import math
import torch

torch.set_default_dtype(torch.float64)

@dataclass
class PDEConfig:
    N: int = 2048
    k: float = 1.0
    lam_bc: float = 0.0
    domain_left: float = -math.pi
    domain_right: float = math.pi

def make_grid(cfg: PDEConfig, device=None, dtype=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.get_default_dtype()
    x = torch.linspace(cfg.domain_left, cfg.domain_right, cfg.N, device=device, dtype=dtype).reshape(-1, 1)
    dx = (cfg.domain_right - cfg.domain_left) / (cfg.N - 1)
    w = torch.full((cfg.N, 1), dx, device=device, dtype=dtype)
    return x, w

def u_star(x: torch.Tensor, k: float) -> torch.Tensor:
    return torch.sin(k * x)

def forcing(x: torch.Tensor, k: float) -> torch.Tensor:
    # For u'' + f = 0 with u = sin(kx)  =>  f = +k^2 sin(kx)
    return (k ** 2) * torch.sin(k * x)

def u_of_x_builder(model, featurizer=None):
    if featurizer is None:
        return lambda xin: model(xin)
    return lambda xin: model(featurizer(xin))

def residual(u_of_x, x, f_grid):
    """
    r(x) = u''(x) + f(x)
    """
    x_req = x.clone().requires_grad_(True)
    u = u_of_x(x_req)
    du = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x_req, torch.ones_like(du), create_graph=True)[0]
    return d2u + f_grid

def physics_loss(u_of_x, x, weights, f_grid):
    r = residual(u_of_x, x, f_grid)
    return (r ** 2 * weights).sum()

def l2_error(u_pred, x, weights, k: float):
    err = (((u_pred - u_star(x, k)) ** 2) * weights).sum().sqrt()
    ref = (((u_star(x, k)) ** 2) * weights).sum().sqrt()
    return (err / (ref + 1e-12)).item()

def A_matrix(model, x, weights, k: float, featurizer=None, lambda_bc: float = 0.0):
    """
    Construct A = (DJ)^T W (DJ) + lambda * J_b^T W_b J_b
    where J_ij = ∂u(x_i)/∂θ_j and D = d^2/dx^2.
    This version computes J by iterating samples: for each x_i, compute grad u(x_i) wrt params.
    Complexity: O(N) backward passes -> pick moderate N (e.g., 512) for spectrum runs.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    u_of_x = u_of_x_builder(model, featurizer)

    # Build J sample-by-sample: for each i, grad(u_i) wrt θ -> (P,)
    J_rows = []
    x_req = x.clone().requires_grad_(True)
    u_all = u_of_x(x_req)  # (N,1)

    for i in range(u_all.shape[0]):
        ui = u_all[i:i+1]  # scalar
        grads = torch.autograd.grad(ui, params, retain_graph=True, create_graph=True)
        gvec = torch.nn.utils.parameters_to_vector([g.view(-1) for g in grads])  # (P,)
        J_rows.append(gvec)

    J = torch.stack(J_rows, dim=0)  # (N,P)

    # Apply D = d^2/dx^2 to each column via AD by differentiating phi_i(x) across x.
    # To do this, re-materialize phi_i(x) by re-differentiating but batching over params is expensive.
    # A pragmatic approach: finite-difference DJ on the *Jacobian rows* with respect to x.
    # However, to stay faithful to C.4, we use AD:
    DJ_cols = []
    for j in range(P):
        # Recompute phi_j(x) as function of x by differentiating u wrt θ_j across all x.
        # We can get phi_j by taking a dot with a basis vector e_j using autograd again.
        # Trick: take scalar s = sum_{p} θ_p * 0 + θ_j * 1; but θ are not inputs.
        # So we recompute per-x grads and collect j-th param entry -> phi_j(x).
        phi_j = J[:, j:j+1]  # (N,1) already per x (since J rows correspond to x)
        # Differentiate phi_j w.r.t. x twice:
        x_req = x.clone().requires_grad_(True)
        # We need phi_j(x) as function node tied to x_req; safely re-evaluate u and grads to connect graph.
        u = u_of_x(x_req)
        col_vals = []
        for i in range(u.shape[0]):
            ui = u[i:i+1]
            grads = torch.autograd.grad(ui, params, retain_graph=True, create_graph=True)
            gvec = torch.nn.utils.parameters_to_vector([g.view(-1) for g in grads])
            col_vals.append(gvec[j])
        col = torch.stack(col_vals).reshape(-1,1)  # (N,1)
        dcol_dx = torch.autograd.grad(col, x_req, torch.ones_like(col), retain_graph=True, create_graph=True)[0]
        d2col_dx2 = torch.autograd.grad(dcol_dx, x_req, torch.ones_like(dcol_dx), retain_graph=True, create_graph=True)[0]
        DJ_cols.append(d2col_dx2.detach())
    DJ = torch.cat(DJ_cols, dim=1)  # (N,P)

    W = torch.diag(weights.squeeze())
    A = DJ.T @ W @ DJ

    if lambda_bc != 0.0:
        xb = torch.tensor([[x.min().item()], [x.max().item()]], device=x.device, dtype=x.dtype).requires_grad_(True)
        ub = u_of_x(xb)
        Jb_cols = []
        for j in range(P):
            col = []
            for n in range(ub.shape[0]):
                un = ub[n:n+1]
                grads = torch.autograd.grad(un, params, retain_graph=True, create_graph=True)
                gvec = torch.nn.utils.parameters_to_vector([g.view(-1) for g in grads])
                col.append(gvec[j])
            col = torch.stack(col).reshape(2,1)
            Jb_cols.append(col)
        Jb = torch.cat(Jb_cols, dim=1)  # (2,P)
        wb = torch.eye(2, device=x.device, dtype=x.dtype)
        A = A + lambda_bc * (Jb.T @ wb @ Jb)
    return A.detach()
