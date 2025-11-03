
"""
BurgerPreCondition.burger_ops
-----------------------------
Operators, residuals, losses, and A-matrix (linearized) for Burgers.
"""

from dataclasses import dataclass
import math
import torch

from .burger_models import build_u_of_xt, FeatureConfig, fourier_features_2d

torch.set_default_dtype(torch.float64)

@dataclass
class BurgersConfig:
    # domain & grids
    Nx: int = 256
    Nt: int = 128
    a: float = 0.0
    b: float = 2*math.pi
    T: float = 1.0

    # physics
    nu: float = 1e-2
    boundary: str = "periodic"   # periodic|dirichlet|neumann
    hard_periodic: bool = True

    # loss weights
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0

    # misc
    device: str = "cpu"


def make_grid_2d(cfg: BurgersConfig):
    """Return collocation grid (x,t) and quadrature weights w(x,t)."""
    x = torch.linspace(cfg.a, cfg.b, cfg.Nx, device=cfg.device, dtype=torch.get_default_dtype(), requires_grad=True)
    t = torch.linspace(0.0, cfg.T, cfg.Nt, device=cfg.device, dtype=torch.get_default_dtype(), requires_grad=True)
    X, Tt = torch.meshgrid(x, t, indexing="ij")  # (Nx,Nt)
    dx = (cfg.b - cfg.a) / (cfg.Nx - 1)
    dt = cfg.T / (cfg.Nt - 1)
    w = torch.full_like(X, dx*dt)
    # flatten
    xg = X.reshape(-1, 1)
    tg = Tt.reshape(-1, 1)
    wg = w.reshape(-1, 1)
    return xg, tg, wg


def initial_condition(x: torch.Tensor):
    return -torch.sin(x)


def residual(u_of_xt, x, t, cfg: BurgersConfig):
    """r = u_t + u u_x - nu u_xx"""
    u = u_of_xt(x, t)
    ones = torch.ones_like(u)
    ut = torch.autograd.grad(u, t, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    ux = torch.autograd.grad(u, x, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), retain_graph=True, create_graph=True)[0]
    return ut + u*ux - cfg.nu * uxx


def bc_penalty(u_of_xt, cfg: BurgersConfig):
    if cfg.boundary == "periodic":
        # sample times
        t = torch.linspace(0.0, cfg.T, cfg.Nt, device=cfg.device, dtype=torch.get_default_dtype()).view(-1,1)
        # ensure both x endpoints require gradients for derivative matching
        t = t.detach().requires_grad_(True)
        x0 = torch.full_like(t, cfg.a)
        xL = torch.full_like(t, cfg.b)
        x0.requires_grad_(True)
        xL.requires_grad_(True)

        u0 = u_of_xt(x0, t)
        uL = u_of_xt(xL, t)

        # first derivatives should also match
        ones0 = torch.ones_like(u0)
        ux0 = torch.autograd.grad(u0, x0, grad_outputs=ones0, retain_graph=True, create_graph=True)[0]
        uxL = torch.autograd.grad(uL, xL, grad_outputs=torch.ones_like(uL), retain_graph=True, create_graph=True)[0]
        return ((u0 - uL)**2).mean() + ((ux0 - uxL)**2).mean()
    else:
        # For non-periodic BCs, add Dirichlet/Neumann penalties here in future.
        return torch.tensor(0.0, device=cfg.device, dtype=torch.get_default_dtype())


def physics_loss(u_of_xt, cfg: BurgersConfig):
    xg, tg, wg = make_grid_2d(cfg)
    r = residual(u_of_xt, xg, tg, cfg)
    loss_int = (r**2 * wg).sum()

    # IC term
    x0 = torch.linspace(cfg.a, cfg.b, cfg.Nx, device=cfg.device, dtype=torch.get_default_dtype()).view(-1,1)
    x0 = x0.detach().requires_grad_(True)
    t0 = torch.zeros_like(x0, requires_grad=True)
    ic = (u_of_xt(x0, t0) - initial_condition(x0))**2
    loss_ic = ic.mean()

    # BC term (periodic only for now)
    loss_bc = bc_penalty(u_of_xt, cfg)

    return loss_int + cfg.lambda_ic*loss_ic + cfg.lambda_bc*loss_bc


def l2_error_vs_reference(u_of_xt, x_eval, t_eval, w_eval, u_ref):
    u = u_of_xt(x_eval, t_eval)
    err = ((u - u_ref)**2 * w_eval).sum().sqrt()
    ref = ((u_ref**2) * w_eval).sum().sqrt()
    return (err / (ref + 1e-12)).item()


def A_matrix_linearized(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor,
                        cfg: BurgersConfig, featurizer=None):
    """
    Build A ≈ (D_lin J)^T W (D_lin J) where D_lin = ∂t - nu ∂xx.
    Mirrors your Poisson construction but in 2D and with the linearized operator.
    Note: This is intentionally simple (and slow) but faithful to your existing flow.
    """
    u_of_xt = build_u_of_xt(model, featurizer)
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    cols = []
    # Compute parameter-wise sensitivity by repeated backward passes (costly but simple)
    for j in range(P):
        # Recompute phi_j for all samples
        x_req = x.detach().clone().requires_grad_(True)
        t_req = t.detach().clone().requires_grad_(True)
        phi_vals = []
        for i in range(x_req.shape[0]):
            ui = u_of_xt(x_req[i:i+1], t_req[i:i+1])
            grads = torch.autograd.grad(ui, params, retain_graph=True, create_graph=True)
            gvec = torch.nn.utils.parameters_to_vector([g.reshape(-1) for g in grads])
            phi_vals.append(gvec[j])
        ph = torch.stack(phi_vals).view(-1,1)

        ph_t = torch.autograd.grad(ph, t_req, grad_outputs=torch.ones_like(ph), retain_graph=True, create_graph=True)[0]
        ph_x = torch.autograd.grad(ph, x_req, grad_outputs=torch.ones_like(ph), retain_graph=True, create_graph=True)[0]
        ph_xx = torch.autograd.grad(ph_x, x_req, grad_outputs=torch.ones_like(ph_x), retain_graph=True, create_graph=True)[0]
        Dphi = ph_t - cfg.nu * ph_xx
        cols.append(Dphi)

    DJ = torch.cat(cols, dim=1)  # (N, P)
    W = torch.diag(w.view(-1))
    A = DJ.T @ W @ DJ
    return A.detach()
