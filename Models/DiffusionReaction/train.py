# train.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..Shared.models import DeepONet
from ..Shared.data import make_grid, sample_grf
from ..Shared.utils import set_seed, make_run_dir, save_json, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser("Physics-informed DeepONet – Diffusion–reaction PDE")
    # data / grids
    p.add_argument('--ntrain', type=int, default=10000, help='training functions')
    p.add_argument('--ntest',  type=int, default=128, help='test functions (saved for eval)')
    p.add_argument('--m',      type=int, default=100, help='# x-grid sensor points on [0,1]')
    p.add_argument('--ell',    type=float, default=0.2, help='RBF kernel length-scale for u(x)')
    # network
    p.add_argument('--width',  type=int, default=50)
    p.add_argument('--depth',  type=int, default=5)
    p.add_argument('--feat-dim', type=int, default=50)
    p.add_argument('--activation', default='tanh', choices=['tanh','relu','silu','gelu','softplus'])
    # physics / training
    p.add_argument('--steps',  type=int, default=12000, help='training iterations')
    p.add_argument('--batch',  type=int, default=1000, help='batch size (functions)')
    p.add_argument('--q',      type=int, default=256, help='# collocation points per function')
    p.add_argument('--q-ic',   type=int, default=128, help='# IC points per function')
    p.add_argument('--q-bc',   type=int, default=128, help='# BC points per function (each boundary)')
    p.add_argument('--D',      type=float, default=0.01, help='diffusion coefficient')
    p.add_argument('--k',      type=float, default=0.01, help='reaction rate')
    p.add_argument('--w-pde',  type=float, default=1.0, help='weight for PDE residual')
    p.add_argument('--w-ic',   type=float, default=1.0, help='weight for IC loss')
    p.add_argument('--w-bc',   type=float, default=1.0, help='weight for BC loss')
    p.add_argument('--lr',     type=float, default=1e-3, help='Adam lr')
    # housekeeping
    p.add_argument('--save-dir', default='Models/DiffusionReaction/checkpoints')
    p.add_argument('--run-name', default='dr')
    p.add_argument('--seed',   type=int, default=123)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- activation mapping ---
    act_map = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'silu': nn.SiLU,
               'gelu': nn.GELU, 'softplus': nn.Softplus}
    act = act_map[args.activation]

    # --- run dir & config ---
    run_dir = make_run_dir(args.save_dir, args.run_name)
    save_json(vars(args), os.path.join(run_dir, 'config.json'))
    print(f"Run directory: {run_dir}")

    # --- data ---
    x_grid = make_grid(args.m, 0.0, 1.0)            # np.float64
    Utrain = sample_grf(args.ntrain, x_grid, ell=args.ell)  # (ntrain, m) float32
    Utest  = sample_grf(args.ntest , x_grid, ell=args.ell)
    np.save(os.path.join(run_dir, 'test_U.npy'), Utest)
    np.save(os.path.join(run_dir, 'grid.npy'), x_grid)

    u_train = torch.tensor(Utrain, dtype=torch.float32, device=device)
    u_test  = torch.tensor(Utest , dtype=torch.float32, device=device)
    grid_t  = torch.tensor(x_grid, dtype=torch.float32, device=device) # (m,)

    # --- model & optimizer ---
    model = DeepONet(m=args.m, width=args.width, depth=args.depth,
                     feat_dim=args.feat_dim, act=act, in_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = float('inf')
    last_ckpt = os.path.join(run_dir, 'last.pt')
    best_ckpt = os.path.join(run_dir, 'best.pt')

    # --- training ---
    pbar = tqdm(range(1, args.steps + 1), ncols=100)
    for step in pbar:
        model.train()
        # sample batch of functions
        idx = torch.randint(0, args.ntrain, (args.batch,), device=device)
        u_b = u_train[idx]                      # (B, m)
        B   = u_b.size(0)

        # ===== PDE collocation (x,t) =====
        x_idx = torch.randint(0, args.m, (B, args.q), device=device)  # (B,Q)
        x_col = grid_t[x_idx]                                         # (B,Q)
        t_col = torch.rand(B, args.q, device=device)                  # (B,Q)
        XT    = torch.stack([x_col, t_col], dim=-1).requires_grad_(True)  # (B,Q,2)

        s = model(u_b, XT)                    # (B,Q)

        # first partials wrt x,t
        grads = torch.autograd.grad(s, XT, grad_outputs=torch.ones_like(s),
                                    create_graph=True)[0]             # (B,Q,2)
        s_x = grads[..., 0]
        s_t = grads[..., 1]
        # second derivative wrt x
        g2   = torch.autograd.grad(s_x, XT, grad_outputs=torch.ones_like(s_x),
                                   create_graph=True)[0]              # (B,Q,2)
        s_xx = g2[..., 0]

        # u(x) at collocation x: gather along sensor dimension
        u_forcing = torch.gather(u_b, dim=1, index=x_idx)             # (B,Q)

        res = s_t - args.D * s_xx - args.k * (s**2) - u_forcing
        loss_pde = (res**2).mean()

        # ===== IC: s(x,0)=0 =====
        x_ic_idx = torch.randint(0, args.m, (B, args.q_ic), device=device)
        x_ic = grid_t[x_ic_idx]
        XT_ic = torch.stack([x_ic, torch.zeros_like(x_ic)], dim=-1).requires_grad_(True)
        s_ic = model(u_b, XT_ic)
        loss_ic = (s_ic**2).mean()

        # ===== BCs: s(0,t)=0 and s(1,t)=0 =====
        t_bc = torch.rand(B, args.q_bc, device=device)
        XT_b0 = torch.stack([torch.zeros_like(t_bc), t_bc], dim=-1).requires_grad_(True)
        XT_b1 = torch.stack([torch.ones_like(t_bc),  t_bc], dim=-1).requires_grad_(True)
        s_b0 = model(u_b, XT_b0)
        s_b1 = model(u_b, XT_b1)
        loss_bc = (s_b0**2).mean() + (s_b1**2).mean()

        loss = args.w_pde*loss_pde + args.w_ic*loss_ic + args.w_bc*loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        # quick physics metric on a slice of test set
        metric = None
        if step % 500 == 0 or step == args.steps:
            model.eval()

            # NOTE: do NOT use torch.no_grad() here; we need input gradients
            Bv = min(32, args.ntest)

            # random collocation points
            x_idx_v = torch.randint(0, args.m, (Bv, args.q), device=device)
            x_v = grid_t[x_idx_v]
            t_v = torch.rand(Bv, args.q, device=device)

            XT_v = torch.stack([x_v, t_v], dim=-1)
            XT_v = XT_v.detach().requires_grad_(True)  # make it a leaf & track grads

            # forward
            s_v = model(u_test[:Bv], XT_v)  # (Bv, Q)

            # first partials wrt x,t
            grads_v = torch.autograd.grad(
                outputs=s_v,
                inputs=XT_v,
                grad_outputs=torch.ones_like(s_v),
                create_graph=True,  # we only need numbers, not higher-order for val
                retain_graph=True
            )[0]  # (Bv, Q, 2)
            s_x_v = grads_v[..., 0]
            s_t_v = grads_v[..., 1]

            # second derivative wrt x
            g2_v = torch.autograd.grad(
                outputs=s_x_v,
                inputs=XT_v,
                grad_outputs=torch.ones_like(s_x_v),
                create_graph=False,
                retain_graph=False
            )[0]
            s_xx_v = g2_v[..., 0]

            u_forcing_v = torch.gather(u_test[:Bv], 1, x_idx_v)
            res_v = s_t_v - args.D * s_xx_v - args.k * (s_v ** 2) - u_forcing_v
            metric = res_v.pow(2).mean().item()

            # save last & best (by residual metric)
            save_checkpoint(last_ckpt, model, opt, step=step, metric_relL2=metric, cfg=vars(args))
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(best_ckpt, model, opt, step=step, metric_relL2=metric, cfg=vars(args))

        pbar.set_description(f"step {step:5d}  loss={loss.item():.3e}" +
                             (f"  resMSE={metric:.3e}" if metric is not None else ""))

    print(f"Training done. Best residual-MSE: {best_metric:.4e}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")

if __name__ == "__main__":
    main()
