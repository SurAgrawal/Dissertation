# train.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from Models.Shared.models import DeepONet
from Models.Shared.data import make_grid, sample_grf
from Models.Shared.utils import set_seed, make_run_dir, save_json, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser("Physics-informed DeepONet – ODE replication")
    p.add_argument('--ntrain', type=int, default=512, help='number of training functions')
    p.add_argument('--ntest',  type=int, default=128, help='number of test functions (saved for eval)')
    p.add_argument('--m',      type=int, default=100, help='number of grid/sensor points on [0,1]')
    p.add_argument('--ell',    type=float, default=0.2, help='RBF kernel length-scale')
    p.add_argument('--steps',  type=int, default=8000, help='training steps/iterations')
    p.add_argument('--batch',  type=int, default=64, help='batch size (# of functions per step)')
    p.add_argument('--width',  type=int, default=50, help='MLP width')
    p.add_argument('--depth',  type=int, default=5,  help='MLP depth (layers)')
    p.add_argument('--activation', default='tanh', choices=['tanh', 'relu', 'silu', 'gelu', 'softplus'])
    p.add_argument('--lr',     type=float, default=1e-3, help='Adam learning rate')
    p.add_argument('--save-dir', default='checkpoints', help='where to store runs (or /mnt/data/checkpoints)')
    p.add_argument('--run-name', default='', help='optional suffix for the run directory name')
    p.add_argument('--seed',   type=int, default=123, help='random seed')
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- map name of activation function ---
    act_map = {
        'tanh': nn.Tanh, 'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'softplus': nn.Softplus
    }
    act = act_map[args.activation]

    # --- Run directory & config ---
    run_dir = make_run_dir(args.save_dir, args.run_name)
    cfg_path = os.path.join(run_dir, 'config.json')
    save_json(vars(args), cfg_path)
    print(f"Run directory: {run_dir}")

    # --- Data (generate train/test & save test for reproducible eval) ---
    grid_np = make_grid(args.m, 0.0, 1.0)
    dx = float(grid_np[1] - grid_np[0])

    Utrain = sample_grf(args.ntrain, grid_np, ell=args.ell)  # (ntrain, m)
    Utest  = sample_grf(args.ntest , grid_np, ell=args.ell)  # (ntest, m)
    np.save(os.path.join(run_dir, 'test_U.npy'), Utest)
    np.save(os.path.join(run_dir, 'grid.npy'), grid_np)
    np.save(os.path.join(run_dir, 'dx.npy'), np.array([dx], dtype=np.float32))

    # --- Torch tensors ---
    u_train = torch.tensor(Utrain, dtype=torch.float32, device=device)  # (ntrain, m)
    u_test  = torch.tensor(Utest , dtype=torch.float32, device=device)  # (ntest, m)
    grid_t  = torch.tensor(grid_np, dtype=torch.float32, device=device) # (m,)

    # --- Model / Optimizer ---
    model = DeepONet(m=args.m, width=args.width, depth=args.depth, feat_dim=50, act=act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_rel = float('inf')
    last_ckpt = os.path.join(run_dir, 'last.pt')
    best_ckpt = os.path.join(run_dir, 'best.pt')

    # --- Training loop ---
    pbar = tqdm(range(1, args.steps + 1), ncols=100)
    for step in pbar:
        # sample a batch of functions
        idx = torch.randint(0, args.ntrain, (args.batch,), device=device)
        u_b = u_train[idx]  # (B, m)

        # collocation points: use sensor grid, enable autograd on x
        x_b = grid_t.view(1, args.m, 1).repeat(args.batch, 1, 1).clone().requires_grad_(True)

        # forward & residual
        s_pred = model(u_b, x_b)  # (B, m)
        ds_dx  = torch.autograd.grad(outputs=s_pred.sum(), inputs=x_b, create_graph=True)[0].squeeze(-1)
        loss_res = torch.mean((ds_dx - u_b) ** 2)

        # initial condition s(0) = 0  (evaluate model at x=0)
        x0 = torch.zeros(args.batch, 1, 1, device=device, requires_grad=True)
        s0 = model(u_b, x0)
        loss_ic = torch.mean(s0 ** 2)

        loss = loss_res + loss_ic

        opt.zero_grad()
        loss.backward()
        opt.step()

        # quick validation on a slice of test set every 500 steps
        rel = None
        if step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                Bv = min(32, args.ntest)
                x_t = grid_t.view(1, args.m, 1).repeat(Bv, 1, 1)
                s_hat = model(u_test[:Bv], x_t)
                s_true = torch.cumsum(u_test[:Bv], dim=1) * dx
                num = torch.linalg.norm(s_hat - s_true, dim=1)
                den = torch.linalg.norm(s_true, dim=1) + 1e-12
                rel = (num / den).mean().item()

            # save last
            save_checkpoint(last_ckpt, model, opt, step=step, metric_relL2=rel, cfg=vars(args))
            # save best
            if rel < best_rel:
                best_rel = rel
                save_checkpoint(best_ckpt, model, opt, step=step, metric_relL2=rel, cfg=vars(args))

        pbar.set_description(f"step {step:5d}  loss={loss.item():.3e}" + (f"  relL2={rel:.3e}" if rel is not None else ""))

    print(f"Training done. Best rel-L2: {best_rel:.4e}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")

if __name__ == "__main__":
    main()
