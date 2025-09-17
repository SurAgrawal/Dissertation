# eval.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import DeepONet
from utils import load_checkpoint, set_seed

def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained DeepONet ODE model")
    p.add_argument('--ckpt', required=True, help='path to best.pt or last.pt')
    p.add_argument('--num-examples', type=int, default=6, help='how many examples to plot')
    p.add_argument('--outdir', default='', help='optional directory for plots; defaults to ckpt dir')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load checkpoint and config ---
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = ckpt.get('cfg', {})
    m = int(cfg.get('m', 100))
    width = int(cfg.get('width', 50))
    depth = int(cfg.get('depth', 5))

    model = DeepONet(m=m, width=width, depth=depth).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # --- Find run directory & default outdir ---
    run_dir = os.path.dirname(args.ckpt)
    outdir = args.outdir or run_dir
    os.makedirs(outdir, exist_ok=True)

    # --- Load saved test set (fallback to random if missing) ---
    grid_path = os.path.join(run_dir, 'grid.npy')
    U_path    = os.path.join(run_dir, 'test_U.npy')
    dx_path   = os.path.join(run_dir, 'dx.npy')

    if os.path.isfile(grid_path) and os.path.isfile(U_path) and os.path.isfile(dx_path):
        grid = np.load(grid_path)
        U    = np.load(U_path)
        dx   = float(np.load(dx_path)[0])
    else:
        print("No saved test set found; generating a fresh one for quick plots.")
        from data import make_grid, sample_grf
        ell = float(cfg.get('ell', 0.2))
        ntest = int(cfg.get('ntest', 128))
        grid = make_grid(m, 0.0, 1.0)
        dx = float(grid[1] - grid[0])
        U = sample_grf(ntest, grid, ell=ell)

    # --- Compute predictions & metrics on all test samples ---
    u_test = torch.tensor(U, dtype=torch.float32, device=device)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    x_t = grid_t.view(1, m, 1).repeat(u_test.size(0), 1, 1)

    with torch.no_grad():
        s_hat = model(u_test, x_t)  # (N, m)
        s_true = torch.cumsum(u_test, dim=1) * (grid[1] - grid[0])
        num = torch.linalg.norm(s_hat - s_true, dim=1)
        den = torch.linalg.norm(s_true, dim=1) + 1e-12
        rel = (num / den).mean().item()
    print(f"Test relative L2 (on saved/fresh test set): {rel:.6f}")

    # --- Plots for a few examples ---
    k = min(args.num_examples, u_test.size(0))
    xs = grid
    s_hat_np = s_hat.detach().cpu().numpy()
    s_true_np = s_true.detach().cpu().numpy()
    U_np = U

    for i in range(k):
        fig = plt.figure(figsize=(7, 4.5))
        ax1 = plt.gca()
        ax1.plot(xs, s_true_np[i], label='true s(x)')
        ax1.plot(xs, s_hat_np[i], linestyle='--', label='pred s_hat(x)')
        ax1.set_xlabel('x'); ax1.set_ylabel('solution s')
        ax1.legend(loc='best')
        ax1.set_title(f'Example {i}: solution')

        fig.tight_layout()
        out_path = os.path.join(outdir, f'example_{i:02d}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        # also plot u(x)
        fig = plt.figure(figsize=(7, 3.0))
        ax2 = plt.gca()
        ax2.plot(xs, U_np[i], label='u(x)')
        ax2.set_xlabel('x'); ax2.set_ylabel('u')
        ax2.set_title(f'Example {i}: input u(x)')
        fig.tight_layout()
        out_path_u = os.path.join(outdir, f'example_{i:02d}_u.png')
        plt.savefig(out_path_u, dpi=150)
        plt.close(fig)

    print(f"Wrote {k*2} PNGs to: {outdir}")

if __name__ == "__main__":
    main()
