# eval.py
from __future__ import annotations
import os, argparse, time, csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..Shared.models import DeepONet
from ..Shared.utils import load_checkpoint, set_seed
from ..Shared.data import make_grid, sample_grf


def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained DeepONet ODE model (with points & L1/L2 errors)")
    p.add_argument('--ckpt', required=True, help='path to best.pt or last.pt')
    p.add_argument('--num-examples', type=int, default=6, help='how many examples to plot')
    p.add_argument('--outdir', default='', help='optional directory for plots; defaults to ckpt dir')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def per_sample_metrics(s_true: np.ndarray, s_pred: np.ndarray):
    """
    s_true, s_pred: shape (m,)
    returns dict with relL2, MAE, RMSE, MaxAbs, R2
    """
    err = s_pred - s_true
    relL2 = np.linalg.norm(err) / (np.linalg.norm(s_true) + 1e-12)
    mae   = np.mean(np.abs(err))
    rmse  = np.sqrt(np.mean(err**2))
    maxae = np.max(np.abs(err))
    sst   = np.sum((s_true - s_true.mean())**2)
    r2    = 1.0 - (np.sum(err**2) / (sst + 1e-12))
    return dict(relL2=relL2, MAE=mae, RMSE=rmse, MaxAbs=maxae, R2=r2)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load checkpoint + model ---
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = ckpt.get('cfg', {})
    m = int(cfg.get('m', 100))
    width = int(cfg.get('width', 50))
    depth = int(cfg.get('depth', 5))
    feat = int(cfg.get('feat_dim', 50)) if 'feat_dim' in cfg else 50

    model = DeepONet(m=m, width=width, depth=depth, feat_dim=feat).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # --- Paths / outdir ---
    run_dir = os.path.dirname(args.ckpt)
    ckpt_name = os.path.basename(args.ckpt).lower()
    base_out = args.outdir or run_dir
    label = 'best' if 'best' in ckpt_name else ('last' if 'last' in ckpt_name else 'ckpt')
    outdir = os.path.join(base_out, f"eval-{label}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    # --- Load saved test set (or generate quick one) ---
    grid_path = os.path.join(run_dir, 'grid.npy')
    U_path    = os.path.join(run_dir, 'test_U.npy')
    dx_path   = os.path.join(run_dir, 'dx.npy')

    if os.path.isfile(grid_path) and os.path.isfile(U_path) and os.path.isfile(dx_path):
        xs  = np.load(grid_path)
        U   = np.load(U_path)
        dx  = float(np.load(dx_path)[0])
    else:
        print("No saved test set found; generating a fresh one for quick plots.")
        ell   = float(cfg.get('ell', 0.2))
        ntest = int(cfg.get('ntest', 128)) if 'ntest' in cfg else 128
        xs = make_grid(m, 0.0, 1.0)
        dx = float(xs[1] - xs[0])
        U  = sample_grf(ntest, xs, ell=ell)

    # --- Predict on all test samples ---
    u_test = torch.tensor(U, dtype=torch.float32, device=device)      # (N, m)
    xs_t   = torch.tensor(xs, dtype=torch.float32, device=device)     # (m,)
    x_t    = xs_t.view(1, m, 1).repeat(u_test.size(0), 1, 1)          # (N, m, 1)

    with torch.no_grad():
        s_hat  = model(u_test, x_t)                                    # (N, m)
        s_true = torch.cumsum(u_test, dim=1) * dx                      # (N, m)

    s_hat_np  = s_hat.cpu().numpy()
    s_true_np = s_true.cpu().numpy()
    N         = s_hat_np.shape[0]

    # --- Overall metrics ---
    rels, maes, rmses, maxabs, r2s = [], [], [], [], []
    for i in range(N):
        m_i = per_sample_metrics(s_true_np[i], s_hat_np[i])
        rels.append(m_i['relL2']); maes.append(m_i['MAE']); rmses.append(m_i['RMSE'])
        maxabs.append(m_i['MaxAbs']); r2s.append(m_i['R2'])

    print(f"Overall (mean over {N} tests):")
    print(f"  relL2     = {np.mean(rels):.6f}")
    print(f"  MAE       = {np.mean(maes):.6f}")
    print(f"  RMSE      = {np.mean(rmses):.6f}")
    print(f"  MaxAbs    = {np.mean(maxabs):.6f}")
    print(f"  R^2       = {np.mean(r2s):.6f}")

    # Save per-sample metrics CSV
    per_metrics_path = os.path.join(outdir, "per_sample_metrics.csv")
    with open(per_metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","relL2","MAE","RMSE","MaxAbs","R2"])
        for i in range(N):
            w.writerow([i, rels[i], maes[i], rmses[i], maxabs[i], r2s[i]])
    print(f"Wrote per-sample metrics to: {per_metrics_path}")

    # --- Plots for a few examples (solution + L1 + L2) ---
    k = min(args.num_examples, N)
    for i in range(k):
        y_true = s_true_np[i]
        y_pred = s_hat_np[i]
        err_L1 = np.abs(y_pred - y_true)          # pointwise L1
        err_L2 = (y_pred - y_true) ** 2           # pointwise L2 (squared error)

        # Save the per-point table for this example (now includes L1 and L2)
        table = np.column_stack([xs, y_true, y_pred, err_L1, err_L2])
        np.savetxt(os.path.join(outdir, f'example_{i:02d}_points.csv'),
                   table, delimiter=',', header='x,s_true,s_pred,abs_error,L2_error', comments='')

        # 3-panel figure: solution (with predicted points), L1 error, L2 error
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios':[2,1,1]})

        # left: true + pred curves, plus pred *points*
        axs[0].plot(xs, y_true, label='true s(x)')
        axs[0].plot(xs, y_pred, linestyle='--', label='pred ŝ(x)')
        axs[0].plot(xs, y_pred, marker='o', linestyle='None', markersize=3, label='pred points')
        axs[0].set_xlabel('x'); axs[0].set_ylabel('solution s'); axs[0].legend(loc='best')
        axs[0].set_title(f'Example {i}: solution & predicted points'); axs[0].grid(alpha=0.3)

        # middle: pointwise absolute error (L1)
        axs[1].plot(xs, err_L1, marker='o', markersize=2)
        axs[1].set_xlabel('x'); axs[1].set_ylabel('|pred − true|')
        axs[1].set_title('Pointwise L1 error'); axs[1].grid(alpha=0.3)

        # right: pointwise squared error (L2)
        axs[2].plot(xs, err_L2, marker='o', markersize=2)
        axs[2].set_xlabel('x'); axs[2].set_ylabel('(pred − true)$^2$')
        axs[2].set_title('Pointwise L2 error'); axs[2].grid(alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(outdir, f'example_{i:02d}_solution_L1_L2.png')
        plt.savefig(out_path, dpi=150); plt.close(fig)

        # Keep the u(x) plot too
        fig_u, axu = plt.subplots(figsize=(7, 3.0))
        axu.plot(xs, U[i], label='u(x)')
        axu.set_xlabel('x'); axu.set_ylabel('u'); axu.set_title(f'Example {i}: input u(x)')
        axu.grid(alpha=0.3); fig_u.tight_layout()
        out_path_u = os.path.join(outdir, f'example_{i:02d}_u.png')
        plt.savefig(out_path_u, dpi=150); plt.close(fig_u)

    print(f"Wrote {k*2} PNGs and {k} CSVs to: {outdir}")

if __name__ == "__main__":
    main()
