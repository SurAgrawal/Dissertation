# eval.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time


from Models.Shared.models import DeepONet
from Models.Shared.utils import load_checkpoint, set_seed

def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained DeepONet ODE model (with points & errors)")
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
    # R^2 = 1 - SSE/SST; handle zero-variance edge case
    sst = np.sum((s_true - s_true.mean())**2)
    r2  = 1.0 - (np.sum(err**2) / (sst + 1e-12))
    return dict(relL2=relL2, MAE=mae, RMSE=rmse, MaxAbs=maxae, R2=r2)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load checkpoint and model ---
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = ckpt.get('cfg', {})
    m = int(cfg.get('m', 100))
    width = int(cfg.get('width', 50))
    depth = int(cfg.get('depth', 5))

    model = DeepONet(m=m, width=width, depth=depth).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # --- Paths / outdir ---
    run_dir = os.path.dirname(args.ckpt)
    ckpt_name = os.path.basename(args.ckpt).lower()
    base_out = args.outdir or run_dir
    label = 'best' if 'best' in ckpt_name else ('last' if 'last' in ckpt_name else 'ckpt')

    # put each eval into its own timestamped folder
    outdir = os.path.join(base_out, f"eval-{label}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    # --- Load saved test set (or generate if missing) ---
    grid_path = os.path.join(run_dir, 'grid.npy')
    U_path    = os.path.join(run_dir, 'test_U.npy')
    dx_path   = os.path.join(run_dir, 'dx.npy')

    if os.path.isfile(grid_path) and os.path.isfile(U_path) and os.path.isfile(dx_path):
        grid = np.load(grid_path)
        U    = np.load(U_path)
        dx   = float(np.load(dx_path)[0])
    else:
        print("No saved test set found; generating a fresh one for quick plots.")
        from Models.Shared.data import make_grid, sample_grf
        ell = float(cfg.get('ell', 0.2))
        ntest = int(cfg.get('ntest', 128))
        grid = make_grid(m, 0.0, 1.0)
        dx = float(grid[1] - grid[0])
        U = sample_grf(ntest, grid, ell=ell)

    # --- Predict on all test samples ---
    xs = grid
    u_test = torch.tensor(U, dtype=torch.float32, device=device)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    x_t = grid_t.view(1, m, 1).repeat(u_test.size(0), 1, 1)

    with torch.no_grad():
        s_hat = model(u_test, x_t)  # (N, m)
        s_true = torch.cumsum(u_test, dim=1) * (xs[1] - xs[0])

    s_hat_np  = s_hat.cpu().numpy()
    s_true_np = s_true.cpu().numpy()
    U_np      = U
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
    import csv
    with open(per_metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","relL2","MAE","RMSE","MaxAbs","R2"])
        for i in range(N):
            w.writerow([i, rels[i], maes[i], rmses[i], maxabs[i], r2s[i]])
    print(f"Wrote per-sample metrics to: {per_metrics_path}")

    # --- Plots for a few examples (solution with points+error; plus u) ---
    k = min(args.num_examples, N)
    for i in range(k):
        y_true = s_true_np[i]
        y_pred = s_hat_np[i]
        err    = np.abs(y_pred - y_true)

        # Save the per-point table for this example
        table = np.column_stack([xs, y_true, y_pred, err])
        np.savetxt(os.path.join(outdir, f'example_{i:02d}_points.csv'),
                   table, delimiter=',', header='x,s_true,s_pred,abs_error', comments='')

        # Composite figure: left panel (curves + predicted points), right panel (pointwise |error|)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios':[2,1]})

        # left: true + pred curves, plus pred *points*
        axs[0].plot(xs, y_true, label='true s(x)')
        axs[0].plot(xs, y_pred, linestyle='--', label='pred s_hat(x)')
        axs[0].plot(xs, y_pred, marker='o', linestyle='None', markersize=3, label='pred points')
        axs[0].set_xlabel('x'); axs[0].set_ylabel('solution s'); axs[0].legend(loc='best')
        axs[0].set_title(f'Example {i}: solution & predicted points')

        # right: pointwise absolute error
        axs[1].plot(xs, err, marker='o', markersize=2)
        axs[1].set_xlabel('x'); axs[1].set_ylabel('|error|')
        axs[1].set_title('Pointwise absolute error')

        fig.tight_layout()
        out_path = os.path.join(outdir, f'example_{i:02d}_solution_and_error.png')
        plt.savefig(out_path, dpi=150); plt.close(fig)

        # Keep the u(x) plot too (as before)
        fig_u = plt.figure(figsize=(7, 3.0))
        axu = plt.gca()
        axu.plot(xs, U_np[i], label='u(x)')
        axu.set_xlabel('x'); axu.set_ylabel('u')
        axu.set_title(f'Example {i}: input u(x)')
        fig_u.tight_layout()
        out_path_u = os.path.join(outdir, f'example_{i:02d}_u.png')
        plt.savefig(out_path_u, dpi=150); plt.close(fig_u)

    print(f"Wrote {k*2} PNGs and {k} CSVs to: {outdir}")

if __name__ == "__main__":
    main()
