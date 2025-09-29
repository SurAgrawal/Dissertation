# eval.py
from __future__ import annotations
import os, argparse, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..Shared.models import DeepONet
from ..Shared.utils import set_seed, load_checkpoint

def parse_args():
    p = argparse.ArgumentParser("Evaluate physics-informed DeepONet on diffusion–reaction PDE")
    p.add_argument('--ckpt', required=True, help='path to best.pt or last.pt')
    p.add_argument('--nt', type=int, default=100, help='# time steps for visualization on [0,1]')
    p.add_argument('--num-examples', type=int, default=4, help='how many test inputs to plot')
    p.add_argument('--with-reference', action='store_true', help='compute FD reference & rel-L2')
    p.add_argument('--outdir', default='')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def pde_residual(model, u_row_t, xs_t, ts_t, device, D, K):
    """
    Compute PDE residual map for a single example on meshgrid (m, nt).
    Returns arrays s_hat(m, nt), res(m, nt), ic_err(m,), bc0_err(nt,), bc1_err(nt,)
    """
    m = xs_t.numel()
    nt = ts_t.numel()
    # Build full grid of (x,t)
    X = xs_t.view(m, 1).repeat(1, nt)
    T = ts_t.view(1, nt).repeat(m, 1)
    XT = torch.stack([X, T], dim=-1).unsqueeze(0).to(device)  # (1, m, nt, 2)
    XT = XT.view(1, m*nt, 2).requires_grad_(True)            # (1, m*nt, 2)

    s = model(u_row_t, XT).view(m, nt)                       # (m, nt)

    # first partials
    grads = torch.autograd.grad(s.sum(), XT, create_graph=True)[0].view(1, m, nt, 2)
    s_x = grads[..., 0].squeeze(0)                           # (m, nt)
    s_t = grads[..., 1].squeeze(0)
    # second x-derivative
    g2 = torch.autograd.grad(s_x.sum(), XT, create_graph=True)[0].view(1, m, nt, 2)
    s_xx = g2[..., 0].squeeze(0)

    # u(x) broadcast in time
    u_forcing = u_row_t.squeeze(0).view(m, 1).repeat(1, nt)  # (m, nt)

    res = s_t - D*s_xx - K*(s**2) - u_forcing

    # IC/BC errors
    ic_err = s[:, 0].abs()                 # s(x,0)=0
    bc0_err = s[0, :].abs()                # s(0,t)=0
    bc1_err = s[-1, :].abs()               # s(1,t)=0

    return s.detach().cpu().numpy(), res.detach().cpu().numpy(), \
           ic_err.detach().cpu().numpy(), bc0_err.detach().cpu().numpy(), bc1_err.detach().cpu().numpy()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load checkpoint & config
    ckpt = load_checkpoint(args.ckpt, device)
    cfg  = ckpt.get('cfg', {})
    m    = int(cfg.get('m', 100))
    width= int(cfg.get('width', 64))
    depth= int(cfg.get('depth', 5))
    feat = int(cfg.get('feat_dim', 64))
    act_name = cfg.get('activation', 'tanh')
    D = float(cfg.get('D', 0.01))
    K = float(cfg.get('k', 0.01))

    act_map = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU,
               'silu': torch.nn.SiLU, 'gelu': torch.nn.GELU, 'softplus': torch.nn.Softplus}
    act = act_map.get(act_name, torch.nn.Tanh)

    model = DeepONet(m=m, width=width, depth=depth, feat_dim=feat, act=act, in_dim=2).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # paths
    run_dir   = os.path.dirname(args.ckpt)
    ckpt_name = os.path.basename(args.ckpt).lower()
    label = 'best' if 'best' in ckpt_name else ('last' if 'last' in ckpt_name else 'ckpt')
    base_out  = args.outdir or run_dir
    outdir = os.path.join(base_out, f"eval-dr-{label}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    # load saved test set
    grid_path = os.path.join(run_dir, 'grid.npy')
    U_path    = os.path.join(run_dir, 'test_U.npy')
    assert os.path.isfile(grid_path) and os.path.isfile(U_path), "Missing grid.npy or test_U.npy"
    xs = np.load(grid_path)                      # (m,)
    U  = np.load(U_path)                         # (N, m)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    ts   = np.linspace(0.0, 1.0, args.nt)
    ts_t = torch.tensor(ts, dtype=torch.float32, device=device)

    # choose examples
    N = U.shape[0]
    k = min(args.num_examples, N)
    sel = np.linspace(0, N-1, k, dtype=int)

    # optional reference
    if args.with_reference:
        from Models.DiffusionReaction.fdm_reference import solve_dr

    # loop and plot
    for i, idx in enumerate(sel):
        u_row = U[idx:idx+1, :]                       # (1, m)
        u_row_t = torch.tensor(u_row, dtype=torch.float32, device=device)

        s_hat, res_map, ic_err, b0_err, b1_err = pde_residual(model, u_row_t, xs_t, ts_t, device, D, K)

        # heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        im0 = axs[0].imshow(s_hat.T, origin='lower', aspect='auto', extent=[xs[0], xs[-1], ts[0], ts[-1]])
        axs[0].set_title(f"Predicted s(x,t)  (example {idx})"); axs[0].set_xlabel('x'); axs[0].set_ylabel('t')
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(np.abs(res_map).T, origin='lower', aspect='auto', extent=[xs[0], xs[-1], ts[0], ts[-1]])
        axs[1].set_title("|PDE residual|"); axs[1].set_xlabel('x'); axs[1].set_ylabel('t')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.savefig(os.path.join(outdir, f"ex_{i:02d}_heatmaps.png"), dpi=150); plt.close(fig)

        # IBC errors
        fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3.0))
        ax2[0].plot(xs, ic_err);  ax2[0].set_title("IC |s(x,0)|")
        ax2[1].plot(ts, b0_err);  ax2[1].set_title("BC |s(0,t)|")
        ax2[2].plot(ts, b1_err);  ax2[2].set_title("BC |s(1,t)|")
        for a in ax2: a.grid(True, alpha=0.3)
        fig2.tight_layout()
        plt.savefig(os.path.join(outdir, f"ex_{i:02d}_ibc_errors.png"), dpi=150); plt.close(fig2)

        # optional reference & rel-L2 over the (x,t) grid
        if args.with_reference:
            s_ref = solve_dr(u_row[0], xs, T=1.0, nt=args.nt, D=D, k=K)   # (m, nt)

            # common scale for Ref & Pred
            vmin = float(min(s_ref.min(), s_hat.min()))
            vmax = float(max(s_ref.max(), s_hat.max()))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            fig4, axs4 = plt.subplots(1, 3, figsize=(16, 4.2))
            ax_ref, ax_pred, ax_err = axs4

            imA = ax_ref.imshow(s_ref.T, origin='lower', aspect='auto',
                                extent=[xs[0], xs[-1], ts[0], ts[-1]], norm=norm)
            ax_ref.set_title("Reference s(x,t)");
            ax_ref.set_xlabel('x');
            ax_ref.set_ylabel('t')

            imB = ax_pred.imshow(s_hat.T, origin='lower', aspect='auto',
                                 extent=[xs[0], xs[-1], ts[0], ts[-1]], norm=norm)
            ax_pred.set_title("Predicted ŝ(x,t)");
            ax_pred.set_xlabel('x');
            ax_pred.set_ylabel('t')

            imC = ax_err.imshow((abs(s_hat - s_ref)).T, origin='lower', aspect='auto',extent=[xs[0], xs[-1], ts[0], ts[-1]])
            ax_err.set_title("|Pred − Ref|");
            ax_err.set_xlabel('x');
            ax_err.set_ylabel('t')

            # 👉 Put the shared colorbar for Ref/Pred to the RIGHT of the Predicted axes,
            # with a larger pad so it sits further right.
            divider = make_axes_locatable(ax_pred)
            cax_pred = divider.append_axes("right", size="3%", pad=0.12)  # ↑ increase pad (try 0.10–0.16)
            cb_shared = fig4.colorbar(imA, cax=cax_pred)
            cb_shared.set_label("s(x,t)")

            # Keep (or move) the error bar as you like; here we keep it beside the error panel
            divider_err = make_axes_locatable(ax_err)
            cax_err = divider_err.append_axes("right", size="3%", pad=0.04)
            cb_err = fig4.colorbar(imC, cax=cax_err)
            cb_err.set_label("error")

            fig4.tight_layout()
            plt.savefig(os.path.join(outdir, f"ex_{i:02d}_ref_pred_same_scale.png"), dpi=150)
            plt.close(fig4)


    print(f"Wrote figures to: {outdir}")

if __name__ == "__main__":
    main()
