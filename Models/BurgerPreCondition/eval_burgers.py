"""
Evaluation for Burgers experiments:
- Plots per-epoch physics loss curves.
- Plots time evolution at several slices:
    * time_evolution_grid.png  (rows = time slices, columns = FD | Plain | Precond)
    * frames_time/frame_XXX.png (per-slice side-by-side)
Optional: A-matrix spectrum (unchanged behavior).
"""

import os, json, argparse, numpy as np, torch, matplotlib.pyplot as plt
from glob import glob

from .burger_ops import BurgersConfig, make_grid_2d, A_matrix_linearized
from .burger_models import FeatureConfig, build_u_of_xt, fourier_features_2d
from .fdm_reference_burgers import solve_burgers_1d
from ..Shared.pinn_models import MLP, MLPConfig

torch.set_default_dtype(torch.float64)

def latest_run(outdir):
    runs = sorted(glob(os.path.join(outdir, "*") ))
    return runs[-1] if runs else None

def read_config(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="./Models/BurgerPreCondition/checkpoints")
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--compute-spectrum", action="store_true")
    p.add_argument("--skip-spectrum", action="store_true")
    p.add_argument("--spectral_Nx", type=int, default=128)
    p.add_argument("--spectral_Nt", type=int, default=64)
    p.add_argument("--num-slices", type=int, default=5, help="number of time slices in [0,T] to plot")
    args = p.parse_args()

    run_dir = args.run or latest_run(args.outdir)
    if run_dir is None:
        raise SystemExit("No run directory found. Train first or pass --run.")
    print("Using run:", run_dir)

    cfg_json = read_config(run_dir)
    phys = cfg_json["physics"]
    feats = cfg_json["features"]
    m_plain = cfg_json["model_plain"]
    m_ff = cfg_json["model_ff"]

    # Rebuild models
    plain = MLP(MLPConfig(in_dim=m_plain["in_dim"], width=m_plain["width"], depth=m_plain["depth"]))
    ff = MLP(MLPConfig(in_dim=m_ff["in_dim"], width=m_ff["width"], depth=m_ff["depth"]))

    # Load checkpoints (PyTorch>=2.6 safety)
    s_plain = torch.load(os.path.join(run_dir, "plain_last.pt"), map_location="cpu", weights_only=False)
    s_ff = torch.load(os.path.join(run_dir, "ff_last.pt"), map_location="cpu", weights_only=False)
    plain.load_state_dict(s_plain["model_state"])
    ff.load_state_dict(s_ff["model_state"])

    # Featurizer for FF model
    feat_cfg = FeatureConfig(Kx=feats["Kx"], Mt=feats["Mt"], T=phys["T"], nu=phys["nu"], hard_periodic=phys["hard_periodic"])
    def zmap(x,t):
        return fourier_features_2d(x,t,feat_cfg)

    # === Loss curves ===
    for tag in ["plain","ff"]:
        loss_path = os.path.join(run_dir, f"{tag}_losses.npy")
        if os.path.exists(loss_path):
            losses = np.load(loss_path)
        else:
            ck = torch.load(os.path.join(run_dir, f"{tag}_last.pt"), map_location="cpu", weights_only=False)
            losses = ck.get("losses", [])
        if len(losses):
            plt.plot(losses, label=f"{tag} loss")
    plt.yscale("log")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("physics loss")
    plt.title("Training loss (per-epoch)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=160)
    plt.close()

    # === Time evolution: FD vs Plain vs FF ===
    num_slices = max(2, int(args.num_slices))
    t_slices = np.linspace(0.0, phys["T"], num_slices)

    # FD reference series (dense Nx for display)
    Nx_ref = 1024
    x_ref, series_ref, t_rec = solve_burgers_1d(
        T=phys["T"], Nx=Nx_ref, Nt=2048, nu=phys["nu"],
        boundary=phys["boundary"], times=t_slices, return_series=True
    )

    x_torch = torch.tensor(x_ref, dtype=torch.get_default_dtype()).view(-1,1)

    # Prepare per-slice predictions
    preds_plain = []
    preds_ff = []

    for tt in t_slices:
        t_col = torch.full_like(x_torch, float(tt))
        with torch.no_grad():
            u_plain = build_u_of_xt(plain, None)(x_torch, t_col).cpu().numpy().ravel()
            u_ff    = build_u_of_xt(ff, zmap)(x_torch, t_col).cpu().numpy().ravel()
        preds_plain.append(u_plain)
        preds_ff.append(u_ff)

    preds_plain = np.stack(preds_plain, axis=0)     # [S, Nx]
    preds_ff    = np.stack(preds_ff, axis=0)        # [S, Nx]

    # Per-slice side-by-side frames
    frames_dir = os.path.join(run_dir, "frames_time")
    os.makedirs(frames_dir, exist_ok=True)
    for si, tt in enumerate(t_slices):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)
        axes[0].plot(x_ref, series_ref[si]); axes[0].set_title(f"FD  t={tt:.3f}")
        axes[1].plot(x_ref, preds_plain[si]); axes[1].set_title("Plain MLP")
        axes[2].plot(x_ref, preds_ff[si]); axes[2].set_title("Precond FF-MLP")
        for ax in axes:
            ax.set_xlabel("x")
        axes[0].set_ylabel("u(x,t)")
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f"frame_{si:03d}.png"), dpi=160)
        plt.close()

    # Grid montage: rows = time slices, cols = FD | Plain | Precond
    nrows, ncols = num_slices, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])  # ensure 2D

    for si, tt in enumerate(t_slices):
        axes[si,0].plot(x_ref, series_ref[si]); axes[si,0].set_ylabel("u(x,t)")
        axes[si,1].plot(x_ref, preds_plain[si])
        axes[si,2].plot(x_ref, preds_ff[si])
        axes[si,0].set_title(f"FD  t={tt:.3f}")
        axes[si,1].set_title("Plain MLP")
        axes[si,2].set_title("Precond FF-MLP")
        for j in range(ncols):
            axes[si,j].set_xlabel("x")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "time_evolution_grid.png"), dpi=160)
    plt.close()

    # Optional spectra (unchanged)
    if (not args.skip_spectrum) and args.compute_spectrum:
        print("Computing A spectra (linearized)... this can be slow.")
        bx = BurgersConfig(Nx=args.spectral_Nx, Nt=args.spectral_Nt,
                           a=phys["a"], b=phys["b"], T=phys["T"], nu=phys["nu"], device="cpu")
        x, t, w = make_grid_2d(bx)
        w = w / w.mean()
        A_plain = A_matrix_linearized(plain, x, t, w, bx, featurizer=None)
        A_ff = A_matrix_linearized(ff, x, t, w, bx, featurizer=lambda X,T: fourier_features_2d(X,T,feat_cfg))
        evals_plain = torch.linalg.eigvalsh(A_plain).cpu().numpy()
        evals_ff = torch.linalg.eigvalsh(A_ff).cpu().numpy()
        np.save(os.path.join(run_dir, "evals_plain.npy"), evals_plain)
        np.save(os.path.join(run_dir, "evals_ff.npy"), evals_ff)
        print("Saved eigenvalues to run dir.")

    print("Done. Artifacts saved in:", run_dir)

if __name__ == "__main__":
    main()
