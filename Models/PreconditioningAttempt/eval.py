
"""
Evaluation & figure generation for C.4-style analysis.
Generates:
  - Eigenvalue spectra of A at initialization for both models
  - Training curves (loss and rel-L2) if checkpoint logs exist
Usage:
    python -m Models.PreconditioningAttempt.eval --K 20 --width 64 --depth 3 --spectral_N 512 --outdir ./Models/PreconditioningAttempt/checkpoints
"""

import os, math, argparse, numpy as np, torch, matplotlib.pyplot as plt
from ..Shared.pinn_models import MLP, MLPConfig, fourier_features
from ..Shared.pinn_ops import PDEConfig, make_grid, forcing, A_matrix, u_of_x_builder
from ..Shared.plot_spectrum import plot_spectrum, plot_losses, plot_relerrs

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--spectral_N", type=int, default=512, help="Grid size for A-matrix (keep modest)")
    p.add_argument("--outdir", type=str, default=os.path.join("Models", "PreconditioningAttempt", "checkpoints"))
    p.add_argument("--prefix", type=str, default="poisson")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Smaller grid for spectral construction (A is expensive)
    cfg = PDEConfig(N=args.spectral_N, k=1.0, lam_bc=0.0, domain_left=-math.pi, domain_right=math.pi)
    x, w = make_grid(cfg, device=device)
    f_grid = forcing(x, cfg.k)

    # Instantiate models at initialization
    plain = MLP(MLPConfig(in_dim=1, out_dim=1, width=args.width, depth=args.depth)).to(device)
    ff = MLP(MLPConfig(in_dim=2*args.K, out_dim=1, width=args.width, depth=args.depth)).to(device)

    # Compute A and spectra (normalize by max eigenvalue)
    print("Building A for Plain MLP (may take time)...")
    A_p = A_matrix(plain, x, w, cfg.k, featurizer=None, lambda_bc=0.0)
    evals_p = torch.linalg.eigvalsh(A_p).clamp_min(0)
    evals_p = (evals_p / evals_p.max()).cpu().numpy()

    print("Building A for FF-MLP (may take time)...")
    A_f = A_matrix(ff, x, w, cfg.k, featurizer=lambda xin: fourier_features(xin, args.K), lambda_bc=0.0)
    evals_f = torch.linalg.eigvalsh(A_f).clamp_min(0)
    evals_f = (evals_f / evals_f.max()).cpu().numpy()

    # Plots
    plot_spectrum(evals_p, "Normalized eigenvalues of A (Plain MLP)")
    plot_spectrum(evals_f, "Normalized eigenvalues of A (FF-MLP)")

    # Training curves if available
    loss_p_path = os.path.join(args.outdir, f"{args.prefix}_loss_plain.npy")
    loss_f_path = os.path.join(args.outdir, f"{args.prefix}_loss_ff.npy")
    rel_p_path  = os.path.join(args.outdir, f"{args.prefix}_rel_plain.npy")
    rel_f_path  = os.path.join(args.outdir, f"{args.prefix}_rel_ff.npy")
    if all(os.path.exists(pth) for pth in [loss_p_path, loss_f_path, rel_p_path, rel_f_path]):
        loss_p = np.load(loss_p_path, allow_pickle=True)
        loss_f = np.load(loss_f_path, allow_pickle=True)
        rel_p  = np.load(rel_p_path, allow_pickle=True)
        rel_f  = np.load(rel_f_path, allow_pickle=True)
        plot_losses(loss_p, loss_f)
        plot_relerrs(rel_p, rel_f)
    else:
        print("Training logs not found; run the train script first to produce loss/rel-L2 curves.")

    # Save figures
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    for i, f in enumerate(plt.get_fignums(), start=1):
        plt.figure(f)
        outp = os.path.join(figdir, f"figure_{i:02d}.png")
        plt.savefig(outp, dpi=160, bbox_inches="tight")
        print("Saved", outp)

    print("Done.")

if __name__ == "__main__":
    main()
