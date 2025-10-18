
"""
Train PINNs to solve 1D Poisson with/without Fourier-feature preconditioning and save logs.
Usage:
    python -m Models.PreconditioningAttempt.train --steps 2000 --width 64 --depth 3 --K 20 --outdir ./Models/PreconditioningAttempt/checkpoints
"""

import os, math, argparse, torch, numpy as np
from ..Shared.pinn_models import MLP, MLPConfig, fourier_features
from ..Shared.pinn_ops import PDEConfig, make_grid, forcing, u_of_x_builder
from ..Shared.pinn_train import train_model

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--K", type=int, default=20, help="Fourier features count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--outdir", type=str, default=os.path.join("Models", "PreconditioningAttempt", "checkpoints"))
    p.add_argument("--prefix", type=str, default="poisson")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = PDEConfig(N=2048, k=1.0, lam_bc=0.0, domain_left=-math.pi, domain_right=math.pi)
    x, w = make_grid(cfg, device=device)
    f_grid = forcing(x, cfg.k)

    # Plain MLP
    plain = MLP(MLPConfig(in_dim=1, out_dim=1, width=args.width, depth=args.depth)).to(device)
    losses_p, rel_p = train_model(plain, x, w, f_grid, cfg.k, featurizer=None,
                                  steps=args.steps, lr=args.lr,
                                  save_path=os.path.join(args.outdir, f"{args.prefix}_plain.pt"))

    # FF-MLP
    ff = MLP(MLPConfig(in_dim=2*args.K, out_dim=1, width=args.width, depth=args.depth)).to(device)
    losses_f, rel_f = train_model(ff, x, w, f_grid, cfg.k, featurizer=lambda xin: fourier_features(xin, args.K),
                                  steps=args.steps, lr=args.lr,
                                  save_path=os.path.join(args.outdir, f"{args.prefix}_ff.pt"))

    # Save curve numpy for quick plotting later
    np.save(os.path.join(args.outdir, f"{args.prefix}_loss_plain.npy"), losses_p)
    np.save(os.path.join(args.outdir, f"{args.prefix}_loss_ff.npy"), losses_f)
    np.save(os.path.join(args.outdir, f"{args.prefix}_rel_plain.npy"), rel_p)
    np.save(os.path.join(args.outdir, f"{args.prefix}_rel_ff.npy"), rel_f)

if __name__ == "__main__":
    main()
