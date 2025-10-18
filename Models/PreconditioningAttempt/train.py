
"""
Train PINNs to solve 1D Poisson with/without Fourier-feature preconditioning and save logs in per-run folders.
Usage:
    python -m Models.PreconditioningAttempt.train --steps 2000 --width 64 --depth 3 --K 20 --outdir ./Models/PreconditioningAttempt/checkpoints
"""

import os, math, argparse, json, socket, time
import torch, numpy as np
from ..Shared.pinn_models import MLP, MLPConfig, fourier_features
from ..Shared.pinn_ops import PDEConfig, make_grid, forcing
from ..Shared.pinn_train import train_model

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_run_dir(outdir: str, tag: str = "precond"):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-{tag}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def save_config(run_dir: str, args: argparse.Namespace, extra: dict):
    cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "device": device,
        "args": vars(args),
        "extra": extra,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--K", type=int, default=20, help="Fourier features count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--outdir", type=str, default=os.path.join("Models", "PreconditioningAttempt", "checkpoints"))
    p.add_argument("--prefix", type=str, default="poisson")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = make_run_dir(args.outdir, tag="precond")
    os.makedirs(run_dir, exist_ok=True)

    cfg = PDEConfig(N=2048, k=1.0, lam_bc=0.0, domain_left=-math.pi, domain_right=math.pi)
    x, w = make_grid(cfg, device=device)
    f_grid = forcing(x, cfg.k)

    # Plain MLP
    plain = MLP(MLPConfig(in_dim=1, out_dim=1, width=args.width, depth=args.depth)).to(device)
    losses_p, rel_p = train_model(
        plain, x, w, f_grid, cfg.k, featurizer=None,
        steps=args.steps, lr=args.lr, save_path=os.path.join(run_dir, f"{args.prefix}_plain_last.pt"),
        progress=not args.no_progress
    )

    # FF-MLP
    ff = MLP(MLPConfig(in_dim=2*args.K, out_dim=1, width=args.width, depth=args.depth)).to(device)
    losses_f, rel_f = train_model(
        ff, x, w, f_grid, cfg.k, featurizer=lambda xin: fourier_features(xin, args.K),
        steps=args.steps, lr=args.lr, save_path=os.path.join(run_dir, f"{args.prefix}_ff_last.pt"),
        progress=not args.no_progress
    )

    # Save curves as numpy for quick plotting later
    np.save(os.path.join(run_dir, f"{args.prefix}_loss_plain.npy"), losses_p)
    np.save(os.path.join(run_dir, f"{args.prefix}_loss_ff.npy"),    losses_f)
    np.save(os.path.join(run_dir, f"{args.prefix}_rel_plain.npy"),  rel_p)
    np.save(os.path.join(run_dir, f"{args.prefix}_rel_ff.npy"),     rel_f)

    # Save a minimal run config
    extra = {
        "pde": dict(N=cfg.N, k=cfg.k, lam_bc=cfg.lam_bc, domain_left=cfg.domain_left, domain_right=cfg.domain_right),
        "model_plain": dict(in_dim=1, width=args.width, depth=args.depth),
        "model_ff": dict(in_dim=2*args.K, width=args.width, depth=args.depth, K=args.K),
        "optimizer": dict(lr=args.lr, steps=args.steps),
        "files": {
            "plain_ckpt": f"{args.prefix}_plain_last.pt",
            "ff_ckpt": f"{args.prefix}_ff_last.pt"
        }
    }
    save_config(run_dir, args, extra)

    print(f"Run directory created: {run_dir}")

if __name__ == "__main__":
    main()
