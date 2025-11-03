
"""
Train PINNs for viscous Burgers with/without preconditioned Fourier features.
Saves logs and checkpoints in a run directory.
"""

import os, json, time, argparse, socket
from tqdm import trange
import torch
import numpy as np
from dataclasses import asdict

from ..Shared.pinn_models import MLP, MLPConfig
from .burger_models import FeatureConfig, fourier_features_2d, build_u_of_xt
from .burger_ops import BurgersConfig, make_grid_2d, physics_loss, l2_error_vs_reference
from .fdm_reference_burgers import solve_burgers_1d

torch.set_default_dtype(torch.float64)

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def run_dir_name(prefix):
    ts = time.strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split(".")[0]
    return f"{prefix}-{ts}-{host}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="./Models/BurgerPreCondition/checkpoints")
    p.add_argument("--prefix", type=str, default="burgers")

    # physics & grids
    p.add_argument("--nu", type=float, default=1e-2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--Nx", type=int, default=256)
    p.add_argument("--Nt", type=int, default=128)

    # boundaries
    p.add_argument("--boundary", type=str, default="periodic", choices=["periodic","dirichlet","neumann"])
    p.add_argument("--hard-periodic", action="store_true", default=True)

    # model sizes
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)

    # features
    p.add_argument("--Kx", type=int, default=24)
    p.add_argument("--Mt", type=int, default=24)

    # optim
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    torch.manual_seed(args.seed)

    # Build configs
    cfg = BurgersConfig(Nx=args.Nx, Nt=args.Nt, nu=args.nu, T=args.T,
                        boundary=args.boundary, hard_periodic=args.hard_periodic,
                        device=device)
    feat_cfg = FeatureConfig(Kx=args.Kx, Mt=args.Mt, T=args.T, nu=args.nu,
                             hard_periodic=(args.boundary=="periodic" and args.hard_periodic))

    # Models
    plain = MLP(MLPConfig(in_dim=2, width=args.width, depth=args.depth)).to(device)
    ff = MLP(MLPConfig(in_dim=2*args.Kx*args.Mt, width=args.width, depth=args.depth)).to(device)

    # Featurizer closure
    def zmap(x,t):
        return fourier_features_2d(x,t,feat_cfg)

    u_plain = build_u_of_xt(plain, featurizer=None)
    u_ff = build_u_of_xt(ff, featurizer=zmap)

    # FD reference on a denser eval grid
    x_ref, u_ref_T = solve_burgers_1d(T=args.T, Nx=1024, Nt=2048, nu=args.nu, boundary=args.boundary)
    # Build eval grid matching x_ref at final time
    x_eval = torch.tensor(x_ref, device=device, dtype=torch.get_default_dtype()).view(-1,1)
    t_eval = torch.full_like(x_eval, args.T)
    w_eval = torch.full_like(x_eval, (x_ref[1]-x_ref[0]))
    u_ref_torch = torch.tensor(u_ref_T, device=device, dtype=torch.get_default_dtype()).view(-1,1)

    # Optims
    opt_plain = torch.optim.Adam(plain.parameters(), lr=args.lr)
    opt_ff = torch.optim.Adam(ff.parameters(), lr=args.lr)

    # Run directory
    run_dir = os.path.join(args.outdir, run_dir_name(args.prefix))
    os.makedirs(run_dir, exist_ok=True)

    save_json({
        "physics": asdict(cfg),
        "features": asdict(feat_cfg),
        "model_plain": {"in_dim":2, "width":args.width, "depth":args.depth},
        "model_ff": {"in_dim":2*args.Kx*args.Mt, "width":args.width, "depth":args.depth, "Kx":args.Kx, "Mt":args.Mt},
        "optimizer": {"lr": args.lr, "epochs": args.epochs},
    }, os.path.join(run_dir, "config.json"))

    # Training loop (simple shared loop for both models)
    def train_one(model, u_map, opt, tag):
        # Track per-epoch loss, and rel-L2 at a stride (every 50 epochs + first)
        losses_all = []
        rel_full = [float('nan')] * args.epochs
        bar = trange(args.epochs, desc=f"{tag}", dynamic_ncols=True)
        last_rel = None
        for epoch in bar:
            opt.zero_grad()
            loss = physics_loss(u_map, cfg)
            loss.backward()
            opt.step()

            # record loss every epoch
            losses_all.append(float(loss))

            # compute rel-L2 on a stride (cheap but informative)
            if (epoch+1) % 50 == 0 or epoch == 0:
                with torch.no_grad():
                    rel = l2_error_vs_reference(u_map, x_eval, t_eval, w_eval, u_ref_torch)
                rel_full[epoch] = float(rel)
                last_rel = float(rel)

            bar.set_postfix(loss=float(loss), rel=last_rel if last_rel is not None else float('nan'))
        # save ckpt + npy artifacts for easy plotting later
        losses_arr = np.array(losses_all)
        rel_arr = np.array(rel_full)
        state = {"model_state": model.state_dict(), "losses": losses_arr, "relerrs": rel_arr}
        torch.save(state, os.path.join(run_dir, f"{tag}_last.pt"))
        np.save(os.path.join(run_dir, f"{tag}_losses.npy"), losses_arr)
        np.save(os.path.join(run_dir, f"{tag}_relerrs.npy"), rel_arr)
        return losses_all, rel_full

    print("Training plain MLP...")
    train_one(plain, u_plain, opt_plain, tag="plain")

    print("Training preconditioned FF-MLP...")
    train_one(ff, u_ff, opt_ff, tag="ff")

    print("Run directory:", run_dir)

if __name__ == "__main__":
    main()
