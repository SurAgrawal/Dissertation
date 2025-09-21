# utils.py
from __future__ import annotations
import os, time, random, json
import numpy as np
import torch

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_run_dir(save_dir: str, run_name: str = '') -> str:
    """
    Create a timestamped run directory.
    If /mnt/data exists, prefer it (useful on ephemeral machines).
    """
    base = save_dir
    if os.path.isdir('/mnt/data'):
        base = save_dir if save_dir.startswith('/mnt/data') else '/mnt/data/checkpoints'
    os.makedirs(base, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = f"{stamp}-{run_name}" if run_name else stamp
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_json(obj: dict, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_checkpoint(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer,
                    step: int, metric_relL2: float | None, cfg: dict):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "step": step,
        "metric_relL2": metric_relL2,
        "cfg": cfg,
    }, path)

def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)
