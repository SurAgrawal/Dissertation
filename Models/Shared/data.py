# data.py
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky

def make_grid(m: int, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    """Uniform 1D grid [start, end] with m points."""
    return np.linspace(start, end, m, dtype=np.float64)

def rbf_cov(x: np.ndarray, ell: float) -> np.ndarray:
    """
    Squared-exponential (RBF) kernel matrix on a 1D grid x.
    k(x,x') = exp(-|x-x'|^2 / (2 ell^2))
    """
    x = x.reshape(-1, 1)
    d2 = (x - x.T) ** 2
    return np.exp(-0.5 * d2 / (ell ** 2))

def sample_grf(n: int, grid: np.ndarray, ell: float = 0.2, jitter: float = 1e-8) -> np.ndarray:
    """
    Sample n functions from a zero-mean Gaussian Random Field with RBF kernel on `grid`.
    Returns array of shape (n, m) where m = len(grid).
    """
    m = grid.size
    K = rbf_cov(grid, ell) + jitter * np.eye(m)
    L = cholesky(K, lower=True, overwrite_a=False, check_finite=True)  # K = L L^T
    Z = np.random.randn(m, n)
    U = (L @ Z).T  # (n, m)
    return U.astype(np.float32)

def antiderivative_batch(U: np.ndarray, dx: float) -> np.ndarray:
    """
    Simple left Riemann-sum antiderivative s(x) ~ sum_{k<=i} u_k * dx, batched over rows.
    U: (n, m) -> returns S: (n, m)
    """
    return np.cumsum(U, axis=1) * dx
