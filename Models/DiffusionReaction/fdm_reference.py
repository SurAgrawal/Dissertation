# fdm_reference.py
from __future__ import annotations
import numpy as np

def solve_dr(u_on_grid: np.ndarray, x_grid: np.ndarray,
             T: float = 1.0, nt: int = 200, D: float = 0.01, k: float = 0.01) -> np.ndarray:
    """
    Solve s_t = D s_xx + k s^2 + u(x),  x in [0,1], t in [0,T],
    with s(x,0)=0, s(0,t)=s(1,t)=0 (Dirichlet).
    Returns s array of shape (m, nt) aligned with x_grid and uniform time grid.
    Scheme: IMEX: (I - dt*D*L) s^{n+1}_int = s^n_int + dt*(k*(s^n_int)^2 + u_int)
    """
    x = x_grid
    m = len(x)
    assert m >= 3, "Need at least 3 grid points for Dirichlet."
    dx = float(x[1]-x[0])
    dt = T / nt
    # Build Laplacian for interior nodes (1..m-2)
    r = D * dt / (dx*dx)
    main = (1 + 2*r) * np.ones(m-2)
    off  = (-r) * np.ones(m-3)
    # Tridiagonal A for implicit diffusion
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    # Pre-factorization via np.linalg.solve each step (m small); for speed, use scipy if desired
    s = np.zeros((m, nt), dtype=np.float64)  # includes boundaries (always zero)
    u_int = u_on_grid[1:-1].astype(np.float64)

    rhs = np.zeros(m-2, dtype=np.float64)
    s_int = np.zeros(m-2, dtype=np.float64)

    for n in range(nt-1):
        rhs[:] = s_int + dt*(k*(s_int**2) + u_int)
        s_int = np.linalg.solve(A, rhs)  # implicit step on interior
        s[1:-1, n+1] = s_int  # boundaries remain zero
    return s
