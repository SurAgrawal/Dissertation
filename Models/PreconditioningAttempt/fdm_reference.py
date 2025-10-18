
"""
Finite Difference reference solver for 1D Poisson on (-pi, pi) with Dirichlet BCs.
Solves u'' + f = 0 with f = k^2 sin(kx) (matching the PINN target).
"""

import math
import numpy as np

def solve_poisson_1d(N=2048, k=1.0, a=-math.pi, b=math.pi):
    x = np.linspace(a, b, N)
    dx = (b - a) / (N - 1)
    f = (k**2) * np.sin(k * x)

    # Build tri-diagonal for second derivative: u'' ~ (u_{i-1} - 2u_i + u_{i+1})/dx^2
    main = -2.0 * np.ones(N-2)
    off = 1.0 * np.ones(N-3)
    A = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx**2)

    rhs = -f[1:-1]  # u'' + f = 0 -> A u_interior = -f_interior
    u_int = np.linalg.solve(A, rhs)

    u = np.zeros_like(x)
    u[1:-1] = u_int
    # boundaries are 0 (Dirichlet), consistent with sin(kx) if k integer
    return x, u
