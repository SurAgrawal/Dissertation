"""
ENGD optimization for a PINN solving 1D viscous Burgers' equation.

PDE (viscous Burgers):
    u_t + u * u_x - nu * u_xx = 0,     (x, t) in (0, 1) x (0, 1)

Boundary / initial conditions:
    u(x, 0)      = u0(x) = -sin(pi * x)          (initial condition)
    u(0, t)      = 0,                            (Dirichlet BC)
    u(1, t)      = 0.

We parametrize u(x, t) by an MLP and use energy natural gradient
descent (ENGD) with a Laplace-based Gramian + boundary Gramian,
similar in spirit to the Poisson example.

Coordinates:
    input to the network is z = [x, t], with x = z[0], t = z[1].
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import del_i, grid_line_search_factory
from ngrad.inner import model_laplace, model_identity
from ngrad.gram import gram_factory, nat_grad_factory
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------
# Problem setup
# --------------------------------------------------------------------

# viscosity (classic PINN example usually uses 0.01 / pi)
nu = 0.01 / jnp.pi

# random seed
seed = 0

# domains in (x, t) with x, t in [0, 1]
# Use Square for interior (space-time).
interior = Square(1.0)  # (x, t) in (0, 1) x (0, 1)

# Boundaries:
#   side 0: t = 0           -> initial condition u(x, 0) = u0(x)
#   side 1: x = 1           -> spatial BC
#   side 2: t = 1           -> (not constrained here, optional)
#   side 3: x = 0           -> spatial BC
# We'll use:
#   - SquareBoundary(a, 0)       for IC (bottom edge)
#   - SquareBoundary(a, [1,3])   for spatial BCs (right & left edges)
#   - SquareBoundary(a, slice(0,4)) for the Gram boundary term.

boundary_all = SquareBoundary(1.0, side_number=slice(0, 4))           # for Gram
boundary_ic = SquareBoundary(1.0, side_number=0)                      # t = 0
boundary_x = SquareBoundary(1.0, side_number=slice(1, 4, 2))          # x = 1 and x = 0

# integrators
# N_* control the density of collocation points.
interior_integrator = DeterministicIntegrator(interior, N=40)
boundary_gram_integrator = DeterministicIntegrator(boundary_all, N=40)
ic_integrator = DeterministicIntegrator(boundary_ic, N=40)
bc_integrator = DeterministicIntegrator(boundary_x, N=40)

# --------------------------------------------------------------------
# Model: u_theta(x, t)
# --------------------------------------------------------------------

activation = lambda x: jnp.tanh(x)
# input dimension 2 (x, t), some hidden layers, scalar output
layer_sizes = [2, 8, 8, 1]

params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))  # vectorized over inputs


# --------------------------------------------------------------------
# Initial condition u0(x)
# --------------------------------------------------------------------
@jit
def u0(z):
    """
    Initial condition u(x, 0) = -sin(pi * x).

    Input z is a vector [x, t], but only x = z[0] is used.
    """
    x = z[0]
    return -jnp.sin(jnp.pi * x)


# --------------------------------------------------------------------
# Burgers PDE residual
# --------------------------------------------------------------------

def burgers_residual(params, z):
    """
    PDE residual at a single point z = [x, t]:

        r(z) = u_t + u * u_x - nu * u_xx

    where u(x,t) = model(params, [x,t]).
    """

    # g: R^2 -> scalar
    def g(z_):
        return model(params, z_)

    # first derivatives using helper del_i:
    #   argnum = 0 -> derivative w.r.t x,
    #   argnum = 1 -> derivative w.r.t t.
    u_x_fn = del_i(g, argnum=0)
    u_t_fn = del_i(g, argnum=1)
    u_xx_fn = del_i(u_x_fn, argnum=0)  # second derivative in x

    u = g(z)
    u_x = u_x_fn(z)
    u_t = u_t_fn(z)
    u_xx = u_xx_fn(z)

    return u_t + u * u_x - nu * u_xx


def residual_sq(params, z):
    r = burgers_residual(params, z)
    return r ** 2


v_residual_sq = jit(vmap(residual_sq, (None, 0)))


# --------------------------------------------------------------------
# Loss terms: interior (PDE), initial condition, spatial boundary
# --------------------------------------------------------------------

@jit
def interior_loss(params):
    # Integrate residual^2 over interior of space-time.
    return interior_integrator(lambda x: v_residual_sq(params, x))


def ic_residual_sq(params, z):
    """
    Initial condition residual at z = [x, 0]:
        (u(x,0) - u0(x))^2
    """
    return (model(params, z) - u0(z)) ** 2


v_ic_residual_sq = jit(vmap(ic_residual_sq, (None, 0)))


@jit
def ic_loss(params):
    return ic_integrator(lambda x: v_ic_residual_sq(params, x))


def bc_residual_sq(params, z):
    """
    Spatial boundary condition:
        u(0, t) = 0, u(1, t) = 0  ->  (u(x_b, t))^2
    """
    return model(params, z) ** 2


v_bc_residual_sq = jit(vmap(bc_residual_sq, (None, 0)))


@jit
def bc_loss(params):
    return bc_integrator(lambda x: v_bc_residual_sq(params, x))


@jit
def total_loss(params):
    """
    Full PINN loss:
        L = L_pde + L_ic + L_bc
    """
    return interior_loss(params) + ic_loss(params) + bc_loss(params)


# --------------------------------------------------------------------
# Gramians & Energy Natural Gradient
# (Laplacian-based + boundary identity, similar to Poisson example)
# --------------------------------------------------------------------

# Gramian using Laplacian of the parameter-space tangent fields
gram_laplace = gram_factory(
    model=model,
    trafo=model_laplace,                 # uses Laplacian of del_theta u(z)
    integrator=interior_integrator,
)

# Gramian using identity transform at boundary (value-based)
gram_bdry = gram_factory(
    model=model,
    trafo=model_identity,
    integrator=boundary_gram_integrator,
)


@jit
def gram(params):
    # Combine interior and boundary Gramians
    return gram_laplace(params) + gram_bdry(params)


nat_grad = nat_grad_factory(gram)


# --------------------------------------------------------------------
# Line search for step sizes (same pattern as Poisson example)
# --------------------------------------------------------------------

grid = jnp.linspace(0, 30, 31)
steps = 0.5 ** grid
ls_update = grid_line_search_factory(total_loss, steps)


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------

def train(num_iterations=200):
    global params

    l = []
    i = []

    for iteration in range(num_iterations + 1):
        # gradient of total loss (parameter space)
        grads = grad(total_loss)(params)

        # natural gradient direction (preconditioned by Gramian)
        nat_grads = nat_grad(params, grads)

        # update parameters using line search along natural gradient
        params, actual_step = ls_update(params, nat_grads)

        if iteration % 10 == 0:
            loss_val = total_loss(params)
            print(
                f"Iter {iteration:4d} | "
                f"Loss = {loss_val:.3e} | "
                f"step = {float(actual_step):.3e}"
            )

            l.append(loss_val)
            i.append(iteration)

    # After training — plot
    plt.figure()
    plt.plot(i, l, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('L2 Error')
    plt.title('Training: L2 error over iterations')
    plt.yscale('log')  # optional: often error spans many orders of magnitude
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train(num_iterations=6000)
