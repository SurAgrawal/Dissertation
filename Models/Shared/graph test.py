import torch
from torchviz import make_dot  # helps visualize the gradient graph

# Define inputs that require gradients
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define computation
z = x**2 + 3*y

# Visualize the computational graph
make_dot(z, params={"x": x, "y": y})
