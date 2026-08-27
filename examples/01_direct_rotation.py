"""Minimal reference example: U(x) = exp(-i x X)."""

import torch

from lpfn import DirectControls, GeneratorSet, TorchMatrixEngine


torch.set_default_dtype(torch.float64)

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
generators = GeneratorSet.from_matrices(
    [X], labels=["X"], convention="hermitian"
)

controls = DirectControls(
    lambda x: x[:, :1].reshape(-1, 1, 1),
    depth=1,
    num_generators=1,
)

x = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
theta = controls(x)
U = TorchMatrixEngine().execute(theta, generators)

print("theta shape:", theta.shape)
print("U shape:", U.shape)
print("U(0.5):\n", U[-1])

identity = torch.eye(2, dtype=torch.complex128)
unitarity_defect = torch.linalg.matrix_norm(U[-1].mH @ U[-1] - identity, ord=2)
print("unitarity defect:", float(unitarity_defect))
