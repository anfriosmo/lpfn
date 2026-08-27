import torch

from lpfn.engines import TorchMatrixEngine
from lpfn.generators import GeneratorSet


def test_gradient_through_matrix_exponential_and_product(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([X, Z], labels=["X", "Z"], convention="hermitian")
    theta = torch.tensor([[[0.31, -0.27]]], dtype=torch.float64, requires_grad=True)
    out = TorchMatrixEngine().execute(theta, gs)

    # A real scalar loss, matching PyTorch's complex-autograd semantics.
    loss = (out.real.square() + out.imag.square() * 0.37).sum()
    grad = torch.autograd.grad(loss, theta)[0]

    assert grad.shape == theta.shape
    assert torch.isfinite(grad).all()
