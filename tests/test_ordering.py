import torch

from lpfn.engines import TorchMatrixEngine
from lpfn.generators import GeneratorSet


def test_noncommutative_generator_order_is_locked(paulis):
    _, X, _, Z = paulis
    # Increasing a-order is [Z, X]. The engine left-multiplies each new factor,
    # so the final product is exp(-i x X) exp(-i y Z).
    gs = GeneratorSet.from_matrices([Z, X], labels=["Z", "X"], convention="hermitian")
    x = torch.tensor(0.43, dtype=torch.float64)
    y = torch.tensor(-0.61, dtype=torch.float64)
    theta = torch.tensor([[[y.item(), x.item()]]], dtype=torch.float64)

    out = TorchMatrixEngine().execute(theta, gs)[0]
    Qx = torch.linalg.matrix_exp((-1j * x).to(torch.complex128) * X)
    Qz = torch.linalg.matrix_exp((-1j * y).to(torch.complex128) * Z)
    expected = Qx @ Qz
    reversed_product = Qz @ Qx

    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(out, reversed_product, atol=1e-7, rtol=1e-7)


def test_block_order_is_locked(paulis):
    _, X, _, Z = paulis
    # One generator per block is represented using both generators with one
    # coefficient set to zero. Block k=0 acts before block k=1.
    gs = GeneratorSet.from_matrices([X, Z], labels=["X", "Z"], convention="hermitian")
    alpha = torch.tensor(0.22, dtype=torch.float64)
    beta = torch.tensor(0.73, dtype=torch.float64)
    theta = torch.tensor(
        [[[alpha.item(), 0.0], [0.0, beta.item()]]], dtype=torch.float64
    )

    out = TorchMatrixEngine().execute(theta, gs)[0]
    Qx = torch.linalg.matrix_exp((-1j * alpha).to(torch.complex128) * X)
    Qz = torch.linalg.matrix_exp((-1j * beta).to(torch.complex128) * Z)
    expected = Qz @ Qx

    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12)
