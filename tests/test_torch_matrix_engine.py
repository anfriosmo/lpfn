import torch

from lpfn.engines import TorchMatrixEngine
from lpfn.generators import GeneratorSet


def test_zero_controls_return_base_point(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([X, Z], labels=["X", "Z"], convention="hermitian")
    alpha = torch.tensor(0.37, dtype=torch.float64)
    U0 = torch.linalg.matrix_exp((-1j * alpha).to(torch.complex128) * Z)
    theta = torch.zeros(4, 3, 2, dtype=torch.float64)
    out = TorchMatrixEngine().execute(theta, gs, base_point=U0)
    assert out.shape == (4, 2, 2)
    assert torch.allclose(out, U0.expand(4, -1, -1), atol=1e-12, rtol=1e-12)


def test_random_outputs_are_unitary(paulis):
    _, X, Y, Z = paulis
    gs = GeneratorSet.from_matrices([X, Y, Z], labels=["X", "Y", "Z"], convention="hermitian")
    torch.manual_seed(7)
    theta = torch.randn(12, 4, 3, dtype=torch.float64)
    out = TorchMatrixEngine().execute(theta, gs)
    eye = torch.eye(2, dtype=torch.complex128).expand(12, -1, -1)
    defect = out.mH @ out - eye
    assert torch.max(torch.linalg.matrix_norm(defect, ord=2, dim=(-2, -1))) < 1e-11


def test_single_x_generator_matches_exact_matrix_exponential(paulis):
    _, X, _, _ = paulis
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    theta = x.reshape(-1, 1, 1)
    out = TorchMatrixEngine().execute(theta, gs)
    exact = torch.linalg.matrix_exp((-1j * x.to(torch.complex128))[:, None, None] * X)
    assert torch.allclose(out, exact, atol=1e-12, rtol=1e-12)


def test_batched_execution_matches_external_loop(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([X, Z], labels=["X", "Z"], convention="hermitian")
    torch.manual_seed(3)
    theta = torch.randn(5, 2, 2, dtype=torch.float64)
    engine = TorchMatrixEngine()
    batched = engine.execute(theta, gs)
    looped = torch.cat([engine.execute(theta[j:j+1], gs) for j in range(theta.shape[0])], dim=0)
    assert torch.allclose(batched, looped, atol=5e-10, rtol=5e-10)


def test_base_point_is_on_the_right(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")
    theta = torch.tensor([[[0.41]]], dtype=torch.float64)
    beta = torch.tensor(0.29, dtype=torch.float64)
    U0 = torch.linalg.matrix_exp((-1j * beta).to(torch.complex128) * Z)
    out = TorchMatrixEngine().execute(theta, gs, base_point=U0)
    Q = torch.linalg.matrix_exp((-1j * theta[0, 0, 0]).to(torch.complex128) * X)
    expected = (Q @ U0).unsqueeze(0)
    wrong = (U0 @ Q).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(out, wrong, atol=1e-7, rtol=1e-7)
