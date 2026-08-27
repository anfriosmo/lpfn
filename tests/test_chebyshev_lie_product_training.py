import torch

from lpfn import ChebyshevControls, LieProductNetwork
from lpfn.engines import PauliMatrixEngine
from lpfn.generators import GeneratorSet


def test_chebyshev_lie_product_can_fit_single_x_rotation(paulis):
    _, X, _, _ = paulis
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")
    controls = ChebyshevControls(
        input_dim=1,
        depth=1,
        num_generators=1,
        degree=3,
        init_scale=0.02,
    )
    model = LieProductNetwork(
        generators=gs,
        controls=controls,
        engine=PauliMatrixEngine(),
    )

    x = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64).reshape(-1, 1)
    target = torch.linalg.matrix_exp(
        (-1j * x[:, 0].to(torch.complex128))[:, None, None] * X
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    initial = None
    for step in range(250):
        optimizer.zero_grad()
        pred = model(x)
        loss = (pred - target).abs().square().mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()

    final = float(((model(x) - target).abs().square().mean()).detach())
    assert initial is not None
    assert final < 1e-8
    assert final < initial * 1e-5
