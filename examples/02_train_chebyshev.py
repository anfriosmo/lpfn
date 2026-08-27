"""Train a Chebyshev-controlled LPFN on U(x)=exp(-i x X)."""

import torch

from lpfn import (
    ChebyshevControls,
    GeneratorSet,
    LieProductNetwork,
    PauliMatrixEngine,
    frobenius_loss,
    operator_error,
    unitarity_defect,
)

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
generators = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")
controls = ChebyshevControls(
    input_dim=1,
    depth=1,
    num_generators=1,
    degree=3,
    init_scale=0.02,
)
model = LieProductNetwork(
    generators=generators,
    controls=controls,
    engine=PauliMatrixEngine(),
)

x = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64).reshape(-1, 1)
target = torch.linalg.matrix_exp(
    (-1j * x[:, 0].to(torch.complex128))[:, None, None] * X
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

for _ in range(250):
    optimizer.zero_grad()
    pred = model(x)
    loss = frobenius_loss(pred, target)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred = model(x)
    print("final Frobenius loss:", float(frobenius_loss(pred, target)))
    print("max operator error:", float(operator_error(pred, target).max()))
    print("max unitarity defect:", float(unitarity_defect(pred).max()))
