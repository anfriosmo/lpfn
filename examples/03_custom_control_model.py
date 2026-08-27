"""Minimal custom ControlModel example."""

import torch
from lpfn import ControlModel, LieProductNetwork, PauliGeneratorSet, PauliMatrixEngine


class LinearControls(ControlModel):
    def __init__(self, input_dim: int, depth: int, num_generators: int) -> None:
        super().__init__(depth=depth, num_generators=num_generators)
        self.linear = torch.nn.Linear(input_dim, depth * num_generators, bias=True, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.linear(x).reshape(x.shape[0], self.depth, self.num_generators)
        return self.validate_output(theta, batch_size=x.shape[0])


generators = PauliGeneratorSet(1)
controls = LinearControls(input_dim=1, depth=1, num_generators=generators.num_generators)
model = LieProductNetwork(generators=generators, controls=controls, engine=PauliMatrixEngine())

x = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
print(model(x).shape)
