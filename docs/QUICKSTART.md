# Quick start

## Install

```bash
pip install lpfn
```

For development:

```bash
pip install -e ".[dev]"
```

## Build a one-qubit LPFN

```python
import torch
from lpfn import (
    ChebyshevControls,
    LieProductNetwork,
    PauliGeneratorSet,
    PauliMatrixEngine,
)

generators = PauliGeneratorSet(1)
controls = ChebyshevControls(
    input_dim=1,
    depth=1,
    num_generators=generators.num_generators,
    degree=2,
)
model = LieProductNetwork(generators, controls, PauliMatrixEngine())

x = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).reshape(-1, 1)
U = model(x)
assert U.shape == (32, 2, 2)
```

## Swap the scalar control family

The Lie geometry does not change:

```python
from lpfn import FourierControls

controls = FourierControls(
    input_dim=1,
    depth=1,
    num_generators=generators.num_generators,
    max_frequency=2,
)
model = LieProductNetwork(generators, controls, PauliMatrixEngine())
```

## Swap the execution engine

```python
from lpfn import TorchMatrixEngine
model = LieProductNetwork(generators, controls, TorchMatrixEngine())
```

`TorchMatrixEngine` is the reference semantics. Specialized or future circuit
engines should reproduce it on small validation problems.
