from __future__ import annotations

from itertools import product

import torch

from .base import GeneratorSet


_SINGLE_QUBIT = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}


def _kron_all(factors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    out = factors[0]
    for factor in factors[1:]:
        out = torch.kron(out, factor)
    return out


class PauliGeneratorSet(GeneratorSet):
    """Full n-qubit Pauli-string generator collection.

    With ``include_identity=False`` this gives the usual traceless Hermitian
    Pauli basis for su(2**n) after multiplication by -i.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        include_identity: bool = False,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str | None = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError("Pauli generators require complex64 or complex128")

        labels: list[str] = []
        matrices: list[torch.Tensor] = []
        locality: dict[str, int] = {}

        for symbols in product(("I", "X", "Y", "Z"), repeat=n_qubits):
            label = "".join(symbols)
            if not include_identity and all(symbol == "I" for symbol in symbols):
                continue
            matrix = _kron_all(tuple(_SINGLE_QUBIT[s] for s in symbols))
            labels.append(label)
            matrices.append(matrix.to(device=device, dtype=dtype))
            locality[label] = sum(symbol != "I" for symbol in symbols)

        stacked = torch.stack(matrices, dim=0)
        super().__init__(
            stacked,
            labels=labels,
            convention="hermitian",
            normalization="Pauli operator norm = 1",
            metadata={
                "family": "pauli_strings",
                "n_qubits": n_qubits,
                "include_identity": include_identity,
                "locality": locality,
            },
        )
