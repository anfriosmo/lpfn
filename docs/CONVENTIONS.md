# LPFN conventions — v0.1 reference core

## Generator convention

A `GeneratorSet` must declare one of two conventions explicitly:

- `hermitian`: matrices are `H_a = H_a†` and the engine internally uses `B_a = -i H_a`;
- `skew_hermitian`: matrices are already `B_a` with `B_a† = -B_a`.

The library never guesses this convention.

## Tensor shapes

- inputs `x`: `[batch, input_dim]`;
- controls `theta`: `[batch, K, r]`;
- generators: `[r, N, N]`;
- matrix output: `[batch, N, N]`.

A batch of size one is never squeezed.

## Product ordering

Flatten the factors by increasing block index `k`, then increasing generator
index `a`, and call them `Q_1, ..., Q_M`. The reference engine returns

`U = Q_M ... Q_2 Q_1 U_0`.

Therefore `Q_1` acts first. This convention is locked by non-commutative tests.

## Numerical reference precision

Mathematical validation uses `float64` controls and `complex128` matrices where
possible. Lower precision is an experimental/performance decision, not the
reference semantics.
