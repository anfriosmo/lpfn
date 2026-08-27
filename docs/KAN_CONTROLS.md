# KANControls design note

`KANControls` implements a Kolmogorov–Arnold-style scalar-control model while
preserving the LPFN contract

```text
x:     [batch, input_dim]
theta: [batch, K, r]
```

## Layer semantics

A KAN layer places a trainable univariate function on every edge and sums edge
values at output nodes. In this reference implementation each edge function is
an open-uniform B-spline expansion.

A direct one-layer model therefore has

```text
theta_j(x) = sum_i phi_{ij}(x_i).
```

For a one-hidden-layer model,

```text
h_q(x)     = mean_i phi^(1)_{iq}(x_i),
theta_j(x) = sum_q phi^(2)_{qj}(h_q(x)).
```

The mean in hidden layers is only a fixed rescaling of the KAN sum. Hidden edge
coefficients are mapped through `tanh`, so each hidden edge value lies in
`[-1,1]`; because B-splines form a partition of unity, hidden node values also
remain in `[-1,1]`. This avoids clipping or an adaptive hidden grid. The final
layer is unconstrained, since LPFN control angles need not be bounded.

## Parameter count

With `B` basis coefficients per edge, input width `d`, output width `m=K*r`,
and no hidden layer,

```text
#params = B * d * m.
```

With one hidden layer of width `h`,

```text
#params = B * (d*h + h*m).
```

There are no node biases: constants are already representable by the univariate
edge functions.

## Interpretability API

`forward_with_edges(x)` returns both `theta` and one tensor of edge
contributions per KAN layer. For a layer `n_in -> n_out`, the corresponding edge
tensor has shape

```text
[batch, n_in, n_out].
```

`effective_coefficients(layer_index)` exposes the actual coefficients defining
the learned edge functions. This is intended for pruning, plotting, symbolic
inspection, and the LP-QKAN interpretability experiments.

## Current benchmark search space

Milestone 09 uses the same hard parameter caps and train/validation/test splits
as the four-family pilot. Validation chooses among

```text
hidden_widths      in {direct, 1, 2}
num_basis_per_edge in {1, 2, 4}
spline_degree      in {0, 1, 3}
```

subject to the same parameter cap in every final benchmark cell.
