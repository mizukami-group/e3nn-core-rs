# Mathematical Conventions Specification

## Purpose

This project is convention-sensitive. A working implementation with mismatched signs, ordering, or normalization is not useful for transferring models. This document defines what must be pinned, tested, and never changed silently.

## Reference policy

- Pin the reference Python packages used for fixture generation.
- Store package versions, random seed, dtype, and convention metadata in every fixture.
- Do not assume an e3nn convention from memory. Verify by fixture.
- If a convention is changed, regenerate fixtures and update this document.

## O(3) irreps

An irrep is:

```text
(l, parity)
```

where:

```text
l >= 0
parity ∈ {even, odd}
dim(l) = 2l + 1
```

Text representation:

```text
0e, 1o, 2e, ...
```

Irreps direct sums use multiplicities:

```text
16x0e + 16x1o + 8x2e
```

## Layout

The layout must be explicit and tested.

Default layout principle:

```text
for each IrrepMul term in order:
  for mul index:
    store all components of that irrep contiguously
```

The exact component order within an irrep is convention-sensitive and must be pinned by reference fixtures. The implementation should expose layout metadata rather than relying on implicit indexing.

## Real spherical harmonics

Required dimensions:

```text
Y_l has dim 2l + 1
Y_0..Y_lmax total dim = (l_max + 1)^2
```

Implemented baseline:

```text
l_max <= 4
f64
```

Nequip-lite target:

```text
l_max <= 4
f64
```

Later optional path:

```text
f32
```

### `normalize_input`

This is separate from output normalization.

```text
normalize_input = true:
  u = r / ||r||
  Y = Y(u)

normalize_input = false:
  Y = Y(r)
```

Force VJP must account for `u = r / ||r||` when normalization is enabled.

### Output normalization

Represent supported modes explicitly:

```rust
pub enum ShNormalization {
    Component,
    Norm,
    Integral,
}
```

The numerical scaling for each mode must match pinned reference fixtures.

### Component ordering

Do not document a guessed `m` ordering as truth. Instead:

- Implement an internal order.
- Generate fixtures that identify output indices.
- Store convention metadata.
- Test against reference outputs for known vectors.

For `l_max <= 4`, `normalization="component"`, and `normalize_input=true`,
the e3nn `0.6.0` fixture pinned at commit
`2aa7f58440a06b15352a2cbce01fa4c26f824969` establishes the implemented
Cartesian component order, signs, normalization, and VJP behavior. The stable
fixture set includes:

```text
crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax2_component_normalized_f64.json
crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax3_component_normalized_f64.json
crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax4_component_normalized_f64.json
```

Let `u = r / ||r|| = (x, y, z)` after the documented near-zero denominator
policy. The `l <= 2` output is concatenated by `l` block:

```text
Y[0] = 1

Y[1] = sqrt(3) * x
Y[2] = sqrt(3) * y
Y[3] = sqrt(3) * z

Y[4] = sqrt(15) * x * z
Y[5] = sqrt(15) * x * y
Y[6] = sqrt(5)/2 * (2*y^2 - x^2 - z^2)
Y[7] = sqrt(15) * y * z
Y[8] = sqrt(15)/2 * (z^2 - x^2)
```

This is not documented as a general textbook `m=-l..l` order. The `l=3` and
`l=4` polynomial coefficients are implementation details derived from the same
pinned fixture convention. Do not rewrite or reorder those components from
memory or textbook translations. Other normalization modes remain unimplemented
until fixture-pinned.

## Wigner 3j / Clebsch-Gordan coefficients

The tensor product uses coefficients coupling:

```text
l1 ⊗ l2 -> l3
```

Allowed coupling must satisfy triangular rules:

```text
|l1 - l2| <= l3 <= l1 + l2
```

Parity selection:

```text
p_out = p1 * p2
```

where parity multiplication follows even/odd multiplication.

The exact sign and normalization of CG coefficients must match fixtures.

## TensorProduct path convention

For a path:

```text
out_k += path_weight * w_p * C_ijk * x_i * y_j
```

where:

- `C_ijk` is the sparse CG coefficient.
- `w_p` may be a dynamic edge weight from radial MLP.
- `path_weight` is deterministic and stored in the model/plan.

All path metadata must be serializable:

```text
in1 segment
in2 segment
out segment
cg entries
weight offset
path weight
```

The initial Rust `TensorProductPlan` expands e3nn-style `uvw` paths into
segment-level instructions. Each instruction references one input-1 segment, one
input-2 segment, one output segment, one optional dynamic `weight_offset`, and a
slice of explicit sparse CG entries whose component indices are local to those
segments. The pinned tensor-product fixture stores this expanded instruction
metadata alongside forward and VJP reference tensors.

## Dynamic versus static weights

Force inference requires:

- VJP wrt dynamic edge weights.
- No VJP wrt learned stored parameters.

If a weight is generated from distance through radial MLP, it is dynamic for the current edge and must receive a cotangent.


## Scope

This document describes the mathematical core. Model execution and model-specific contracts belong to the separate inference engine.
