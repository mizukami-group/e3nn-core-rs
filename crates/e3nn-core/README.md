# e3nn-core

Framework-free Rust primitives for fixed-weight equivariant inference:
irreps, real spherical harmonics through l=4, tensor products, radial bases,
SO(2)/SO(3) operations, scatter, normalization, activations, and fixed MLPs.
Supported operations provide explicit input VJP/JVP kernels. There is no
training or generic automatic differentiation engine.

The default CPU path uses f64 and requires no Python or tensor framework.
Optional features: `parallel`, `serde`, and platform-specific BLAS providers.
Select one BLAS provider at a time. Rust 1.89 or newer is required.

```rust
use e3nn_core::SphericalHarmonicsPlan;
let plan = SphericalHarmonicsPlan::component(2, true)?;
let mut output = vec![0.0; plan.checked_output_dim()?];
plan.forward([0.1, -0.4, 0.9], &mut output)?;
# Ok::<(), e3nn_core::CoreError>(())
```

The API is pre-1.0; model-derived kernels remain experimental. Numerical
fixtures are bundled so `cargo test` needs no upstream checkout or Python.

[Repository and documentation](https://github.com/mizukami-group/e3nn-core-rs)

Licensed under MIT OR Apache-2.0; both license texts are included.

This is an independent project, not affiliated with or endorsed by e3nn.
Numerical references include [e3nn](https://github.com/e3nn/e3nn),
[MACE](https://github.com/ACEsuit/mace), and
[fairchem](https://github.com/FAIR-Chem/fairchem). See the bundled
[third-party notices](THIRD_PARTY_NOTICES.md) for scientific sources and licenses.
Binaries using `intel-mkl` must also comply with the license and notice
requirements supplied with Intel oneMKL (Intel Simplified Software License).
