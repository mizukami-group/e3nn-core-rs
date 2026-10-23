<h1 align="center">e3nn-core-rs</h1>

<p align="center"><b>Framework-free E(3)-equivariant primitives in Rust, parity-pinned against e3nn (PyTorch)</b></p>

<p align="center">
  <a href="LICENSE-MIT"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue"></a>
  <img alt="Made with Rust" src="https://img.shields.io/badge/made%20with-Rust-orange">
  <img alt="Parity: e3nn 0.6.0" src="https://img.shields.io/badge/parity-e3nn%200.6.0%20(pinned)-informational">
  <img alt="unsafe: deny" src="https://img.shields.io/badge/unsafe__code-deny-brightgreen">
</p>

<p align="center">
  English | <a href="README.ja.md">日本語</a>
</p>

---

**e3nn-core-rs** is a framework-free Rust implementation of E(3)-equivariant
primitives — irreps, real spherical harmonics, tensor products, radial bases,
SO(2)/SO(3) operations, scatter, normalization, activations, and small fixed
MLPs — for force-producing machine-learning interatomic potential (MLIP)
inference. It implements **no training and no generic autograd engine**:
gradients are hand-written VJP/JVP kernels, and numerical correctness is
pinned against fixtures generated with [e3nn (PyTorch)](https://github.com/e3nn/e3nn) 0.6.0.

This is an independent project, not affiliated with or endorsed by the e3nn
project. The `e3nn-core` name denotes numerical compatibility targets, not
an official Rust edition of e3nn.

`llmlip` is a separate MLIP inference engine that uses these primitives.
Model loading and end-to-end model execution are outside this repository.

## ✨ Highlights

- 🦀 **Framework-free** — no PyTorch, no tensor framework; plain Rust over
  `&[f64]` slices.
- 🎯 **Parity-pinned** — outputs are tested against fixtures generated with
  e3nn (PyTorch) 0.6.0 (commit `2aa7f584`), with seeds, dtypes, and
  convention metadata recorded.
- ✍️ **Hand-written gradients** — supported operations provide explicit VJP/JVP
  kernels instead of a generic autograd.
- 🔒 **`unsafe_code = "deny"`** — pure-Rust implementation (the exceptions are isolated FFI modules behind optional BLAS
  features).

## 📦 Modules

Public modules of `crates/e3nn-core` (as in `src/lib.rs`):

| Module | Contents |
|---|---|
| `irreps` | O(3) irreducible representations (`Irrep` / `IrrepMul` / `Irreps` / `Parity`): parsing, dimensions, layout (`LayoutSegment`) |
| `sh` | real spherical harmonics (`SphericalHarmonicsPlan`, component normalization, `l_max <= 4`) with VJP/JVP |
| `tp` | tensor products (`TensorProductPlan` / `TensorProductInstruction` / `CgEntry`) |
| `radial` | radial bases and cutoff functions |
| `so2` / `so3` | SO(2)/SO(3) operations (eSEN-style SO(2) convolutions, Wigner rotations, …) |
| `scatter` | edge→node aggregation (scatter) |
| `normalization` | normalization kernels (RMS norm, …) |
| `activation` | gate and S2 activation kernels |
| `mlp` | fixed-weight MLPs (`FixedMlp` / `DenseLayer` / `Activation` / `MlpCache`) |
| `geometry` | `Vec3` (`[f64; 3]`) and edge-vector geometry utilities |
| `error` | `CoreError` / `Result` |

## 🧪 Design: parity via pinned fixtures

This project is convention-sensitive: a single flipped sign, component
order, or normalization makes ported model weights useless. Therefore:

- Fixtures generated with the reference implementation **e3nn (PyTorch)
  0.6.0** (commit `2aa7f584`) are pinned under
  `crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/`, and the Rust outputs are checked
  against them in FP64.
- Conventions are verified by fixtures, never by memory. Changing a
  convention means regenerating fixtures and updating
  [docs/MATH_CONVENTIONS.md](docs/MATH_CONVENTIONS.md).
- The Rust runtime has no Python dependency (Python is used only to
  generate fixtures).

See [docs/MATH_CONVENTIONS.md](docs/MATH_CONVENTIONS.md),
[docs/TESTING.md](docs/TESTING.md), and
[docs/DERIVATIVES.md](docs/DERIVATIVES.md).

## 🚀 Usage example

```rust
use e3nn_core::{Irreps, SphericalHarmonicsPlan, Vec3};

fn main() -> e3nn_core::Result<()> {
    // Parse irreps; dimensions and layout
    let irreps: Irreps = "16x0e + 16x1o + 8x2e".parse()?;
    assert_eq!(irreps.dim(), 16 + 48 + 40);

    // Real spherical harmonics (component normalization, l_max <= 4)
    let plan = SphericalHarmonicsPlan::component(2, true)?;
    let r: Vec3 = [0.1, -0.4, 0.9];
    let mut y = vec![0.0_f64; plan.checked_output_dim()?];
    plan.forward(r, &mut y)?;
    assert_eq!(y.len(), 9); // (l_max + 1)^2

    Ok(())
}
```

For gradients, use the matching VJP/JVP kernels such as
`SphericalHarmonicsPlan::vjp_edge_vector` / `jvp_edge_vector`. For tensor
products, build a plan with `TensorProductPlan::new` and evaluate with
`forward` / `forward_accumulate`.

## 🛠 Build and test

```bash
cargo build
cargo test            # includes fixture-parity tests (fixtures/ is bundled)
cargo bench           # criterion benches (radial / sh / tensor_product)
```

Cargo features (all disabled by default):

| feature | What it enables |
|---|---|
| `parallel` | rayon parallelism |
| `accelerate-blas` | Accelerate BLAS on macOS for some SO(3) paths (falls back to pure Rust elsewhere) |
| `openblas` | OpenBLAS on Linux x86_64 (`OPENBLAS_LIB_DIR` optionally sets the library path) |
| `intel-mkl` | Static sequential Intel oneMKL on Linux x86_64 (`MKLROOT`) |
| `armpl` | ArmPL provider marker for Linux aarch64; enables `openblas` call sites and requires downstream ArmPL linking; SO(3) uses the scalar fallback |
| `serde` | serde support for selected types |

Select one BLAS provider at a time. When distributing binaries built with
`intel-mkl`, include the notices and comply with the license supplied with
your Intel oneMKL installation; see the [Intel oneMKL license FAQ](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html)
(Intel Simplified Software License). The Rust source license does not replace
the licenses of linked native libraries.

## 🧷 API stability

This crate is pre-1.0 and the API is not yet stable. In particular, kernels
carrying model-derived names such as `fairchem_esen_*` / `mace_*` (inside
the `activation` / `normalization` / `radial` / `scatter` / `so2` / `so3`
modules) exist to reproduce the numerical behavior of specific model
architectures and are **experimental / unstable API**: no SemVer stability
is promised for them — renames, merges, and removals may happen in minor
versions.

## ⚖️ License

Licensed under either of **MIT** or **Apache-2.0** at your option
([LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE)).
This repository contains no pretrained model weights — only small numerical
reference fixtures (JSON) for testing.

## 👥 Contributors

This project was developed by:

- Wataru Mizukami ([@wmizukami](https://github.com/wmizukami))
- Do Tuan Minh ([@URPhysChem](https://github.com/URPhysChem))
- Kenji Ishihara ([@kenji-ishihara-os](https://github.com/kenji-ishihara-os))

## 🙏 Acknowledgements

We acknowledge [e3nn](https://github.com/e3nn/e3nn),
[MACE](https://github.com/ACEsuit/mace), and
[fairchem (eSEN/UMA)](https://github.com/FAIR-Chem/fairchem) as numerical
reference implementations. Radial kernels also use the ZBL screening model
(Ziegler, Biersack and Littmark, 1985), covalent radii from
[Cordero et al. (2008)](https://doi.org/10.1039/B801115J) as tabulated by
[ASE](https://ase.gitlab.io/ase/ase/data.html) and used in MACE, and MACE
Agnesi-transform conventions. See [third-party notices](THIRD_PARTY_NOTICES.md)
for source versions, scientific references, and upstream license texts.
These acknowledgements do not grant licenses to pretrained model weights.

This work is supported by JST COI-NEXT "Quantum Software Research Hub",
Grant Number **JPMJPF2014**.

## Local validation

Rust 1.89 or newer is required. No GitHub Actions workflows are provided.
Run `bash scripts/check.sh` locally; see [testing](docs/TESTING.md).
Linux-specific `openblas`, `intel-mkl`, and `armpl` features require their
native libraries and are not validated by the macOS checks. Select one BLAS
provider at a time.

External contributions are not currently accepted.
