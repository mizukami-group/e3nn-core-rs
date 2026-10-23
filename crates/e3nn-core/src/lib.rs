//! e3nn-core: framework-free equivariant primitives for force-producing MLIP inference.
//!
//! This crate intentionally does not implement training or a generic autograd engine.

pub mod activation;
pub mod error;
pub mod geometry;
pub mod irreps;
pub mod mlp;
pub mod normalization;
pub mod radial;
pub mod scatter;
pub mod sh;
pub mod so2;
pub mod so3;
pub mod tp;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum node/edge/row count above which eSEN/UMA elementwise primitives
/// (gate activation, radial envelope, layer-norm/SiLU, SO2 assembly) dispatch
/// to their rayon-parallel path.
///
/// The historical default was 512, which left small UMA structures (node/edge
/// counts in the low hundreds) on the serial path even with many threads
/// available. Lower the bound so those structures parallelize too. The chosen
/// path is numerically identical regardless of threshold (same arithmetic,
/// only the work partition differs). Override with
/// `LLMLIP_E3NN_PARALLEL_THRESHOLD`.
#[cfg(feature = "parallel")]
pub fn esen_parallel_min_nodes() -> usize {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("LLMLIP_E3NN_PARALLEL_THRESHOLD")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(64)
            .max(1)
    })
}

pub use error::{CoreError, Result};
pub use geometry::Vec3;
pub use irreps::{Irrep, IrrepMul, Irreps, LayoutSegment, Parity};
pub use mlp::{Activation, DenseLayer, FixedMlp, MlpCache};
pub use sh::{ShNormalization, SphericalHarmonicsPlan};
pub use tp::{CgEntry, TensorProductInstruction, TensorProductPlan};
