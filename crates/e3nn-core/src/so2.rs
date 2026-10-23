//! SO2 feature primitives used by fixed-weight equivariant models.

use std::cell::RefCell;

use crate::so3::fairchem_esen_l_to_m_permutation;
use crate::so3::{
    fairchem_esen_l_to_m_forward, fairchem_esen_m_to_l_forward, EsenCoefficientMappingConfig,
};
use crate::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const L2_M2_COEFFICIENT_COUNT: usize = 9;
#[allow(dead_code)]
const L2_M2_L_TO_M_PERMUTATION: [usize; 9] = [0, 2, 6, 3, 7, 1, 5, 8, 4];

/// Minimum node (edge) count above which SO2 assembly uses the rayon
/// l-major-direct path instead of the serial m-major + permutation path.
///
/// Small UMA structures have edge counts in the low hundreds, well under the
/// historical 512 default, which forced them onto the serial path (extra
/// l<->m permutation copies, no edge-parallelism). Override with
/// `LLMLIP_SO2_PARALLEL_THRESHOLD` for tuning.
#[allow(dead_code)]
fn so2_parallel_node_threshold() -> usize {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("LLMLIP_SO2_PARALLEL_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1)
    })
}

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(dead_code)]
fn so2_use_parallel_compact_input(nodes: usize) -> bool {
    rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold()
}

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(dead_code)]
fn so2_use_serial_compact_input(nodes: usize) -> bool {
    !so2_use_parallel_compact_input(nodes)
}
const L2_M2_ORDER0_L_MAJOR_SOURCES: [usize; 3] = [0, 2, 6];
const L2_M2_ORDER1_L_MAJOR_SOURCES: [usize; 4] = [3, 7, 1, 5];
const L2_M2_ORDER2_L_MAJOR_SOURCES: [usize; 2] = [8, 4];

thread_local! {
    static L2_M2_RADIAL_GATE_CONV2_FORWARD_SCRATCH: RefCell<L2M2RadialGateConv2ForwardScratch> =
        RefCell::new(L2M2RadialGateConv2ForwardScratch::default());
    static L2_M2_RADIAL_GATE_CONV2_VJP_SCRATCH: RefCell<L2M2RadialGateConv2VjpScratch> =
        RefCell::new(L2M2RadialGateConv2VjpScratch::default());
}

#[derive(Default)]
struct L2M2RadialGateConv2ForwardScratch {
    conv1_m0_input: Vec<f64>,
    conv1_m0_dense_out: Vec<f64>,
    conv2_m0_input: Vec<f64>,
    conv2_m0_out: Vec<f64>,
    conv1_block_input: Vec<f64>,
    conv1_block_out: Vec<f64>,
    conv2_block_input: Vec<f64>,
    conv2_block_out: Vec<f64>,
}

#[derive(Default)]
struct L2M2RadialGateConv2VjpScratch {
    conv1_m0_full_grad_out: Vec<f64>,
    conv2_m0_grad_out: Vec<f64>,
    activated_m0_grad: Vec<f64>,
    conv2_block_grad_out: Vec<f64>,
    activated_block_grad: Vec<f64>,
    conv1_block_grad_out: Vec<f64>,
    conv1_block_grad_modulated: Vec<f64>,
    conv1_m0_grad_modulated: Vec<f64>,
}

#[allow(dead_code)]
fn l2_m2_l_to_m_permutation(l_max: usize, m_max: usize) -> Option<&'static [usize]> {
    (l_max == 2 && m_max == 2).then_some(&L2_M2_L_TO_M_PERMUTATION)
}

#[allow(dead_code)]
fn l2_m2_l_major_sources(
    coefficient_count: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
) -> Option<&'static [usize]> {
    if coefficient_count != L2_M2_COEFFICIENT_COUNT {
        return None;
    }
    match (coefficient_offset, block_coefficients) {
        (0, 3) => Some(&L2_M2_ORDER0_L_MAJOR_SOURCES),
        (3, 4) => Some(&L2_M2_ORDER1_L_MAJOR_SOURCES),
        (7, 2) => Some(&L2_M2_ORDER2_L_MAJOR_SOURCES),
        _ => None,
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, unsafe_code)]
mod accelerate_blas {
    use std::{
        cell::RefCell,
        os::raw::{c_double, c_int},
    };

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    thread_local! {
        static SO2_MCONV_FUSED_WEIGHT_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    }

    fn build_so2_mconv_fused_weight_into(
        weight_out_by_in: &[f64],
        in_dim: usize,
        out_dim: usize,
        fused_weight: &mut Vec<f64>,
    ) {
        let fused_in_dim = 2 * in_dim;
        let fused_out_dim = 2 * out_dim;
        fused_weight.clear();
        fused_weight.resize(fused_out_dim * fused_in_dim, 0.0);
        for out_index in 0..out_dim {
            let weight_real_start = out_index * in_dim;
            let weight_imag_start = (out_dim + out_index) * in_dim;
            let fused_real_start = out_index * fused_in_dim;
            let fused_imag_start = (out_dim + out_index) * fused_in_dim;
            for in_index in 0..in_dim {
                let real_weight = weight_out_by_in[weight_real_start + in_index];
                let imag_weight = weight_out_by_in[weight_imag_start + in_index];
                fused_weight[fused_real_start + in_index] = real_weight;
                fused_weight[fused_real_start + in_dim + in_index] = -imag_weight;
                fused_weight[fused_imag_start + in_index] = imag_weight;
                fused_weight[fused_imag_start + in_dim + in_index] = real_weight;
            }
        }
    }

    pub(super) fn build_so2_mconv_fused_weight(
        weight_out_by_in: &[f64],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f64> {
        let mut fused_weight = Vec::new();
        build_so2_mconv_fused_weight_into(weight_out_by_in, in_dim, out_dim, &mut fused_weight);
        fused_weight
    }

    fn with_so2_mconv_fused_weight<T>(
        weight_out_by_in: &[f64],
        in_dim: usize,
        out_dim: usize,
        f: impl FnOnce(&[f64]) -> T,
    ) -> T {
        #[cfg(not(feature = "parallel"))]
        {
            let fused_weight = build_so2_mconv_fused_weight(weight_out_by_in, in_dim, out_dim);
            return f(&fused_weight);
        }

        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() <= 1 {
            let fused_weight = build_so2_mconv_fused_weight(weight_out_by_in, in_dim, out_dim);
            return f(&fused_weight);
        }

        #[allow(unreachable_code)]
        SO2_MCONV_FUSED_WEIGHT_SCRATCH.with(|scratch| {
            let mut fused_weight = scratch.borrow_mut();
            build_so2_mconv_fused_weight_into(weight_out_by_in, in_dim, out_dim, &mut fused_weight);
            f(&fused_weight)
        })
    }

    #[cfg_attr(
        all(feature = "accelerate-blas", target_os = "macos"),
        link(name = "Accelerate", kind = "framework")
    )]
    #[cfg_attr(
        all(feature = "openblas", not(feature = "armpl"), target_os = "linux"),
        link(name = "openblas")
    )]
    #[cfg_attr(
        all(feature = "armpl", target_os = "linux"),
        link(name = "armpl_lp64_mp")
    )]
    #[cfg_attr(all(feature = "armpl", target_os = "linux"), link(name = "gomp"))]
    extern "C" {
        fn cblas_dgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
    }

    pub fn dense_forward_transposed_weight(
        input: &[f64],
        rows: usize,
        in_dim: usize,
        weight_out_by_in: &[f64],
        out_dim: usize,
        out: &mut [f64],
    ) {
        unsafe {
            // Invariant: all slices are validated by the caller as row-major
            // [rows, in_dim], [out_dim, in_dim], and [rows, out_dim].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                rows as c_int,
                out_dim as c_int,
                in_dim as c_int,
                1.0,
                input.as_ptr(),
                in_dim as c_int,
                weight_out_by_in.as_ptr(),
                in_dim as c_int,
                0.0,
                out.as_mut_ptr(),
                out_dim as c_int,
            );
        }
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub fn dense_forward_strided_input_transposed_weight_add(
        input: &[f64],
        rows: usize,
        input_row_stride: usize,
        input_feature_offset: usize,
        in_dim: usize,
        weight_out_by_in: &[f64],
        weight_row_stride: usize,
        out_dim: usize,
        beta: f64,
        out: &mut [f64],
    ) {
        unsafe {
            // Invariant: `input` is row-major with row stride
            // `input_row_stride`; the selected feature block has length
            // `in_dim`. `weight_out_by_in` is row-major [out_dim,
            // weight_row_stride], and the first `in_dim` values at each row
            // are the selected transposed-weight block.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                rows as c_int,
                out_dim as c_int,
                in_dim as c_int,
                1.0,
                input.as_ptr().add(input_feature_offset),
                input_row_stride as c_int,
                weight_out_by_in.as_ptr(),
                weight_row_stride as c_int,
                beta,
                out.as_mut_ptr(),
                out_dim as c_int,
            );
        }
    }

    pub fn dense_vjp_input_add(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
    ) {
        unsafe {
            // Invariant: computes grad_out [rows, out_dim] * weight
            // [out_dim, in_dim] and accumulates into grad_input.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_out.as_ptr(),
                out_dim as c_int,
                weight_out_by_in.as_ptr(),
                in_dim as c_int,
                1.0,
                grad_input.as_mut_ptr(),
                in_dim as c_int,
            );
        }
    }

    #[allow(dead_code)]
    pub fn dense_vjp_input_set(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
    ) {
        unsafe {
            // Invariant: computes grad_out [rows, out_dim] * weight
            // [out_dim, in_dim] and overwrites grad_input.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_out.as_ptr(),
                out_dim as c_int,
                weight_out_by_in.as_ptr(),
                in_dim as c_int,
                0.0,
                grad_input.as_mut_ptr(),
                in_dim as c_int,
            );
        }
    }

    pub fn so2_mconv_forward_fused_weight(
        input: &[f64],
        rows: usize,
        in_dim: usize,
        weight_out_by_in: &[f64],
        out_dim: usize,
        out: &mut [f64],
    ) {
        with_so2_mconv_fused_weight(weight_out_by_in, in_dim, out_dim, |fused_weight| {
            so2_mconv_forward_precomputed_fused_weight(
                input,
                rows,
                in_dim,
                fused_weight,
                out_dim,
                out,
            );
        });
    }

    pub fn so2_mconv_forward_precomputed_fused_weight(
        input: &[f64],
        rows: usize,
        in_dim: usize,
        fused_weight: &[f64],
        out_dim: usize,
        out: &mut [f64],
    ) {
        let fused_in_dim = 2 * in_dim;
        let fused_out_dim = 2 * out_dim;
        unsafe {
            // Invariant: input is row-major [rows, 2 * in_dim]. The fused
            // row-major weight [2 * out_dim, 2 * in_dim] encodes:
            // real = input_real * W0 - input_imag * W1
            // imag = input_real * W1 + input_imag * W0.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                rows as c_int,
                fused_out_dim as c_int,
                fused_in_dim as c_int,
                1.0,
                input.as_ptr(),
                fused_in_dim as c_int,
                fused_weight.as_ptr(),
                fused_in_dim as c_int,
                0.0,
                out.as_mut_ptr(),
                fused_out_dim as c_int,
            );
        }
    }

    pub fn so2_mconv_vjp_input_add(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
        grad_input_row_stride: usize,
    ) {
        // This is a single early-return dispatch: the 512 cutoff selects when
        // precomputing the fused BLAS matrix pays for itself, not a paired
        // parallel/scalar guard.
        if rows >= 512 {
            so2_mconv_vjp_input_add_fused(
                grad_out,
                rows,
                out_dim,
                weight_out_by_in,
                in_dim,
                grad_input,
                grad_input_row_stride,
            );
            return;
        }

        unsafe {
            // Invariant: grad_out is row-major [rows, 2 * out_dim] with real
            // and imaginary halves. weight is row-major [2 * out_dim, in_dim].
            // grad_input is row-major [rows, 2 * in_dim] or a compatible
            // strided view. This computes:
            // real = grad_real * W0 + grad_imag * W1
            // imag = grad_imag * W0 - grad_real * W1
            let grad_real = grad_out.as_ptr();
            let grad_imag = grad_out.as_ptr().add(out_dim);
            let weight_real = weight_out_by_in.as_ptr();
            let weight_imag = weight_out_by_in.as_ptr().add(out_dim * in_dim);
            let grad_input_real = grad_input.as_mut_ptr();
            let grad_input_imag = grad_input.as_mut_ptr().add(in_dim);
            let grad_out_row_stride = 2 * out_dim;

            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_real,
                grad_out_row_stride as c_int,
                weight_real,
                in_dim as c_int,
                1.0,
                grad_input_real,
                grad_input_row_stride as c_int,
            );
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_imag,
                grad_out_row_stride as c_int,
                weight_imag,
                in_dim as c_int,
                1.0,
                grad_input_real,
                grad_input_row_stride as c_int,
            );
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_imag,
                grad_out_row_stride as c_int,
                weight_real,
                in_dim as c_int,
                1.0,
                grad_input_imag,
                grad_input_row_stride as c_int,
            );
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                -1.0,
                grad_real,
                grad_out_row_stride as c_int,
                weight_imag,
                in_dim as c_int,
                1.0,
                grad_input_imag,
                grad_input_row_stride as c_int,
            );
        }
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub fn dense_vjp_input_strided_grad_out_add(
        grad_out: &[f64],
        rows: usize,
        grad_out_row_stride: usize,
        grad_out_feature_offset: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        weight_row_stride: usize,
        in_dim: usize,
        grad_input: &mut [f64],
    ) {
        unsafe {
            // Invariant: `grad_out` is row-major with row stride
            // `grad_out_row_stride`; the selected output block has length
            // `out_dim`. `weight_out_by_in` is row-major [out_dim,
            // weight_row_stride], and the first `in_dim` values in each row
            // are the selected input-gradient block.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                in_dim as c_int,
                out_dim as c_int,
                1.0,
                grad_out.as_ptr().add(grad_out_feature_offset),
                grad_out_row_stride as c_int,
                weight_out_by_in.as_ptr(),
                weight_row_stride as c_int,
                1.0,
                grad_input.as_mut_ptr(),
                in_dim as c_int,
            );
        }
    }

    fn so2_mconv_vjp_input_add_fused(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
        grad_input_row_stride: usize,
    ) {
        with_so2_mconv_fused_weight(weight_out_by_in, in_dim, out_dim, |fused_weight| {
            so2_mconv_vjp_input_add_precomputed_fused(
                grad_out,
                rows,
                out_dim,
                fused_weight,
                in_dim,
                grad_input,
                grad_input_row_stride,
            );
        });
    }

    pub fn so2_mconv_vjp_input_add_precomputed_fused(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        fused_weight: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
        grad_input_row_stride: usize,
    ) {
        let fused_in_dim = 2 * in_dim;
        let fused_out_dim = 2 * out_dim;
        unsafe {
            // Invariant: grad_out is row-major [rows, 2 * out_dim]. The fused
            // weight is row-major [2 * out_dim, 2 * in_dim] and encodes:
            // real = grad_real * W0 + grad_imag * W1
            // imag = grad_imag * W0 - grad_real * W1.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                fused_in_dim as c_int,
                fused_out_dim as c_int,
                1.0,
                grad_out.as_ptr(),
                fused_out_dim as c_int,
                fused_weight.as_ptr(),
                fused_in_dim as c_int,
                1.0,
                grad_input.as_mut_ptr(),
                grad_input_row_stride as c_int,
            );
        }
    }

    pub fn so2_mconv_vjp_input_set(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
        grad_input_row_stride: usize,
    ) {
        // This is a single early-return dispatch: the 512 cutoff selects when
        // precomputing the fused BLAS matrix pays for itself, not a paired
        // parallel/scalar guard.
        if rows >= 512 {
            with_so2_mconv_fused_weight(weight_out_by_in, in_dim, out_dim, |fused_weight| {
                so2_mconv_vjp_input_set_precomputed_fused(
                    grad_out,
                    rows,
                    out_dim,
                    fused_weight,
                    in_dim,
                    grad_input,
                    grad_input_row_stride,
                );
            });
            return;
        }

        so2_mconv_vjp_input_add(
            grad_out,
            rows,
            out_dim,
            weight_out_by_in,
            in_dim,
            grad_input,
            grad_input_row_stride,
        );
    }

    pub fn so2_mconv_vjp_input_set_precomputed_fused(
        grad_out: &[f64],
        rows: usize,
        out_dim: usize,
        fused_weight: &[f64],
        in_dim: usize,
        grad_input: &mut [f64],
        grad_input_row_stride: usize,
    ) {
        let fused_in_dim = 2 * in_dim;
        let fused_out_dim = 2 * out_dim;
        unsafe {
            // Invariant: same layout as `so2_mconv_vjp_input_add_precomputed_fused`,
            // but callers pass a fresh zeroed buffer and want overwrite semantics.
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                fused_in_dim as c_int,
                fused_out_dim as c_int,
                1.0,
                grad_out.as_ptr(),
                fused_out_dim as c_int,
                fused_weight.as_ptr(),
                fused_in_dim as c_int,
                0.0,
                grad_input.as_mut_ptr(),
                grad_input_row_stride as c_int,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo2LinearConfig {
    l_max: usize,
    m_max: usize,
    in_channels: usize,
    out_channels: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo2ConvolutionConfig {
    l_max: usize,
    m_max: usize,
    in_channels: usize,
    out_channels: usize,
    extra_m0_output_channels: usize,
}

impl EsenSo2ConvolutionConfig {
    pub fn new(
        l_max: usize,
        m_max: usize,
        in_channels: usize,
        out_channels: usize,
        extra_m0_output_channels: usize,
    ) -> Result<Self> {
        if m_max > l_max {
            return Err(invalid_so2(format!(
                "m_max must be <= l_max, got {m_max} > {l_max}"
            )));
        }
        if in_channels == 0 || out_channels == 0 {
            return Err(invalid_so2(format!(
                "channel dimensions must be nonzero, got {in_channels}->{out_channels}"
            )));
        }
        reduced_coefficient_count(l_max, m_max)
            .and_then(|count| count.checked_mul(in_channels))
            .and_then(|_| reduced_coefficient_count(l_max, m_max)?.checked_mul(out_channels))
            .and_then(|_| {
                (l_max + 1)
                    .checked_mul(out_channels)
                    .and_then(|main| main.checked_add(extra_m0_output_channels))
            })
            .ok_or_else(|| invalid_so2("SO2 convolution dimensions overflow"))?;
        Ok(Self {
            l_max,
            m_max,
            in_channels,
            out_channels,
            extra_m0_output_channels,
        })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[must_use]
    pub fn extra_m0_output_channels(&self) -> usize {
        self.extra_m0_output_channels
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        (0..=self.l_max)
            .map(|degree| 2 * self.m_max.min(degree) + 1)
            .sum()
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.coefficient_count() * self.in_channels
    }

    #[must_use]
    pub fn output_node_stride(&self) -> usize {
        self.coefficient_count() * self.out_channels
    }

    #[must_use]
    pub fn radial_node_stride(&self) -> usize {
        let mut stride = 0;
        for order in 0..=self.m_max {
            stride += self.m_size_unchecked(order) * self.in_channels;
        }
        stride
    }

    #[must_use]
    pub fn m_size(&self, order: usize) -> Option<usize> {
        (order <= self.l_max).then(|| self.m_size_unchecked(order))
    }

    #[must_use]
    fn m_size_unchecked(&self, order: usize) -> usize {
        debug_assert!(order <= self.l_max);
        self.l_max - order + 1
    }

    #[must_use]
    pub fn m0_input_feature_dim(&self) -> usize {
        self.m_size_unchecked(0) * self.in_channels
    }

    #[must_use]
    pub fn m0_main_output_feature_dim(&self) -> usize {
        self.m_size_unchecked(0) * self.out_channels
    }

    #[must_use]
    pub fn m0_output_feature_dim(&self) -> usize {
        self.extra_m0_output_channels + self.m0_main_output_feature_dim()
    }

    #[must_use]
    pub fn m0_weight_len(&self) -> usize {
        self.m0_output_feature_dim() * self.m0_input_feature_dim()
    }

    #[must_use]
    pub fn m0_bias_len(&self) -> usize {
        self.m0_output_feature_dim()
    }

    #[must_use]
    pub fn m_conv_weight_len(&self, order: usize) -> usize {
        let m_size = self.m_size_unchecked(order);
        2 * m_size * self.out_channels * m_size * self.in_channels
    }

    #[must_use]
    pub fn m_conv_fused_weight_len(&self, order: usize) -> usize {
        let m_size = self.m_size_unchecked(order);
        let in_dim = m_size * self.in_channels;
        let out_dim = m_size * self.out_channels;
        4 * in_dim * out_dim
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EsenSo2ConvolutionPreparedPlan {
    config: EsenSo2ConvolutionConfig,
    coeff_offsets: Vec<usize>,
    radial_offsets: Vec<usize>,
    l_to_m_permutation: Vec<usize>,
}

impl EsenSo2ConvolutionPreparedPlan {
    pub fn new(config: EsenSo2ConvolutionConfig) -> Result<Self> {
        let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
            .map_err(|err| invalid_so2(err.to_string()))?;
        let (coeff_offsets, radial_offsets) = so2_convolution_order_offsets(config);
        Ok(Self {
            config,
            coeff_offsets,
            radial_offsets,
            l_to_m_permutation: fairchem_esen_l_to_m_permutation(mapping),
        })
    }

    #[must_use]
    pub fn config(&self) -> EsenSo2ConvolutionConfig {
        self.config
    }
}

pub fn fairchem_esen_so2_m_conv_fused_weight(
    config: EsenSo2ConvolutionConfig,
    order: usize,
    weight: &[f64],
) -> Result<Vec<f64>> {
    if order == 0 || order > config.m_max {
        return Err(invalid_so2(format!(
            "SO2 fused m-conv order must be in 1..={}, got {order}",
            config.m_max
        )));
    }
    check_len(
        "fairchem_esen_so2_m_conv_fused_weight weight",
        config.m_conv_weight_len(order),
        weight.len(),
    )?;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        let m_size = config.m_size_unchecked(order);
        let in_dim = m_size * config.in_channels;
        let out_dim = m_size * config.out_channels;
        Ok(accelerate_blas::build_so2_mconv_fused_weight(
            weight, in_dim, out_dim,
        ))
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        Ok(Vec::new())
    }
}

fn check_so2_m_conv_fused_weights(
    config: EsenSo2ConvolutionConfig,
    m_fused_weights: Option<&[&[f64]]>,
) -> Result<()> {
    let Some(weights) = m_fused_weights else {
        return Ok(());
    };
    check_len(
        "SO2 fused m-conv weight slice count",
        config.m_max,
        weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "SO2 fused m-conv order weight",
            config.m_conv_fused_weight_len(order),
            weights[order - 1].len(),
        )?;
    }
    Ok(())
}

fn so2_convolution_order_offsets(config: EsenSo2ConvolutionConfig) -> (Vec<usize>, Vec<usize>) {
    let mut coeff_offsets = Vec::with_capacity(config.m_max + 1);
    let mut radial_offsets = Vec::with_capacity(config.m_max + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.in_channels;
    }
    (coeff_offsets, radial_offsets)
}

fn gather_so2_radial_from_block_source<F>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    radial_block_source: &F,
) -> Result<Vec<f64>>
where
    F: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    let (_, radial_offsets) = so2_convolution_order_offsets(config);
    for (order, &radial_offset) in radial_offsets.iter().enumerate().take(config.m_max + 1) {
        let feature_dim = config.m_size_unchecked(order) * config.in_channels;
        let mut block = vec![0.0; nodes * feature_dim];
        radial_block_source(radial_offset, feature_dim, &mut block)?;
        for node in 0..nodes {
            let source_start = node * feature_dim;
            let target_start = node * config.radial_node_stride() + radial_offset;
            radial[target_start..target_start + feature_dim]
                .copy_from_slice(&block[source_start..source_start + feature_dim]);
        }
    }
    Ok(radial)
}

fn gather_so2_radial_from_strided_block_source<F>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    radial_strided_block_source: &F,
) -> Result<Vec<f64>>
where
    F: Fn(usize, usize, usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    let (_, radial_offsets) = so2_convolution_order_offsets(config);
    for (order, &radial_offset) in radial_offsets.iter().enumerate().take(config.m_max + 1) {
        let feature_dim = config.m_size_unchecked(order) * config.in_channels;
        let mut block = vec![0.0; nodes * feature_dim];
        radial_strided_block_source(radial_offset, feature_dim, feature_dim, 0, &mut block)?;
        for node in 0..nodes {
            let source_start = node * feature_dim;
            let target_start = node * config.radial_node_stride() + radial_offset;
            radial[target_start..target_start + feature_dim]
                .copy_from_slice(&block[source_start..source_start + feature_dim]);
        }
    }
    Ok(radial)
}

impl EsenSo2LinearConfig {
    pub fn new(
        l_max: usize,
        m_max: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        if m_max > l_max {
            return Err(invalid_so2(format!(
                "m_max must be <= l_max, got {m_max} > {l_max}"
            )));
        }
        if in_channels == 0 || out_channels == 0 {
            return Err(invalid_so2(format!(
                "channel dimensions must be nonzero, got {in_channels}->{out_channels}"
            )));
        }
        reduced_coefficient_count(l_max, m_max)
            .and_then(|count| count.checked_mul(in_channels))
            .and_then(|_| reduced_coefficient_count(l_max, m_max)?.checked_mul(out_channels))
            .ok_or_else(|| invalid_so2("SO2 linear dimensions overflow"))?;
        Ok(Self {
            l_max,
            m_max,
            in_channels,
            out_channels,
        })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        (0..=self.l_max)
            .map(|degree| 2 * self.m_max.min(degree) + 1)
            .sum()
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.coefficient_count() * self.in_channels
    }

    #[must_use]
    pub fn output_node_stride(&self) -> usize {
        self.coefficient_count() * self.out_channels
    }

    #[must_use]
    pub fn radial_node_stride(&self) -> usize {
        let mut stride = 0;
        for order in 0..=self.m_max {
            stride += self.m_size_unchecked(order) * self.in_channels;
        }
        stride
    }

    #[must_use]
    pub fn m_size(&self, order: usize) -> Option<usize> {
        (order <= self.l_max).then(|| self.m_size_unchecked(order))
    }

    #[must_use]
    fn m_size_unchecked(&self, order: usize) -> usize {
        debug_assert!(order <= self.l_max);
        self.l_max - order + 1
    }

    #[must_use]
    pub fn m_weight_len(&self, order: usize) -> usize {
        let m_size = self.m_size_unchecked(order);
        m_size * self.out_channels * m_size * self.in_channels
    }

    #[must_use]
    pub fn m0_bias_len(&self) -> usize {
        self.m_size_unchecked(0) * self.out_channels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo2LinearBlockConfig {
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    copies: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo2MConvBlockConfig {
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo2RadialModulationConfig {
    feature_dim: usize,
    copies: usize,
}

impl EsenSo2RadialModulationConfig {
    pub fn new(feature_dim: usize, copies: usize) -> Result<Self> {
        if feature_dim == 0 {
            return Err(invalid_so2("feature_dim must be nonzero"));
        }
        if copies != 1 && copies != 2 {
            return Err(invalid_so2(format!(
                "copies must be 1 for m=0 or 2 for m>0, got {copies}"
            )));
        }
        Ok(Self {
            feature_dim,
            copies,
        })
    }

    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    #[must_use]
    pub fn copies(&self) -> usize {
        self.copies
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.copies * self.feature_dim
    }
}

/// FairChem eSEN `SO2_Linear` assembly for l-major reduced coefficient tensors.
pub fn fairchem_esen_so2_linear_forward(
    config: EsenSo2LinearConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_l_major: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_linear_assembly_shapes(
        "fairchem_esen_so2_linear_forward",
        config,
        input_l_major,
        radial,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
    )?;
    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        let permutation = fairchem_esen_l_to_m_permutation(mapping);
        return run_so2_linear_assembly_forward_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            out_l_major,
            &permutation,
        );
    }

    let mut input_m_major = vec![0.0; input_l_major.len()];
    let mut out_m_major = vec![0.0; out_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.in_channels,
        input_l_major,
        &mut input_m_major,
    )?;

    run_so2_linear_assembly_forward_m_major(
        config,
        nodes,
        &input_m_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        &mut out_m_major,
    )?;
    fairchem_esen_m_to_l_forward(mapping, config.out_channels, &out_m_major, out_l_major)
}

/// Accumulate input and radial VJPs of [`fairchem_esen_so2_linear_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_linear_vjp(
    config: EsenSo2LinearConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_l_major: &[f64],
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let nodes = check_so2_linear_assembly_shapes(
        "fairchem_esen_so2_linear_vjp",
        config,
        input_l_major,
        radial,
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
    )?;
    check_len(
        "fairchem_esen_so2_linear_vjp grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    if let Some(radial_values) = radial {
        let Some(grad_radial_values) = grad_radial.as_deref() else {
            return Err(invalid_so2(
                "grad_radial is required when radial is present",
            ));
        };
        check_len(
            "fairchem_esen_so2_linear_vjp grad_radial",
            radial_values.len(),
            grad_radial_values.len(),
        )?;
    }

    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        let permutation = fairchem_esen_l_to_m_permutation(mapping);
        return run_so2_linear_assembly_vjp_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m_weights,
            grad_out_l_major,
            grad_input_l_major,
            grad_radial,
            &permutation,
        );
    }

    let mut input_m_major = vec![0.0; input_l_major.len()];
    let mut grad_out_m_major = vec![0.0; grad_out_l_major.len()];
    let mut grad_input_m_major = vec![0.0; input_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.in_channels,
        input_l_major,
        &mut input_m_major,
    )?;
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels,
        grad_out_l_major,
        &mut grad_out_m_major,
    )?;

    run_so2_linear_assembly_vjp_m_major(
        config,
        nodes,
        &input_m_major,
        radial,
        m0_weight,
        m_weights,
        &grad_out_m_major,
        &mut grad_input_m_major,
        grad_radial,
    )?;

    let mut grad_input_l_temp = vec![0.0; input_l_major.len()];
    fairchem_esen_m_to_l_forward(
        mapping,
        config.in_channels,
        &grad_input_m_major,
        &mut grad_input_l_temp,
    )?;
    for (target, value) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
        *target += value;
    }
    Ok(())
}

/// FairChem eSEN `SO2_Convolution` assembly for l-major reduced coefficients.
///
/// This covers the m=0 dense path, optional extra m=0 output features, m>0
/// real/imag SO2 convolution blocks, optional radial modulation, and the
/// l-major/m-major remapping around those blocks.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    fairchem_esen_so2_convolution_forward_with_fused_m_weights(
        config,
        input_l_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        None,
        out_l_major,
        extra_m0_out,
    )
}

/// FairChem eSEN `SO2_Convolution` assembly with optional precomputed fused
/// m-convolution weights for the Accelerate BLAS large-row path.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_with_fused_m_weights",
        config,
        input_l_major,
        radial,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        let permutation = fairchem_esen_l_to_m_permutation(mapping);
        let (coeff_offsets, radial_offsets) = so2_convolution_order_offsets(config);
        return run_so2_convolution_assembly_forward_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_l_major,
            extra_m0_out,
            &coeff_offsets,
            &radial_offsets,
            &permutation,
            None,
        );
    }

    let mut input_m_major = vec![0.0; input_l_major.len()];
    let mut out_m_major = vec![0.0; out_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.in_channels,
        input_l_major,
        &mut input_m_major,
    )?;

    run_so2_convolution_assembly_forward_m_major(
        config,
        nodes,
        &input_m_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        &mut out_m_major,
        extra_m0_out,
    )?;
    fairchem_esen_m_to_l_forward(mapping, config.out_channels, &out_m_major, out_l_major)
}

/// Accumulate input and radial VJPs of [`fairchem_esen_so2_convolution_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    fairchem_esen_so2_convolution_vjp_with_fused_m_weights(
        config,
        input_l_major,
        radial,
        m0_weight,
        m_weights,
        None,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

/// VJP for [`fairchem_esen_so2_convolution_forward_with_fused_m_weights`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_with_fused_m_weights",
        config,
        input_l_major,
        radial,
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    if let Some(radial_values) = radial {
        let Some(grad_radial_values) = grad_radial.as_deref() else {
            return Err(invalid_so2(
                "grad_radial is required when radial is present",
            ));
        };
        check_len(
            "fairchem_esen_so2_convolution_vjp grad_radial",
            radial_values.len(),
            grad_radial_values.len(),
        )?;
    }

    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        let permutation = fairchem_esen_l_to_m_permutation(mapping);
        let (coeff_offsets, radial_offsets) = so2_convolution_order_offsets(config);
        return run_so2_convolution_assembly_vjp_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_l_major,
            grad_extra_m0,
            grad_input_l_major,
            grad_radial,
            &coeff_offsets,
            &radial_offsets,
            &permutation,
        );
    }

    let mut input_m_major = vec![0.0; input_l_major.len()];
    let mut grad_out_m_major = vec![0.0; grad_out_l_major.len()];
    let mut grad_input_m_major = vec![0.0; input_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.in_channels,
        input_l_major,
        &mut input_m_major,
    )?;
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels,
        grad_out_l_major,
        &mut grad_out_m_major,
    )?;

    run_so2_convolution_assembly_vjp_m_major(
        config,
        nodes,
        &input_m_major,
        radial,
        m0_weight,
        m_weights,
        &grad_out_m_major,
        grad_extra_m0,
        &mut grad_input_m_major,
        grad_radial,
    )?;

    let mut grad_input_l_temp = vec![0.0; input_l_major.len()];
    fairchem_esen_m_to_l_forward(
        mapping,
        config.in_channels,
        &grad_input_m_major,
        &mut grad_input_l_temp,
    )?;
    for (target, value) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
        *target += value;
    }
    Ok(())
}

/// Fused FairChem eSEN l=2/m=2 GateActivation + no-radial SO2 convolution.
///
/// This specialized path avoids materializing the activated l-major tensor
/// before the second edge-wise SO2 convolution. It is intended for fixed UMA
/// force replay where the gate sigmoid and scalar SiLU derivative caches are
/// still required by the VJP, but the activated tensor itself is not retained.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_l2_m2_from_gate_activation_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    gating_scalars: &mut [f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
) -> Result<()> {
    if config.l_max != 2 || config.m_max != 2 || config.coefficient_count() != 9 {
        return Err(invalid_so2(
            "fused gate activation SO2 path requires l_max=2 and m_max=2",
        ));
    }
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_l2_m2_from_gate_activation_with_fused_m_weights",
        config,
        input_l_major,
        None,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        None,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    check_len(
        "fairchem_esen_so2_convolution_forward_l2_m2_from_gate_activation_with_fused_m_weights gating_scalars",
        nodes * 2 * config.in_channels,
        gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_forward_l2_m2_from_gate_activation_with_fused_m_weights scalar_silu_derivatives",
        nodes * config.in_channels,
        scalar_silu_derivatives.len(),
    )?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let channels = config.in_channels;
    let mut m0_input = vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels];
    fill_l2_m2_gate_activation_m0_input(
        nodes,
        channels,
        input_l_major,
        gating_scalars,
        &mut m0_input,
        scalar_silu_derivatives,
    );
    let mut m0_out = vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels];
    dense_forward_with_bias_rows(
        nodes,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels,
        &m0_input,
        m0_weight,
        m0_bias,
        &mut m0_out,
    );
    scatter_l2_m2_order_block(
        nodes,
        config.out_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        &m0_out,
        out_l_major,
    );

    for (order, sources) in [
        (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
        (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
    ] {
        let m_size = config.m_size_unchecked(order);
        let block_config = EsenSo2MConvBlockConfig::new(m_size, channels, config.out_channels)?;
        let mut block_input = vec![0.0; nodes * sources.len() * channels];
        fill_l2_m2_gate_activation_gated_order_input(
            nodes,
            channels,
            sources,
            input_l_major,
            gating_scalars,
            &mut block_input,
        );
        let mut block_out = vec![0.0; nodes * sources.len() * config.out_channels];
        let weight = m_weights[order - 1];
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                feature = "openblas",
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        ))]
        if let Some(fused_weight) =
            m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
        {
            fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                block_config,
                &block_input,
                weight,
                fused_weight,
                &mut block_out,
            )?;
            scatter_l2_m2_order_block(nodes, config.out_channels, sources, &block_out, out_l_major);
            continue;
        }
        fairchem_esen_so2_m_conv_block_forward(block_config, &block_input, weight, &mut block_out)?;
        scatter_l2_m2_order_block(nodes, config.out_channels, sources, &block_out, out_l_major);
    }

    Ok(())
}

/// Specialized FairChem eSEN l=2/m=2 radial SO2 convolution forward.
///
/// This is the fixed UMA conv1 shape. It avoids the generic coefficient
/// permutation/assembly scaffolding while still using the same BLAS m-blocks.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_l2_m2_radial_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    if config.l_max != 2 || config.m_max != 2 || config.coefficient_count() != 9 {
        return Err(invalid_so2(
            "specialized radial SO2 forward requires l_max=2 and m_max=2",
        ));
    }
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_l2_m2_radial_with_fused_m_weights",
        config,
        input_l_major,
        Some(radial),
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let channels = config.in_channels;
    let mut m0_input = vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels];
    fill_l2_m2_radial_order_input(
        nodes,
        channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        0,
        input_l_major,
        radial,
        &mut m0_input,
    );
    let mut m0_dense_out = vec![0.0; nodes * config.m0_output_feature_dim()];
    dense_forward_with_bias_rows(
        nodes,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
        config.m0_output_feature_dim(),
        &m0_input,
        m0_weight,
        m0_bias,
        &mut m0_dense_out,
    );
    scatter_l2_m2_m0_dense_out(
        nodes,
        config.out_channels,
        config.extra_m0_output_channels,
        &m0_dense_out,
        out_l_major,
        extra_m0_out,
    )?;

    let mut radial_offset = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels;
    for (order, sources) in [
        (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
        (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
    ] {
        let m_size = config.m_size_unchecked(order);
        let block_config = EsenSo2MConvBlockConfig::new(m_size, channels, config.out_channels)?;
        let mut block_input = vec![0.0; nodes * sources.len() * channels];
        fill_l2_m2_radial_order_input(
            nodes,
            channels,
            sources,
            radial_offset,
            input_l_major,
            radial,
            &mut block_input,
        );
        let mut block_out = vec![0.0; nodes * sources.len() * config.out_channels];
        let weight = m_weights[order - 1];
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                feature = "openblas",
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        ))]
        if let Some(fused_weight) =
            m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
        {
            fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                block_config,
                &block_input,
                weight,
                fused_weight,
                &mut block_out,
            )?;
            scatter_l2_m2_order_block(nodes, config.out_channels, sources, &block_out, out_l_major);
            radial_offset += m_size * channels;
            continue;
        }
        fairchem_esen_so2_m_conv_block_forward(block_config, &block_input, weight, &mut block_out)?;
        scatter_l2_m2_order_block(nodes, config.out_channels, sources, &block_out, out_l_major);
        radial_offset += m_size * channels;
    }

    Ok(())
}

/// Specialized UMA l=2/m=2 conv1(radial) + GateActivation + conv2(no-radial).
///
/// The first convolution's output is still materialized for force VJP replay,
/// but the second convolution inputs are filled directly from the first
/// convolution blocks, avoiding a second l-major activation assembly pass.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_l2_m2_radial_gate_conv2_with_fused_m_weights(
    conv1_config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: &[f64],
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv1_m_fused_weights: Option<&[&[f64]]>,
    conv2_config: EsenSo2ConvolutionConfig,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    conv2_m_fused_weights: Option<&[&[f64]]>,
    conv1_out_l_major: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
    conv2_out_l_major: &mut [f64],
) -> Result<()> {
    if conv1_config.l_max != 2
        || conv1_config.m_max != 2
        || conv1_config.coefficient_count() != 9
        || conv2_config.l_max != 2
        || conv2_config.m_max != 2
        || conv2_config.coefficient_count() != 9
    {
        return Err(invalid_so2(
            "fused UMA SO2 forward requires conv1/conv2 l_max=2 and m_max=2",
        ));
    }
    if conv1_config.out_channels != conv2_config.in_channels {
        return Err(invalid_so2(format!(
            "fused UMA SO2 forward channel mismatch: conv1 out {} vs conv2 in {}",
            conv1_config.out_channels, conv2_config.in_channels
        )));
    }
    if conv1_config.extra_m0_output_channels != 2 * conv1_config.out_channels {
        return Err(invalid_so2(format!(
            "fused UMA SO2 forward requires conv1 extra m0 channels {}, got {}",
            2 * conv1_config.out_channels,
            conv1_config.extra_m0_output_channels
        )));
    }
    if conv2_config.extra_m0_output_channels != 0 {
        return Err(invalid_so2(
            "fused UMA SO2 forward requires conv2 without extra m0 outputs",
        ));
    }
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_l2_m2_radial_gate_conv2_with_fused_m_weights conv1",
        conv1_config,
        input_l_major,
        Some(radial),
        conv1_m0_weight,
        Some(conv1_m0_bias),
        conv1_m_weights,
        conv1_out_l_major,
        Some(gate_sigmoids),
    )?;
    check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_l2_m2_radial_gate_conv2_with_fused_m_weights conv2",
        conv2_config,
        conv1_out_l_major,
        None,
        conv2_m0_weight,
        Some(conv2_m0_bias),
        conv2_m_weights,
        conv2_out_l_major,
        None,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_forward_l2_m2_radial_gate_conv2_with_fused_m_weights scalar_silu_derivatives",
        nodes * conv2_config.in_channels,
        scalar_silu_derivatives.len(),
    )?;
    check_so2_m_conv_fused_weights(conv1_config, conv1_m_fused_weights)?;
    check_so2_m_conv_fused_weights(conv2_config, conv2_m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let _ = conv1_m_fused_weights;
        let _ = conv2_m_fused_weights;
    }

    L2_M2_RADIAL_GATE_CONV2_FORWARD_SCRATCH.with(|scratch| -> Result<()> {
        let mut scratch = scratch.borrow_mut();
        let L2M2RadialGateConv2ForwardScratch {
            conv1_m0_input,
            conv1_m0_dense_out,
            conv2_m0_input,
            conv2_m0_out,
            conv1_block_input,
            conv1_block_out,
            conv2_block_input,
            conv2_block_out,
        } = &mut *scratch;

        let conv1_in_channels = conv1_config.in_channels;
        let channels = conv1_config.out_channels;
        reset_zeroed_f64_vec(
            conv1_m0_input,
            nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv1_in_channels,
        );
        let conv1_m0_input = &mut conv1_m0_input[..];
        fill_l2_m2_radial_order_input(
            nodes,
            conv1_in_channels,
            &L2_M2_ORDER0_L_MAJOR_SOURCES,
            0,
            input_l_major,
            radial,
            conv1_m0_input,
        );
        reset_zeroed_f64_vec(
            conv1_m0_dense_out,
            nodes * conv1_config.m0_output_feature_dim(),
        );
        let conv1_m0_dense_out = &mut conv1_m0_dense_out[..];
        dense_forward_with_bias_rows(
            nodes,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv1_in_channels,
            conv1_config.m0_output_feature_dim(),
            conv1_m0_input,
            conv1_m0_weight,
            conv1_m0_bias,
            conv1_m0_dense_out,
        );

        reset_zeroed_f64_vec(
            conv2_m0_input,
            nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
        );
        let conv2_m0_input = &mut conv2_m0_input[..];
        fill_l2_m2_gate_conv2_m0_input_from_conv1_dense_out(
            nodes,
            channels,
            conv1_config.extra_m0_output_channels,
            conv1_m0_dense_out,
            conv1_out_l_major,
            gate_sigmoids,
            conv2_m0_input,
            scalar_silu_derivatives,
        );
        reset_zeroed_f64_vec(
            conv2_m0_out,
            nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
        );
        let conv2_m0_out = &mut conv2_m0_out[..];
        dense_forward_with_bias_rows(
            nodes,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv2_config.out_channels,
            conv2_m0_input,
            conv2_m0_weight,
            conv2_m0_bias,
            conv2_m0_out,
        );
        scatter_l2_m2_order_block(
            nodes,
            conv2_config.out_channels,
            &L2_M2_ORDER0_L_MAJOR_SOURCES,
            conv2_m0_out,
            conv2_out_l_major,
        );

        let max_order_sources = L2_M2_ORDER1_L_MAJOR_SOURCES.len();
        reset_zeroed_f64_vec(
            conv1_block_input,
            nodes * max_order_sources * conv1_in_channels,
        );
        reset_zeroed_f64_vec(conv1_block_out, nodes * max_order_sources * channels);
        reset_zeroed_f64_vec(conv2_block_input, nodes * max_order_sources * channels);
        reset_zeroed_f64_vec(
            conv2_block_out,
            nodes * max_order_sources * conv2_config.out_channels,
        );

        let mut radial_offset = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv1_in_channels;
        for (order, sources) in [
            (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
            (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
        ] {
            let m_size = conv1_config.m_size_unchecked(order);
            let conv1_block_config =
                EsenSo2MConvBlockConfig::new(m_size, conv1_in_channels, channels)?;
            let conv1_block_input_len = nodes * sources.len() * conv1_in_channels;
            let conv1_block_input = &mut conv1_block_input[..conv1_block_input_len];
            fill_l2_m2_radial_order_input(
                nodes,
                conv1_in_channels,
                sources,
                radial_offset,
                input_l_major,
                radial,
                conv1_block_input,
            );
            let conv1_block_out_len = nodes * sources.len() * channels;
            let conv1_block_out = &mut conv1_block_out[..conv1_block_out_len];
            let conv1_weight = conv1_m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            if let Some(fused_weight) =
                conv1_m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
            {
                fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                    conv1_block_config,
                    conv1_block_input,
                    conv1_weight,
                    fused_weight,
                    conv1_block_out,
                )?;
            } else {
                fairchem_esen_so2_m_conv_block_forward(
                    conv1_block_config,
                    conv1_block_input,
                    conv1_weight,
                    conv1_block_out,
                )?;
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_forward(
                conv1_block_config,
                conv1_block_input,
                conv1_weight,
                conv1_block_out,
            )?;

            let conv2_block_input_len = nodes * sources.len() * channels;
            let conv2_block_input = &mut conv2_block_input[..conv2_block_input_len];
            scatter_l2_m2_order_block_and_fill_gate_input(
                nodes,
                channels,
                sources,
                conv1_block_out,
                gate_sigmoids,
                conv1_out_l_major,
                conv2_block_input,
            );

            let conv2_block_config =
                EsenSo2MConvBlockConfig::new(m_size, channels, conv2_config.out_channels)?;
            let conv2_block_out_len = nodes * sources.len() * conv2_config.out_channels;
            let conv2_block_out = &mut conv2_block_out[..conv2_block_out_len];
            let conv2_weight = conv2_m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            if let Some(fused_weight) =
                conv2_m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
            {
                fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                    conv2_block_config,
                    conv2_block_input,
                    conv2_weight,
                    fused_weight,
                    conv2_block_out,
                )?;
                scatter_l2_m2_order_block(
                    nodes,
                    conv2_config.out_channels,
                    sources,
                    conv2_block_out,
                    conv2_out_l_major,
                );
                radial_offset += m_size * conv1_in_channels;
                continue;
            }
            fairchem_esen_so2_m_conv_block_forward(
                conv2_block_config,
                conv2_block_input,
                conv2_weight,
                conv2_block_out,
            )?;
            scatter_l2_m2_order_block(
                nodes,
                conv2_config.out_channels,
                sources,
                conv2_block_out,
                conv2_out_l_major,
            );
            radial_offset += m_size * conv1_in_channels;
        }

        Ok(())
    })
}

/// Specialized l=2/m=2 no-radial input VJP for FairChem eSEN SO2 convolution.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_input_l_major: &mut [f64],
) -> Result<()> {
    if config.l_max != 2 || config.m_max != 2 || config.coefficient_count() != 9 {
        return Err(invalid_so2(
            "specialized no-radial SO2 VJP requires l_max=2 and m_max=2",
        ));
    }
    if config.extra_m0_output_channels != 0 {
        return Err(invalid_so2(
            "specialized no-radial SO2 VJP does not support extra m0 outputs",
        ));
    }
    let output_node_stride = config.output_node_stride();
    if !grad_out_l_major.len().is_multiple_of(output_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights grad_out length {} is not a multiple of node stride {output_node_stride}",
            grad_out_l_major.len()
        )));
    }
    let nodes = grad_out_l_major.len() / output_node_stride;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights grad_input",
        nodes * config.input_node_stride(),
        grad_input_l_major.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights m0_weight",
        config.m0_weight_len(),
        m0_weight.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights m_weights",
        config.m_max,
        m_weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "fairchem_esen_so2_convolution_vjp_l2_m2_no_radial_with_fused_m_weights m_weight",
            config.m_conv_weight_len(order),
            m_weights[order - 1].len(),
        )?;
    }
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let mut m0_grad_out =
        vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels];
    gather_l2_m2_order_block(
        nodes,
        config.out_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        grad_out_l_major,
        &mut m0_grad_out,
    );
    let mut m0_grad_input =
        vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels];
    dense_vjp_input_no_bias_rows_add(
        nodes,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels,
        m0_weight,
        &m0_grad_out,
        &mut m0_grad_input,
    );
    scatter_add_l2_m2_order_block(
        nodes,
        config.in_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        &m0_grad_input,
        grad_input_l_major,
    );

    for (order, sources) in [
        (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
        (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
    ] {
        let m_size = config.m_size_unchecked(order);
        let block_config =
            EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
        let mut block_grad_out = vec![0.0; nodes * sources.len() * config.out_channels];
        gather_l2_m2_order_block(
            nodes,
            config.out_channels,
            sources,
            grad_out_l_major,
            &mut block_grad_out,
        );
        let mut block_grad_input = vec![0.0; nodes * sources.len() * config.in_channels];
        let weight = m_weights[order - 1];
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                feature = "openblas",
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        ))]
        if let Some(fused_weight) =
            m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
        {
            fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight(
                block_config,
                weight,
                fused_weight,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            scatter_add_l2_m2_order_block(
                nodes,
                config.in_channels,
                sources,
                &block_grad_input,
                grad_input_l_major,
            );
            continue;
        }
        fairchem_esen_so2_m_conv_block_vjp(
            block_config,
            weight,
            &block_grad_out,
            &mut block_grad_input,
        )?;
        scatter_add_l2_m2_order_block(
            nodes,
            config.in_channels,
            sources,
            &block_grad_input,
            grad_input_l_major,
        );
    }

    Ok(())
}

/// Fused l=2/m=2 no-radial SO2 input VJP followed by GateActivation VJP.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    input_l_major: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_l_major: &mut [f64],
) -> Result<()> {
    if config.l_max != 2 || config.m_max != 2 || config.coefficient_count() != 9 {
        return Err(invalid_so2(
            "fused no-radial SO2/GateActivation VJP requires l_max=2 and m_max=2",
        ));
    }
    if config.extra_m0_output_channels != 0 {
        return Err(invalid_so2(
            "fused no-radial SO2/GateActivation VJP does not support extra m0 outputs",
        ));
    }
    let output_node_stride = config.output_node_stride();
    if !grad_out_l_major.len().is_multiple_of(output_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights grad_out length {} is not a multiple of node stride {output_node_stride}",
            grad_out_l_major.len()
        )));
    }
    let nodes = grad_out_l_major.len() / output_node_stride;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights input",
        nodes * config.input_node_stride(),
        input_l_major.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights gate_sigmoids",
        nodes * 2 * config.in_channels,
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights scalar_silu_derivatives",
        nodes * config.in_channels,
        scalar_silu_derivatives.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights grad_gating",
        gate_sigmoids.len(),
        grad_gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights grad_input",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights m0_weight",
        config.m0_weight_len(),
        m0_weight.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights m_weights",
        config.m_max,
        m_weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "fairchem_esen_so2_convolution_gate_activation_vjp_l2_m2_no_radial_with_fused_m_weights m_weight",
            config.m_conv_weight_len(order),
            m_weights[order - 1].len(),
        )?;
    }
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let mut m0_grad_out =
        vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels];
    gather_l2_m2_order_block(
        nodes,
        config.out_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        grad_out_l_major,
        &mut m0_grad_out,
    );
    let mut m0_grad_input =
        vec![0.0; nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels];
    dense_vjp_input_no_bias_rows_add(
        nodes,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.out_channels,
        m0_weight,
        &m0_grad_out,
        &mut m0_grad_input,
    );
    accumulate_l2_m2_gate_activation_vjp_from_order_block(
        nodes,
        config.in_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        &m0_grad_input,
        input_l_major,
        gate_sigmoids,
        scalar_silu_derivatives,
        grad_gating_scalars,
        grad_input_l_major,
    );

    for (order, sources) in [
        (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
        (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
    ] {
        let m_size = config.m_size_unchecked(order);
        let block_config =
            EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
        let mut block_grad_out = vec![0.0; nodes * sources.len() * config.out_channels];
        gather_l2_m2_order_block(
            nodes,
            config.out_channels,
            sources,
            grad_out_l_major,
            &mut block_grad_out,
        );
        let mut block_grad_input = vec![0.0; nodes * sources.len() * config.in_channels];
        let weight = m_weights[order - 1];
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                feature = "openblas",
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        ))]
        if let Some(fused_weight) =
            m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
        {
            fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight(
                block_config,
                weight,
                fused_weight,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            accumulate_l2_m2_gate_activation_vjp_from_order_block(
                nodes,
                config.in_channels,
                sources,
                &block_grad_input,
                input_l_major,
                gate_sigmoids,
                scalar_silu_derivatives,
                grad_gating_scalars,
                grad_input_l_major,
            );
            continue;
        }
        fairchem_esen_so2_m_conv_block_vjp(
            block_config,
            weight,
            &block_grad_out,
            &mut block_grad_input,
        )?;
        accumulate_l2_m2_gate_activation_vjp_from_order_block(
            nodes,
            config.in_channels,
            sources,
            &block_grad_input,
            input_l_major,
            gate_sigmoids,
            scalar_silu_derivatives,
            grad_gating_scalars,
            grad_input_l_major,
        );
    }
    Ok(())
}

/// Specialized l=2/m=2 radial SO2 input/radial VJP with optional extra m0 grad.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights(
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: &mut [f64],
) -> Result<()> {
    if config.l_max != 2 || config.m_max != 2 || config.coefficient_count() != 9 {
        return Err(invalid_so2(
            "specialized radial SO2 VJP requires l_max=2 and m_max=2",
        ));
    }
    let output_node_stride = config.output_node_stride();
    if !grad_out_l_major.len().is_multiple_of(output_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights grad_out length {} is not a multiple of node stride {output_node_stride}",
            grad_out_l_major.len()
        )));
    }
    let nodes = grad_out_l_major.len() / output_node_stride;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights input",
        nodes * config.input_node_stride(),
        input_l_major.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights grad_input",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights radial",
        nodes * config.radial_node_stride(),
        radial.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights grad_radial",
        radial.len(),
        grad_radial.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights m0_weight",
        config.m0_weight_len(),
        m0_weight.len(),
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights m_weights",
        config.m_max,
        m_weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights m_weight",
            config.m_conv_weight_len(order),
            m_weights[order - 1].len(),
        )?;
    }
    if config.extra_m0_output_channels > 0 {
        let Some(extra) = grad_extra_m0 else {
            return Err(invalid_so2("validated extra m0 grad is missing"));
        };
        check_len(
            "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights extra_m0",
            nodes * config.extra_m0_output_channels,
            extra.len(),
        )?;
    } else if let Some(extra) = grad_extra_m0 {
        check_len(
            "fairchem_esen_so2_convolution_vjp_l2_m2_radial_with_fused_m_weights extra_m0",
            0,
            extra.len(),
        )?;
    }
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let m0_input_len = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels;
    let m0_output_len = config.m0_output_feature_dim();
    let mut m0_full_grad_out = vec![0.0; nodes * m0_output_len];
    gather_l2_m2_m0_grad_out_with_extra(
        nodes,
        config.out_channels,
        config.extra_m0_output_channels,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        grad_out_l_major,
        grad_extra_m0,
        &mut m0_full_grad_out,
    );
    let mut m0_grad_modulated = vec![0.0; nodes * m0_input_len];
    dense_vjp_input_no_bias_rows_add(
        nodes,
        m0_input_len,
        m0_output_len,
        m0_weight,
        &m0_full_grad_out,
        &mut m0_grad_modulated,
    );
    accumulate_l2_m2_radial_modulation_vjp_from_order_block(
        nodes,
        config.in_channels,
        L2_M2_ORDER0_L_MAJOR_SOURCES.len(),
        0,
        &L2_M2_ORDER0_L_MAJOR_SOURCES,
        &m0_grad_modulated,
        input_l_major,
        radial,
        grad_input_l_major,
        grad_radial,
    );

    let mut radial_offset = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * config.in_channels;
    for (order, sources) in [
        (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
        (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
    ] {
        let m_size = config.m_size_unchecked(order);
        let block_config =
            EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
        let mut block_grad_out = vec![0.0; nodes * sources.len() * config.out_channels];
        gather_l2_m2_order_block(
            nodes,
            config.out_channels,
            sources,
            grad_out_l_major,
            &mut block_grad_out,
        );
        let mut block_grad_modulated = vec![0.0; nodes * sources.len() * config.in_channels];
        let weight = m_weights[order - 1];
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                feature = "openblas",
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        ))]
        if let Some(fused_weight) =
            m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
        {
            fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight(
                block_config,
                weight,
                fused_weight,
                &block_grad_out,
                &mut block_grad_modulated,
            )?;
            accumulate_l2_m2_radial_modulation_vjp_from_order_block(
                nodes,
                config.in_channels,
                m_size,
                radial_offset,
                sources,
                &block_grad_modulated,
                input_l_major,
                radial,
                grad_input_l_major,
                grad_radial,
            );
            radial_offset += m_size * config.in_channels;
            continue;
        }
        fairchem_esen_so2_m_conv_block_vjp(
            block_config,
            weight,
            &block_grad_out,
            &mut block_grad_modulated,
        )?;
        accumulate_l2_m2_radial_modulation_vjp_from_order_block(
            nodes,
            config.in_channels,
            m_size,
            radial_offset,
            sources,
            &block_grad_modulated,
            input_l_major,
            radial,
            grad_input_l_major,
            grad_radial,
        );
        radial_offset += m_size * config.in_channels;
    }

    Ok(())
}

/// Fused VJP for UMA conv2(no-radial) + GateActivation + conv1(radial).
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_l2_m2_radial_gate_conv2_with_fused_m_weights(
    conv2_config: EsenSo2ConvolutionConfig,
    conv2_m0_weight: &[f64],
    conv2_m_weights: &[&[f64]],
    conv2_m_fused_weights: Option<&[&[f64]]>,
    grad_conv2_out_l_major: &[f64],
    conv1_out_l_major: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    conv1_config: EsenSo2ConvolutionConfig,
    conv1_input_l_major: &[f64],
    conv1_radial: &[f64],
    conv1_m0_weight: &[f64],
    conv1_m_weights: &[&[f64]],
    conv1_m_fused_weights: Option<&[&[f64]]>,
    grad_conv1_input_l_major: &mut [f64],
    grad_conv1_radial: &mut [f64],
) -> Result<()> {
    if conv1_config.l_max != 2
        || conv1_config.m_max != 2
        || conv1_config.coefficient_count() != 9
        || conv2_config.l_max != 2
        || conv2_config.m_max != 2
        || conv2_config.coefficient_count() != 9
    {
        return Err(invalid_so2(
            "fused UMA SO2 VJP requires conv1/conv2 l_max=2 and m_max=2",
        ));
    }
    if conv1_config.out_channels != conv2_config.in_channels {
        return Err(invalid_so2(format!(
            "fused UMA SO2 VJP channel mismatch: conv1 out {} vs conv2 in {}",
            conv1_config.out_channels, conv2_config.in_channels
        )));
    }
    if conv1_config.extra_m0_output_channels != 2 * conv1_config.out_channels {
        return Err(invalid_so2(format!(
            "fused UMA SO2 VJP requires conv1 extra m0 channels {}, got {}",
            2 * conv1_config.out_channels,
            conv1_config.extra_m0_output_channels
        )));
    }
    if conv2_config.extra_m0_output_channels != 0 {
        return Err(invalid_so2(
            "fused UMA SO2 VJP requires conv2 without extra m0 outputs",
        ));
    }
    let output_node_stride = conv2_config.output_node_stride();
    if !grad_conv2_out_l_major
        .len()
        .is_multiple_of(output_node_stride)
    {
        return Err(invalid_so2(format!(
            "fused UMA SO2 VJP grad_conv2_out length {} is not a multiple of node stride {output_node_stride}",
            grad_conv2_out_l_major.len()
        )));
    }
    let nodes = grad_conv2_out_l_major.len() / output_node_stride;
    check_len(
        "fused UMA SO2 VJP conv1_out",
        nodes * conv2_config.input_node_stride(),
        conv1_out_l_major.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP gate_sigmoids",
        nodes * 2 * conv2_config.in_channels,
        gate_sigmoids.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP scalar_silu_derivatives",
        nodes * conv2_config.in_channels,
        scalar_silu_derivatives.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv1_input",
        nodes * conv1_config.input_node_stride(),
        conv1_input_l_major.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP grad_conv1_input",
        conv1_input_l_major.len(),
        grad_conv1_input_l_major.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv1_radial",
        nodes * conv1_config.radial_node_stride(),
        conv1_radial.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP grad_conv1_radial",
        conv1_radial.len(),
        grad_conv1_radial.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv2_m0_weight",
        conv2_config.m0_weight_len(),
        conv2_m0_weight.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv1_m0_weight",
        conv1_config.m0_weight_len(),
        conv1_m0_weight.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv2_m_weights",
        conv2_config.m_max,
        conv2_m_weights.len(),
    )?;
    check_len(
        "fused UMA SO2 VJP conv1_m_weights",
        conv1_config.m_max,
        conv1_m_weights.len(),
    )?;
    for order in 1..=2 {
        check_len(
            "fused UMA SO2 VJP conv2_m_weight",
            conv2_config.m_conv_weight_len(order),
            conv2_m_weights[order - 1].len(),
        )?;
        check_len(
            "fused UMA SO2 VJP conv1_m_weight",
            conv1_config.m_conv_weight_len(order),
            conv1_m_weights[order - 1].len(),
        )?;
    }
    check_so2_m_conv_fused_weights(conv2_config, conv2_m_fused_weights)?;
    check_so2_m_conv_fused_weights(conv1_config, conv1_m_fused_weights)?;

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let _ = conv2_m_fused_weights;
        let _ = conv1_m_fused_weights;
    }

    L2_M2_RADIAL_GATE_CONV2_VJP_SCRATCH.with(|scratch| -> Result<()> {
        let mut scratch = scratch.borrow_mut();
        let L2M2RadialGateConv2VjpScratch {
            conv1_m0_full_grad_out,
            conv2_m0_grad_out,
            activated_m0_grad,
            conv2_block_grad_out,
            activated_block_grad,
            conv1_block_grad_out,
            conv1_block_grad_modulated,
            conv1_m0_grad_modulated,
        } = &mut *scratch;

        let channels = conv2_config.in_channels;
        let conv1_in_channels = conv1_config.in_channels;
        reset_zeroed_f64_vec(
            conv1_m0_full_grad_out,
            nodes * conv1_config.m0_output_feature_dim(),
        );
        let conv1_m0_full_grad_out = &mut conv1_m0_full_grad_out[..];

        reset_zeroed_f64_vec(
            conv2_m0_grad_out,
            nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv2_config.out_channels,
        );
        let conv2_m0_grad_out = &mut conv2_m0_grad_out[..];
        gather_l2_m2_order_block(
            nodes,
            conv2_config.out_channels,
            &L2_M2_ORDER0_L_MAJOR_SOURCES,
            grad_conv2_out_l_major,
            conv2_m0_grad_out,
        );
        reset_zeroed_f64_vec(
            activated_m0_grad,
            nodes * L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
        );
        let activated_m0_grad = &mut activated_m0_grad[..];
        dense_vjp_input_no_bias_rows_write(
            nodes,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv2_config.out_channels,
            conv2_m0_weight,
            conv2_m0_grad_out,
            activated_m0_grad,
        );
        accumulate_l2_m2_gate_vjp_to_conv1_m0_grad(
            nodes,
            channels,
            activated_m0_grad,
            conv1_out_l_major,
            gate_sigmoids,
            scalar_silu_derivatives,
            conv1_m0_full_grad_out,
        );

        let max_order_sources = L2_M2_ORDER1_L_MAJOR_SOURCES.len();
        reset_zeroed_f64_vec(
            conv2_block_grad_out,
            nodes * max_order_sources * conv2_config.out_channels,
        );
        reset_zeroed_f64_vec(activated_block_grad, nodes * max_order_sources * channels);
        reset_zeroed_f64_vec(conv1_block_grad_out, nodes * max_order_sources * channels);
        reset_zeroed_f64_vec(
            conv1_block_grad_modulated,
            nodes * max_order_sources * conv1_in_channels,
        );

        let mut radial_offset = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv1_in_channels;
        for (order, sources) in [
            (1usize, &L2_M2_ORDER1_L_MAJOR_SOURCES[..]),
            (2usize, &L2_M2_ORDER2_L_MAJOR_SOURCES[..]),
        ] {
            let m_size = conv2_config.m_size_unchecked(order);
            let conv2_block_config =
                EsenSo2MConvBlockConfig::new(m_size, channels, conv2_config.out_channels)?;
            let conv2_block_grad_out_len = nodes * sources.len() * conv2_config.out_channels;
            let conv2_block_grad_out = &mut conv2_block_grad_out[..conv2_block_grad_out_len];
            gather_l2_m2_order_block(
                nodes,
                conv2_config.out_channels,
                sources,
                grad_conv2_out_l_major,
                conv2_block_grad_out,
            );
            let activated_block_grad_len = nodes * sources.len() * channels;
            let activated_block_grad = &mut activated_block_grad[..activated_block_grad_len];
            let conv2_weight = conv2_m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            if let Some(fused_weight) =
                conv2_m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
            {
                fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight_write(
                    conv2_block_config,
                    conv2_weight,
                    fused_weight,
                    conv2_block_grad_out,
                    activated_block_grad,
                )?;
            } else {
                activated_block_grad.fill(0.0);
                fairchem_esen_so2_m_conv_block_vjp(
                    conv2_block_config,
                    conv2_weight,
                    conv2_block_grad_out,
                    activated_block_grad,
                )?;
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            {
                activated_block_grad.fill(0.0);
                fairchem_esen_so2_m_conv_block_vjp(
                    conv2_block_config,
                    conv2_weight,
                    conv2_block_grad_out,
                    activated_block_grad,
                )?;
            }

            let conv1_block_grad_out_len = nodes * sources.len() * channels;
            let conv1_block_grad_out = &mut conv1_block_grad_out[..conv1_block_grad_out_len];
            accumulate_l2_m2_gate_vjp_to_conv1_order_grad(
                nodes,
                channels,
                sources,
                activated_block_grad,
                conv1_out_l_major,
                gate_sigmoids,
                conv1_m0_full_grad_out,
                conv1_block_grad_out,
            );

            let conv1_block_config =
                EsenSo2MConvBlockConfig::new(m_size, conv1_in_channels, channels)?;
            let conv1_block_grad_modulated_len = nodes * sources.len() * conv1_in_channels;
            let conv1_block_grad_modulated =
                &mut conv1_block_grad_modulated[..conv1_block_grad_modulated_len];
            let conv1_weight = conv1_m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            if let Some(fused_weight) =
                conv1_m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
            {
                fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight_write(
                    conv1_block_config,
                    conv1_weight,
                    fused_weight,
                    conv1_block_grad_out,
                    conv1_block_grad_modulated,
                )?;
                accumulate_l2_m2_radial_modulation_vjp_from_order_block(
                    nodes,
                    conv1_in_channels,
                    m_size,
                    radial_offset,
                    sources,
                    conv1_block_grad_modulated,
                    conv1_input_l_major,
                    conv1_radial,
                    grad_conv1_input_l_major,
                    grad_conv1_radial,
                );
                radial_offset += m_size * conv1_in_channels;
                continue;
            }
            conv1_block_grad_modulated.fill(0.0);
            fairchem_esen_so2_m_conv_block_vjp(
                conv1_block_config,
                conv1_weight,
                conv1_block_grad_out,
                conv1_block_grad_modulated,
            )?;
            accumulate_l2_m2_radial_modulation_vjp_from_order_block(
                nodes,
                conv1_in_channels,
                m_size,
                radial_offset,
                sources,
                conv1_block_grad_modulated,
                conv1_input_l_major,
                conv1_radial,
                grad_conv1_input_l_major,
                grad_conv1_radial,
            );
            radial_offset += m_size * conv1_in_channels;
        }

        apply_l2_m2_gate_extra_sigmoid_derivative(
            nodes,
            channels,
            gate_sigmoids,
            conv1_m0_full_grad_out,
        );
        let m0_input_len = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * conv1_in_channels;
        reset_zeroed_f64_vec(conv1_m0_grad_modulated, nodes * m0_input_len);
        let conv1_m0_grad_modulated = &mut conv1_m0_grad_modulated[..];
        dense_vjp_input_no_bias_rows_write(
            nodes,
            m0_input_len,
            conv1_config.m0_output_feature_dim(),
            conv1_m0_weight,
            conv1_m0_full_grad_out,
            conv1_m0_grad_modulated,
        );
        accumulate_l2_m2_radial_modulation_vjp_from_order_block(
            nodes,
            conv1_in_channels,
            L2_M2_ORDER0_L_MAJOR_SOURCES.len(),
            0,
            &L2_M2_ORDER0_L_MAJOR_SOURCES,
            conv1_m0_grad_modulated,
            conv1_input_l_major,
            conv1_radial,
            grad_conv1_input_l_major,
            grad_conv1_radial,
        );

        Ok(())
    })
}

fn accumulate_l2_m2_gate_vjp_to_conv1_m0_grad(
    nodes: usize,
    channels: usize,
    activated_m0_grad: &[f64],
    conv1_out_l_major: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    conv1_m0_full_grad_out: &mut [f64],
) {
    debug_assert_eq!(activated_m0_grad.len(), nodes * 3 * channels);
    debug_assert_eq!(conv1_out_l_major.len(), nodes * 9 * channels);
    debug_assert_eq!(gate_sigmoids.len(), nodes * 2 * channels);
    debug_assert_eq!(scalar_silu_derivatives.len(), nodes * channels);
    debug_assert_eq!(conv1_m0_full_grad_out.len(), nodes * 5 * channels);

    for node_index in 0..nodes {
        let activated_start = node_index * 3 * channels;
        let conv1_start = node_index * 9 * channels;
        let gate_start = node_index * 2 * channels;
        let scalar_deriv_start = node_index * channels;
        let full_grad_start = node_index * 5 * channels;
        for channel in 0..channels {
            let grad_scalar = activated_m0_grad[activated_start + channel];
            conv1_m0_full_grad_out[full_grad_start + 2 * channels + channel] +=
                grad_scalar * scalar_silu_derivatives[scalar_deriv_start + channel];

            let grad_l1_m0 = activated_m0_grad[activated_start + channels + channel];
            let gate_l1 = gate_sigmoids[gate_start + channel];
            let conv1_l1_m0 = conv1_out_l_major[conv1_start + 2 * channels + channel];
            conv1_m0_full_grad_out[full_grad_start + 3 * channels + channel] +=
                grad_l1_m0 * gate_l1;
            conv1_m0_full_grad_out[full_grad_start + channel] += grad_l1_m0 * conv1_l1_m0;

            let grad_l2_m0 = activated_m0_grad[activated_start + 2 * channels + channel];
            let gate_l2 = gate_sigmoids[gate_start + channels + channel];
            let conv1_l2_m0 = conv1_out_l_major[conv1_start + 6 * channels + channel];
            conv1_m0_full_grad_out[full_grad_start + 4 * channels + channel] +=
                grad_l2_m0 * gate_l2;
            conv1_m0_full_grad_out[full_grad_start + channels + channel] +=
                grad_l2_m0 * conv1_l2_m0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_l2_m2_gate_vjp_to_conv1_order_grad(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    activated_block_grad: &[f64],
    conv1_out_l_major: &[f64],
    gate_sigmoids: &[f64],
    conv1_m0_full_grad_out: &mut [f64],
    conv1_block_grad_out: &mut [f64],
) {
    debug_assert_eq!(activated_block_grad.len(), nodes * sources.len() * channels);
    debug_assert_eq!(conv1_out_l_major.len(), nodes * 9 * channels);
    debug_assert_eq!(gate_sigmoids.len(), nodes * 2 * channels);
    debug_assert_eq!(conv1_m0_full_grad_out.len(), nodes * 5 * channels);
    debug_assert_eq!(conv1_block_grad_out.len(), activated_block_grad.len());

    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node_index in 0..nodes {
            let block_start = node_index * 4 * channels;
            let conv1_start = node_index * 9 * channels;
            let gate_start = node_index * 2 * channels;
            let full_grad_start = node_index * 5 * channels;
            for channel in 0..channels {
                let gate_l1 = gate_sigmoids[gate_start + channel];
                let gate_l2 = gate_sigmoids[gate_start + channels + channel];

                let grad3 = activated_block_grad[block_start + channel];
                let grad7 = activated_block_grad[block_start + channels + channel];
                let grad1 = activated_block_grad[block_start + 2 * channels + channel];
                let grad5 = activated_block_grad[block_start + 3 * channels + channel];

                conv1_block_grad_out[block_start + channel] = grad3 * gate_l1;
                conv1_block_grad_out[block_start + channels + channel] = grad7 * gate_l2;
                conv1_block_grad_out[block_start + 2 * channels + channel] = grad1 * gate_l1;
                conv1_block_grad_out[block_start + 3 * channels + channel] = grad5 * gate_l2;

                conv1_m0_full_grad_out[full_grad_start + channel] += grad3
                    * conv1_out_l_major[conv1_start + 3 * channels + channel]
                    + grad1 * conv1_out_l_major[conv1_start + channels + channel];
                conv1_m0_full_grad_out[full_grad_start + channels + channel] += grad7
                    * conv1_out_l_major[conv1_start + 7 * channels + channel]
                    + grad5 * conv1_out_l_major[conv1_start + 5 * channels + channel];
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node_index in 0..nodes {
            let block_start = node_index * 2 * channels;
            let conv1_start = node_index * 9 * channels;
            let gate_start = node_index * 2 * channels;
            let full_grad_start = node_index * 5 * channels;
            for channel in 0..channels {
                let gate_l2 = gate_sigmoids[gate_start + channels + channel];
                let grad8 = activated_block_grad[block_start + channel];
                let grad4 = activated_block_grad[block_start + channels + channel];

                conv1_block_grad_out[block_start + channel] = grad8 * gate_l2;
                conv1_block_grad_out[block_start + channels + channel] = grad4 * gate_l2;
                conv1_m0_full_grad_out[full_grad_start + channels + channel] += grad8
                    * conv1_out_l_major[conv1_start + 8 * channels + channel]
                    + grad4 * conv1_out_l_major[conv1_start + 4 * channels + channel];
            }
        }
        return;
    }

    for node_index in 0..nodes {
        let block_start = node_index * sources.len() * channels;
        let conv1_start = node_index * 9 * channels;
        let gate_start = node_index * 2 * channels;
        let full_grad_start = node_index * 5 * channels;
        for (block_index, &source) in sources.iter().enumerate() {
            let gate_offset = match source {
                1 | 3 => 0,
                4 | 5 | 7 | 8 => channels,
                _ => {
                    debug_assert!(false, "unexpected l=2/m=2 gate source {source}");
                    0
                }
            };
            let source_start = block_start + block_index * channels;
            let conv1_source_start = conv1_start + source * channels;
            for channel in 0..channels {
                let grad = activated_block_grad[source_start + channel];
                let gate = gate_sigmoids[gate_start + gate_offset + channel];
                conv1_block_grad_out[source_start + channel] = grad * gate;
                conv1_m0_full_grad_out[full_grad_start + gate_offset + channel] +=
                    grad * conv1_out_l_major[conv1_source_start + channel];
            }
        }
    }
}

fn apply_l2_m2_gate_extra_sigmoid_derivative(
    nodes: usize,
    channels: usize,
    gate_sigmoids: &[f64],
    conv1_m0_full_grad_out: &mut [f64],
) {
    debug_assert_eq!(gate_sigmoids.len(), nodes * 2 * channels);
    debug_assert_eq!(conv1_m0_full_grad_out.len(), nodes * 5 * channels);

    for node_index in 0..nodes {
        let gate_start = node_index * 2 * channels;
        let full_grad_start = node_index * 5 * channels;
        for channel in 0..channels {
            let gate_l1 = gate_sigmoids[gate_start + channel];
            conv1_m0_full_grad_out[full_grad_start + channel] *= gate_l1 * (1.0 - gate_l1);

            let gate_l2 = gate_sigmoids[gate_start + channels + channel];
            conv1_m0_full_grad_out[full_grad_start + channels + channel] *=
                gate_l2 * (1.0 - gate_l2);
        }
    }
}

/// FairChem eSEN `SO2_Convolution` assembly using a pre-analyzed static plan.
///
/// The plan removes repeated coefficient-permutation and per-order offset
/// construction from fixed-architecture inference hot paths. It preserves the
/// checked public API semantics and falls back to the generic path for small or
/// serial cases where the existing BLAS-backed implementation is faster.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared",
        config,
        input_l_major,
        radial,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_l_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            None,
        );
    }

    fairchem_esen_so2_convolution_forward_with_fused_m_weights(
        config,
        input_l_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        m_fused_weights,
        out_l_major,
        extra_m0_out,
    )
}

/// Prepared SO2 convolution forward with a per-node scale fused into the
/// l-major output scatter.
///
/// This is intended for fixed-architecture inference paths where the next
/// operation is an edgewise row scale, such as eSEN-OAM's conv2 output envelope.
/// The scale is applied only to `out_l_major`; `extra_m0_out` remains unscaled.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared_scaled(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    output_scales: &[f64],
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared_scaled",
        config,
        input_l_major,
        radial,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_forward_prepared_scaled output_scales",
        nodes,
        output_scales.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_l_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            Some(output_scales),
        );
    }

    fairchem_esen_so2_convolution_forward_with_fused_m_weights(
        config,
        input_l_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        m_fused_weights,
        out_l_major,
        extra_m0_out,
    )?;
    scale_l_major_rows(
        nodes,
        config.output_node_stride(),
        output_scales,
        out_l_major,
    );
    Ok(())
}

/// Prepared SO2 convolution forward whose input is already in FairChem eSEN
/// m-major coefficient order and whose output is returned in l-major order.
///
/// This lets fixed-architecture hot paths keep an intermediate tensor in
/// m-major order across operations that do not require l-major storage.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    output_scales: &[f64],
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled",
        config,
        input_m_major,
        radial,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled output_scales",
        nodes,
        output_scales.len(),
    )?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_m_major_input_l_major_out_direct_parallel(
            config,
            nodes,
            input_m_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            None,
            output_scales,
            out_l_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
        );
    }

    let mut out_m_major = vec![0.0; out_l_major.len()];
    run_so2_convolution_assembly_forward_m_major(
        config,
        nodes,
        input_m_major,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        &mut out_m_major,
        extra_m0_out,
    )?;
    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;
    fairchem_esen_m_to_l_forward(mapping, config.out_channels, &out_m_major, out_l_major)?;
    scale_l_major_rows(
        nodes,
        config.output_node_stride(),
        output_scales,
        out_l_major,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source<F>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    radial_block_source: F,
) -> Result<()>
where
    F: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source",
        config,
        input_l_major,
        None,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_l_major_direct_parallel_with_radial_block_source(
            config,
            nodes,
            input_l_major,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_l_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &radial_block_source,
        );
    }

    let radial = gather_so2_radial_from_block_source(config, nodes, &radial_block_source)?;
    fairchem_esen_so2_convolution_forward_prepared(
        plan,
        input_l_major,
        Some(&radial),
        m0_weight,
        m0_bias,
        m_weights,
        m_fused_weights,
        out_l_major,
        extra_m0_out,
    )
}

/// Prepared SO2 convolution forward with l-major input and m-major output.
///
/// The radial values are streamed in per-order blocks. For large parallel
/// workloads this avoids the l-major output scatter at the conv1 -> gate
/// boundary; small/serial workloads use the checked l-major path followed by an
/// explicit l->m coefficient remapping.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source_m_major_out<F>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_m_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    radial_block_source: F,
) -> Result<()>
where
    F: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source_m_major_out",
        config,
        input_l_major,
        None,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_m_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_m_major_out_direct_parallel_with_radial_block_source(
            config,
            nodes,
            input_l_major,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_m_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &radial_block_source,
        );
    }

    let radial = gather_so2_radial_from_block_source(config, nodes, &radial_block_source)?;
    let mut out_l_major = vec![0.0; out_m_major.len()];
    fairchem_esen_so2_convolution_forward_prepared(
        plan,
        input_l_major,
        Some(&radial),
        m0_weight,
        m0_bias,
        m_weights,
        m_fused_weights,
        &mut out_l_major,
        extra_m0_out,
    )?;
    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;
    fairchem_esen_l_to_m_forward(mapping, config.out_channels, &out_l_major, out_m_major)
}

#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_forward_prepared_with_radial_strided_block_source<F>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    radial_strided_block_source: F,
) -> Result<()>
where
    F: Fn(usize, usize, usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    let config = plan.config;
    let extra_m0_like = extra_m0_out.as_ref().map(|values| &values[..]);
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_forward_prepared_with_radial_strided_block_source",
        config,
        input_l_major,
        None,
        m0_weight,
        Some(m0_bias),
        m_weights,
        out_l_major,
        extra_m0_like,
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_forward_l_major_direct_parallel_with_radial_strided_block_source(
            config,
            nodes,
            input_l_major,
            m0_weight,
            m0_bias,
            m_weights,
            m_fused_weights,
            out_l_major,
            extra_m0_out,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &radial_strided_block_source,
        );
    }

    let radial =
        gather_so2_radial_from_strided_block_source(config, nodes, &radial_strided_block_source)?;
    fairchem_esen_so2_convolution_forward_prepared(
        plan,
        input_l_major,
        Some(&radial),
        m0_weight,
        m0_bias,
        m_weights,
        m_fused_weights,
        out_l_major,
        extra_m0_out,
    )
}

/// VJP for [`fairchem_esen_so2_convolution_forward_prepared`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared",
        config,
        input_l_major,
        radial,
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    if let Some(radial_values) = radial {
        let Some(grad_radial_values) = grad_radial.as_deref() else {
            return Err(invalid_so2(
                "grad_radial is required when radial is present",
            ));
        };
        check_len(
            "fairchem_esen_so2_convolution_vjp_prepared grad_radial",
            radial_values.len(),
            grad_radial_values.len(),
        )?;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 {
        return run_so2_convolution_assembly_vjp_l_major_direct_parallel(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_l_major,
            grad_extra_m0,
            grad_input_l_major,
            grad_radial,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
        );
    }

    fairchem_esen_so2_convolution_vjp_with_fused_m_weights(
        config,
        input_l_major,
        radial,
        m0_weight,
        m_weights,
        m_fused_weights,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

/// VJP for
/// [`fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled`]
/// without the row scale. The input and input cotangent are in m-major order;
/// the output cotangent remains in the usual l-major order.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared_m_major_input(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_m_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_input",
        config,
        input_m_major,
        radial,
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_input grad_input_m_major",
        input_m_major.len(),
        grad_input_m_major.len(),
    )?;
    if let Some(radial_values) = radial {
        let Some(grad_radial_values) = grad_radial.as_deref() else {
            return Err(invalid_so2(
                "grad_radial is required when radial is present",
            ));
        };
        check_len(
            "fairchem_esen_so2_convolution_vjp_prepared_m_major_input grad_radial",
            radial_values.len(),
            grad_radial_values.len(),
        )?;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 && radial.is_none() {
        return run_so2_convolution_assembly_vjp_m_major_input_l_major_grad_out_direct_parallel(
            config,
            nodes,
            m0_weight,
            m_weights,
            grad_out_l_major,
            grad_extra_m0,
            grad_input_m_major,
            &plan.coeff_offsets,
            &plan.l_to_m_permutation,
        );
    }

    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;
    let mut grad_out_m_major = vec![0.0; grad_out_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels,
        grad_out_l_major,
        &mut grad_out_m_major,
    )?;
    run_so2_convolution_assembly_vjp_m_major(
        config,
        nodes,
        input_m_major,
        radial,
        m0_weight,
        m_weights,
        &grad_out_m_major,
        grad_extra_m0,
        grad_input_m_major,
        grad_radial,
    )
}

/// VJP for [`fairchem_esen_so2_convolution_forward_prepared`] when the output
/// cotangent is already stored in FairChem eSEN m-major coefficient order.
///
/// This is intended for fused fixed-model hot paths that can produce
/// order-major output cotangents directly and want to avoid materializing a
/// full l-major cotangent only for SO2 to gather it back into order blocks.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out",
        config,
        input_l_major,
        radial,
        m0_weight,
        None,
        m_weights,
        grad_out_m_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    if let Some(radial_values) = radial {
        let Some(grad_radial_values) = grad_radial.as_deref() else {
            return Err(invalid_so2(
                "grad_radial is required when radial is present",
            ));
        };
        check_len(
            "fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out grad_radial",
            radial_values.len(),
            grad_radial_values.len(),
        )?;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if nodes >= so2_parallel_node_threshold() && config.m_max > 0 && radial.is_none() {
        return run_so2_convolution_assembly_vjp_m_major_grad_out_l_major_input_direct_parallel(
            config,
            nodes,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_m_major,
            grad_extra_m0,
            grad_input_l_major,
            &plan.coeff_offsets,
            &plan.l_to_m_permutation,
        );
    }

    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;
    let mut grad_out_l_major = vec![0.0; grad_out_m_major.len()];
    crate::so3::fairchem_esen_m_to_l_forward(
        mapping,
        config.out_channels,
        grad_out_m_major,
        &mut grad_out_l_major,
    )?;
    fairchem_esen_so2_convolution_vjp_prepared(
        plan,
        input_l_major,
        radial,
        m0_weight,
        m_weights,
        m_fused_weights,
        &grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

/// VJP for [`fairchem_esen_so2_convolution_forward_prepared`] that streams
/// radial-output cotangents in contiguous per-order blocks.
///
/// This is intended for fixed-model hot paths that immediately feed the
/// radial cotangent into another dense VJP and do not need a materialized
/// full-width radial-gradient buffer.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink<F>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    radial: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    mut radial_block_sink: F,
) -> Result<()>
where
    F: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink",
        config,
        input_l_major,
        Some(radial),
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        return run_so2_convolution_assembly_vjp_l_major_direct_parallel_with_radial_block_sink(
            config,
            nodes,
            input_l_major,
            radial,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_l_major,
            grad_extra_m0,
            grad_input_l_major,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &mut radial_block_sink,
        );
    }

    let mut grad_radial = vec![0.0; radial.len()];
    fairchem_esen_so2_convolution_vjp_prepared(
        plan,
        input_l_major,
        Some(radial),
        m0_weight,
        m_weights,
        m_fused_weights,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        Some(&mut grad_radial),
    )?;
    radial_block_sink(0, config.radial_node_stride(), &grad_radial)
}

/// VJP for [`fairchem_esen_so2_convolution_forward_prepared`] that streams both
/// radial-output values and radial-output cotangents in contiguous per-order
/// blocks.
///
/// This avoids materializing full `[nodes, radial_node_stride]` radial values
/// and radial cotangents when the caller can regenerate radial output blocks
/// from a fixed final-linear input cache.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink<S, T>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    radial_block_source: S,
    #[cfg_attr(not(feature = "parallel"), allow(unused_mut))] mut radial_block_sink: T,
) -> Result<()>
where
    S: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
    T: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink",
        config,
        input_l_major,
        None,
        m0_weight,
        None,
        m_weights,
        grad_out_l_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        return run_so2_convolution_assembly_vjp_l_major_direct_parallel_with_radial_block_source_and_sink(
            config,
            nodes,
            input_l_major,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_l_major,
            grad_extra_m0,
            grad_input_l_major,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &radial_block_source,
            &mut radial_block_sink,
        );
    }

    let radial = gather_so2_radial_from_block_source(config, nodes, &radial_block_source)?;
    fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink(
        plan,
        input_l_major,
        &radial,
        m0_weight,
        m_weights,
        m_fused_weights,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        radial_block_sink,
    )
}

/// VJP for [`fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source_m_major_out`].
///
/// `input_l_major` remains in the normal l-major storage order, while
/// `grad_out_m_major` is the cotangent of the m-major output. This lets fixed
/// eSEN hot paths keep the gate/second-convolution tensor in m-major order and
/// pass its cotangent directly into the first convolution VJP.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out_with_radial_block_source_and_sink<
    S,
    T,
>(
    plan: &EsenSo2ConvolutionPreparedPlan,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    radial_block_source: S,
    #[cfg_attr(not(feature = "parallel"), allow(unused_mut))] mut radial_block_sink: T,
) -> Result<()>
where
    S: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
    T: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    let config = plan.config;
    let nodes = check_so2_convolution_assembly_shapes(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out_with_radial_block_source_and_sink",
        config,
        input_l_major,
        None,
        m0_weight,
        None,
        m_weights,
        grad_out_m_major,
        grad_extra_m0,
    )?;
    check_len(
        "fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out_with_radial_block_source_and_sink grad_input_l_major",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_so2_m_conv_fused_weights(config, m_fused_weights)?;
    #[cfg(not(feature = "parallel"))]
    let _ = nodes;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        return run_so2_convolution_assembly_vjp_m_major_grad_out_l_major_input_direct_parallel_with_radial_block_source_and_sink(
            config,
            nodes,
            input_l_major,
            m0_weight,
            m_weights,
            m_fused_weights,
            grad_out_m_major,
            grad_extra_m0,
            grad_input_l_major,
            &plan.coeff_offsets,
            &plan.radial_offsets,
            &plan.l_to_m_permutation,
            &radial_block_source,
            &mut radial_block_sink,
        );
    }

    let mapping = EsenCoefficientMappingConfig::new(config.l_max, config.m_max)
        .map_err(|err| invalid_so2(err.to_string()))?;
    let mut grad_out_l_major = vec![0.0; grad_out_m_major.len()];
    crate::so3::fairchem_esen_m_to_l_forward(
        mapping,
        config.out_channels,
        grad_out_m_major,
        &mut grad_out_l_major,
    )?;
    fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink(
        plan,
        input_l_major,
        m0_weight,
        m_weights,
        m_fused_weights,
        &grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        radial_block_source,
        radial_block_sink,
    )
}

impl EsenSo2MConvBlockConfig {
    pub fn new(coefficient_count: usize, in_channels: usize, out_channels: usize) -> Result<Self> {
        if coefficient_count == 0 || in_channels == 0 || out_channels == 0 {
            return Err(invalid_so2(format!(
                "dimensions must be nonzero, got coefficients={coefficient_count}, channels={in_channels}->{out_channels}"
            )));
        }
        coefficient_count
            .checked_mul(in_channels)
            .and_then(|input_len| {
                coefficient_count
                    .checked_mul(out_channels)
                    .and_then(|output_len| input_len.checked_mul(output_len))
                    .and_then(|dense_len| dense_len.checked_mul(2))
            })
            .ok_or_else(|| invalid_so2("SO2 m-conv block dimensions overflow"))?;
        Ok(Self {
            coefficient_count,
            in_channels,
            out_channels,
        })
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[must_use]
    pub fn input_copy_stride(&self) -> usize {
        self.coefficient_count * self.in_channels
    }

    #[must_use]
    pub fn output_copy_stride(&self) -> usize {
        self.coefficient_count * self.out_channels
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        2 * self.input_copy_stride()
    }

    #[must_use]
    pub fn output_node_stride(&self) -> usize {
        2 * self.output_copy_stride()
    }

    #[must_use]
    pub fn weight_len(&self) -> usize {
        2 * self.output_copy_stride() * self.input_copy_stride()
    }
}

impl EsenSo2LinearBlockConfig {
    pub fn new(
        coefficient_count: usize,
        in_channels: usize,
        out_channels: usize,
        copies: usize,
    ) -> Result<Self> {
        if coefficient_count == 0 || in_channels == 0 || out_channels == 0 {
            return Err(invalid_so2(format!(
                "dimensions must be nonzero, got coefficients={coefficient_count}, channels={in_channels}->{out_channels}"
            )));
        }
        if copies != 1 && copies != 2 {
            return Err(invalid_so2(format!(
                "copies must be 1 for m=0 or 2 for m>0, got {copies}"
            )));
        }
        coefficient_count
            .checked_mul(in_channels)
            .and_then(|input_len| {
                coefficient_count
                    .checked_mul(out_channels)
                    .and_then(|output_len| input_len.checked_mul(output_len))
            })
            .ok_or_else(|| invalid_so2("SO2 block dimensions overflow"))?;
        Ok(Self {
            coefficient_count,
            in_channels,
            out_channels,
            copies,
        })
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[must_use]
    pub fn copies(&self) -> usize {
        self.copies
    }

    #[must_use]
    pub fn input_copy_stride(&self) -> usize {
        self.coefficient_count * self.in_channels
    }

    #[must_use]
    pub fn output_copy_stride(&self) -> usize {
        self.coefficient_count * self.out_channels
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.copies * self.input_copy_stride()
    }

    #[must_use]
    pub fn output_node_stride(&self) -> usize {
        self.copies * self.output_copy_stride()
    }

    #[must_use]
    pub fn weight_len(&self) -> usize {
        self.output_copy_stride() * self.input_copy_stride()
    }
}

/// FairChem eSEN SO2 radial modulation before a per-m dense block.
///
/// `input` is flat `[nodes, copies, feature_dim]`, while `radial` is flat
/// `[nodes, feature_dim]`. The same radial slice is applied to each copy.
pub fn fairchem_esen_so2_radial_modulation_forward(
    config: EsenSo2RadialModulationConfig,
    input: &[f64],
    radial: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_radial_modulation_shapes(
        "fairchem_esen_so2_radial_modulation_forward",
        config,
        input,
        radial,
        out,
    )?;
    for node_index in 0..nodes {
        let input_node_start = node_index * config.input_node_stride();
        let radial_node_start = node_index * config.feature_dim;
        for copy_index in 0..config.copies {
            let input_copy_start = input_node_start + copy_index * config.feature_dim;
            for feature in 0..config.feature_dim {
                out[input_copy_start + feature] =
                    input[input_copy_start + feature] * radial[radial_node_start + feature];
            }
        }
    }
    Ok(())
}

/// Accumulate input and radial VJPs of
/// [`fairchem_esen_so2_radial_modulation_forward`].
pub fn fairchem_esen_so2_radial_modulation_vjp(
    config: EsenSo2RadialModulationConfig,
    input: &[f64],
    radial: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
    grad_radial: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_radial_modulation_shapes(
        "fairchem_esen_so2_radial_modulation_vjp",
        config,
        input,
        radial,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_so2_radial_modulation_vjp grad_input",
        input.len(),
        grad_input.len(),
    )?;
    check_len(
        "fairchem_esen_so2_radial_modulation_vjp grad_radial",
        radial.len(),
        grad_radial.len(),
    )?;

    for node_index in 0..nodes {
        let input_node_start = node_index * config.input_node_stride();
        let radial_node_start = node_index * config.feature_dim;
        for copy_index in 0..config.copies {
            let input_copy_start = input_node_start + copy_index * config.feature_dim;
            for feature in 0..config.feature_dim {
                let input_index = input_copy_start + feature;
                let radial_index = radial_node_start + feature;
                let grad = grad_out[input_index];
                grad_input[input_index] += radial[radial_index] * grad;
                grad_radial[radial_index] += input[input_index] * grad;
            }
        }
    }
    Ok(())
}

/// FairChem eSEN SO2 linear block with the preceding radial modulation fused
/// into one forward primitive.
pub fn fairchem_esen_so2_linear_block_with_radial_forward(
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    out: &mut [f64],
) -> Result<()> {
    let radial_config =
        EsenSo2RadialModulationConfig::new(config.input_copy_stride(), config.copies)?;
    let mut modulated = vec![0.0; input.len()];
    fairchem_esen_so2_radial_modulation_forward(radial_config, input, radial, &mut modulated)?;
    fairchem_esen_so2_linear_block_forward(config, &modulated, weight, bias, out)
}

/// Accumulate input and radial VJPs of
/// [`fairchem_esen_so2_linear_block_with_radial_forward`].
pub fn fairchem_esen_so2_linear_block_with_radial_vjp(
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
    grad_radial: &mut [f64],
) -> Result<()> {
    let radial_config =
        EsenSo2RadialModulationConfig::new(config.input_copy_stride(), config.copies)?;
    let mut modulated = vec![0.0; input.len()];
    fairchem_esen_so2_radial_modulation_forward(radial_config, input, radial, &mut modulated)?;
    let mut grad_modulated = vec![0.0; input.len()];
    fairchem_esen_so2_linear_block_vjp(config, weight, grad_out, &mut grad_modulated)?;
    fairchem_esen_so2_radial_modulation_vjp(
        radial_config,
        input,
        radial,
        &grad_modulated,
        grad_input,
        grad_radial,
    )
}

/// FairChem eSEN SO2 m-conv block with the preceding radial modulation fused
/// into one forward primitive.
pub fn fairchem_esen_so2_m_conv_block_with_radial_forward(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let radial_config = EsenSo2RadialModulationConfig::new(config.input_copy_stride(), 2)?;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        check_so2_radial_modulation_shapes(
            "fairchem_esen_so2_m_conv_block_with_radial_forward",
            radial_config,
            input,
            radial,
            input,
        )?;
        let nodes = check_so2_m_conv_block_shapes(
            "fairchem_esen_so2_m_conv_block_with_radial_forward",
            config,
            input,
            weight,
            out,
        )?;
        fairchem_esen_so2_m_conv_block_with_radial_forward_blas(
            nodes,
            config,
            config.input_copy_stride(),
            config.output_copy_stride(),
            input,
            radial,
            weight,
            out,
        )
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let mut modulated = vec![0.0; input.len()];
        fairchem_esen_so2_radial_modulation_forward(radial_config, input, radial, &mut modulated)?;
        fairchem_esen_so2_m_conv_block_forward(config, &modulated, weight, out)
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, dead_code)]
fn fairchem_esen_so2_m_conv_block_with_radial_forward_l_major_direct_to_block(
    nodes: usize,
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    m_size: usize,
    coefficient_offset: usize,
    radial_node_stride: usize,
    radial_offset: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial: &[f64],
    weight: &[f64],
    fused_weight: Option<&[f64]>,
    out: &mut [f64],
) -> Result<()> {
    let input_len = m_size * in_channels;
    let output_len = m_size * out_channels;
    let input_node_stride = coefficient_count * in_channels;

    let mut compact_input = uninit_f64_vec(nodes * 2 * input_len);
    #[cfg(feature = "parallel")]
    if so2_use_parallel_compact_input(nodes) {
        compact_input
            .par_chunks_mut(2 * input_len)
            .zip(input_l_major.par_chunks(input_node_stride))
            .zip(radial.par_chunks(radial_node_stride))
            .for_each(|((compact_node, input_node), radial_node)| {
                let (compact_real, compact_imag) = compact_node.split_at_mut(input_len);
                for block_coefficient in 0..m_size {
                    let real_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let imag_coefficient =
                        l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                    let input_real_start = real_coefficient * in_channels;
                    let input_imag_start = imag_coefficient * in_channels;
                    let radial_start = radial_offset + block_coefficient * in_channels;
                    let compact_start = block_coefficient * in_channels;
                    for channel in 0..in_channels {
                        let radial_value = radial_node[radial_start + channel];
                        compact_real[compact_start + channel] =
                            input_node[input_real_start + channel] * radial_value;
                        compact_imag[compact_start + channel] =
                            input_node[input_imag_start + channel] * radial_value;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for node_index in 0..nodes {
            let input_node_start = node_index * input_node_stride;
            let radial_node_start = node_index * radial_node_stride + radial_offset;
            let compact_real_start = node_index * 2 * input_len;
            let compact_imag_start = compact_real_start + input_len;
            for block_coefficient in 0..m_size {
                let real_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
                let imag_coefficient =
                    l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                let input_real_start = input_node_start + real_coefficient * in_channels;
                let input_imag_start = input_node_start + imag_coefficient * in_channels;
                let radial_start = radial_node_start + block_coefficient * in_channels;
                let compact_start = block_coefficient * in_channels;
                for channel in 0..in_channels {
                    let radial_value = radial[radial_start + channel];
                    compact_input[compact_real_start + compact_start + channel] =
                        input_l_major[input_real_start + channel] * radial_value;
                    compact_input[compact_imag_start + compact_start + channel] =
                        input_l_major[input_imag_start + channel] * radial_value;
                }
            }
        }
    }
    #[cfg(feature = "parallel")]
    if so2_use_serial_compact_input(nodes) {
        for node_index in 0..nodes {
            let input_node_start = node_index * input_node_stride;
            let radial_node_start = node_index * radial_node_stride + radial_offset;
            let compact_real_start = node_index * 2 * input_len;
            let compact_imag_start = compact_real_start + input_len;
            for block_coefficient in 0..m_size {
                let real_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
                let imag_coefficient =
                    l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                let input_real_start = input_node_start + real_coefficient * in_channels;
                let input_imag_start = input_node_start + imag_coefficient * in_channels;
                let radial_start = radial_node_start + block_coefficient * in_channels;
                let compact_start = block_coefficient * in_channels;
                for channel in 0..in_channels {
                    let radial_value = radial[radial_start + channel];
                    compact_input[compact_real_start + compact_start + channel] =
                        input_l_major[input_real_start + channel] * radial_value;
                    compact_input[compact_imag_start + compact_start + channel] =
                        input_l_major[input_imag_start + channel] * radial_value;
                }
            }
        }
    }

    if let Some(fused_weight) = fused_weight {
        check_len(
            "fairchem_esen_so2_m_conv_block_with_radial_forward_l_major_direct_to_block fused weight",
            4 * input_len * output_len,
            fused_weight.len(),
        )?;
        accelerate_blas::so2_mconv_forward_precomputed_fused_weight(
            &compact_input,
            nodes,
            input_len,
            fused_weight,
            output_len,
            out,
        );
    } else {
        accelerate_blas::so2_mconv_forward_fused_weight(
            &compact_input,
            nodes,
            input_len,
            weight,
            output_len,
            out,
        );
    }
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, dead_code)]
fn fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block(
    nodes: usize,
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    m_size: usize,
    coefficient_offset: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial_block: &[f64],
    weight: &[f64],
    fused_weight: Option<&[f64]>,
    out: &mut [f64],
) -> Result<()> {
    let input_len = m_size * in_channels;
    let mut compact_input = uninit_f64_vec(nodes * 2 * input_len);
    fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block_with_compact_scratch(
        nodes,
        coefficient_count,
        in_channels,
        out_channels,
        m_size,
        coefficient_offset,
        l_to_m_permutation,
        input_l_major,
        radial_block,
        weight,
        fused_weight,
        out,
        &mut compact_input,
    )
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block_with_compact_scratch(
    nodes: usize,
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    m_size: usize,
    coefficient_offset: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial_block: &[f64],
    weight: &[f64],
    fused_weight: Option<&[f64]>,
    out: &mut [f64],
    compact_input: &mut [f64],
) -> Result<()> {
    let input_len = m_size * in_channels;
    let output_len = m_size * out_channels;
    let input_node_stride = coefficient_count * in_channels;
    check_len(
        "fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block compact scratch",
        nodes * 2 * input_len,
        compact_input.len(),
    )?;

    #[cfg(feature = "parallel")]
    if so2_use_parallel_compact_input(nodes) {
        compact_input
            .par_chunks_mut(2 * input_len)
            .zip(input_l_major.par_chunks(input_node_stride))
            .zip(radial_block.par_chunks(input_len))
            .for_each(|((compact_node, input_node), radial_node)| {
                let (compact_real, compact_imag) = compact_node.split_at_mut(input_len);
                for block_coefficient in 0..m_size {
                    let real_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let imag_coefficient =
                        l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                    let input_real_start = real_coefficient * in_channels;
                    let input_imag_start = imag_coefficient * in_channels;
                    let feature_start = block_coefficient * in_channels;
                    for channel in 0..in_channels {
                        let radial_value = radial_node[feature_start + channel];
                        compact_real[feature_start + channel] =
                            input_node[input_real_start + channel] * radial_value;
                        compact_imag[feature_start + channel] =
                            input_node[input_imag_start + channel] * radial_value;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for node_index in 0..nodes {
            let input_node_start = node_index * input_node_stride;
            let radial_node_start = node_index * input_len;
            let compact_real_start = node_index * 2 * input_len;
            let compact_imag_start = compact_real_start + input_len;
            for block_coefficient in 0..m_size {
                let real_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
                let imag_coefficient =
                    l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                let input_real_start = input_node_start + real_coefficient * in_channels;
                let input_imag_start = input_node_start + imag_coefficient * in_channels;
                let feature_start = block_coefficient * in_channels;
                for channel in 0..in_channels {
                    let radial_value = radial_block[radial_node_start + feature_start + channel];
                    compact_input[compact_real_start + feature_start + channel] =
                        input_l_major[input_real_start + channel] * radial_value;
                    compact_input[compact_imag_start + feature_start + channel] =
                        input_l_major[input_imag_start + channel] * radial_value;
                }
            }
        }
    }
    #[cfg(feature = "parallel")]
    if so2_use_serial_compact_input(nodes) {
        for node_index in 0..nodes {
            let input_node_start = node_index * input_node_stride;
            let radial_node_start = node_index * input_len;
            let compact_real_start = node_index * 2 * input_len;
            let compact_imag_start = compact_real_start + input_len;
            for block_coefficient in 0..m_size {
                let real_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
                let imag_coefficient =
                    l_to_m_permutation[coefficient_offset + m_size + block_coefficient];
                let input_real_start = input_node_start + real_coefficient * in_channels;
                let input_imag_start = input_node_start + imag_coefficient * in_channels;
                let feature_start = block_coefficient * in_channels;
                for channel in 0..in_channels {
                    let radial_value = radial_block[radial_node_start + feature_start + channel];
                    compact_input[compact_real_start + feature_start + channel] =
                        input_l_major[input_real_start + channel] * radial_value;
                    compact_input[compact_imag_start + feature_start + channel] =
                        input_l_major[input_imag_start + channel] * radial_value;
                }
            }
        }
    }

    if let Some(fused_weight) = fused_weight {
        check_len(
            "fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block fused weight",
            4 * input_len * output_len,
            fused_weight.len(),
        )?;
        accelerate_blas::so2_mconv_forward_precomputed_fused_weight(
            compact_input,
            nodes,
            input_len,
            fused_weight,
            output_len,
            out,
        );
    } else {
        accelerate_blas::so2_mconv_forward_fused_weight(
            compact_input,
            nodes,
            input_len,
            weight,
            output_len,
            out,
        );
    }
    Ok(())
}

/// Accumulate input and radial VJPs of
/// [`fairchem_esen_so2_m_conv_block_with_radial_forward`].
pub fn fairchem_esen_so2_m_conv_block_with_radial_vjp(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
    grad_radial: &mut [f64],
) -> Result<()> {
    let radial_config = EsenSo2RadialModulationConfig::new(config.input_copy_stride(), 2)?;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        check_so2_radial_modulation_shapes(
            "fairchem_esen_so2_m_conv_block_with_radial_vjp",
            radial_config,
            input,
            radial,
            input,
        )?;
        let input_node_stride = config.input_node_stride();
        if !grad_input.len().is_multiple_of(input_node_stride) {
            return Err(invalid_so2(format!(
                "fairchem_esen_so2_m_conv_block_with_radial_vjp grad_input length {} is not a multiple of node stride {input_node_stride}",
                grad_input.len()
            )));
        }
        let nodes = grad_input.len() / input_node_stride;
        check_len(
            "fairchem_esen_so2_m_conv_block_with_radial_vjp weight",
            config.weight_len(),
            weight.len(),
        )?;
        check_len(
            "fairchem_esen_so2_m_conv_block_with_radial_vjp grad_out",
            nodes * config.output_node_stride(),
            grad_out.len(),
        )?;
        check_len(
            "fairchem_esen_so2_m_conv_block_with_radial_vjp grad_radial",
            radial.len(),
            grad_radial.len(),
        )?;
        fairchem_esen_so2_m_conv_block_with_radial_vjp_blas(
            nodes,
            config,
            config.input_copy_stride(),
            config.output_copy_stride(),
            input,
            radial,
            weight,
            grad_out,
            grad_input,
            grad_radial,
        )
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let mut modulated = vec![0.0; input.len()];
        fairchem_esen_so2_radial_modulation_forward(radial_config, input, radial, &mut modulated)?;
        let mut grad_modulated = vec![0.0; input.len()];
        fairchem_esen_so2_m_conv_block_vjp(config, weight, grad_out, &mut grad_modulated)?;
        fairchem_esen_so2_radial_modulation_vjp(
            radial_config,
            input,
            radial,
            &grad_modulated,
            grad_input,
            grad_radial,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_so2_linear_assembly_forward_m_major(
    config: EsenSo2LinearConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_m_major: &mut [f64],
) -> Result<()> {
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        let input_feature_dim = m_size * config.in_channels;
        let output_feature_dim = m_size * config.out_channels;
        let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
        let mut block_out = vec![0.0; nodes * copies * output_feature_dim];
        gather_m_major_block(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            coeff_offset,
            copies * m_size,
            input_m_major,
            &mut block_input,
        );
        let block_radial = radial.map(|radial_values| {
            gather_radial_block(
                nodes,
                config.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                radial_values,
            )
        });
        if order == 0 {
            let block_config =
                EsenSo2LinearBlockConfig::new(m_size, config.in_channels, config.out_channels, 1)?;
            if let Some(block_radial_values) = block_radial.as_deref() {
                fairchem_esen_so2_linear_block_with_radial_forward(
                    block_config,
                    &block_input,
                    block_radial_values,
                    m0_weight,
                    Some(m0_bias),
                    &mut block_out,
                )?;
            } else {
                fairchem_esen_so2_linear_block_forward(
                    block_config,
                    &block_input,
                    m0_weight,
                    Some(m0_bias),
                    &mut block_out,
                )?;
            }
        } else {
            let block_config =
                EsenSo2LinearBlockConfig::new(m_size, config.in_channels, config.out_channels, 2)?;
            let weight = m_weights[order - 1];
            if let Some(block_radial_values) = block_radial.as_deref() {
                fairchem_esen_so2_linear_block_with_radial_forward(
                    block_config,
                    &block_input,
                    block_radial_values,
                    weight,
                    None,
                    &mut block_out,
                )?;
            } else {
                fairchem_esen_so2_linear_block_forward(
                    block_config,
                    &block_input,
                    weight,
                    None,
                    &mut block_out,
                )?;
            }
        }
        scatter_m_major_block(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            coeff_offset,
            copies * m_size,
            &block_out,
            out_m_major,
        );
        coeff_offset += copies * m_size;
        radial_offset += input_feature_dim;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_so2_linear_assembly_vjp_m_major(
    config: EsenSo2LinearConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_m_major: &[f64],
    grad_input_m_major: &mut [f64],
    mut grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        let input_feature_dim = m_size * config.in_channels;
        let output_feature_dim = m_size * config.out_channels;
        let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
        let mut block_grad_out = vec![0.0; nodes * copies * output_feature_dim];
        let mut block_grad_input = vec![0.0; block_input.len()];
        gather_m_major_block(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            coeff_offset,
            copies * m_size,
            input_m_major,
            &mut block_input,
        );
        gather_m_major_block(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            coeff_offset,
            copies * m_size,
            grad_out_m_major,
            &mut block_grad_out,
        );
        let block_radial = radial.map(|radial_values| {
            gather_radial_block(
                nodes,
                config.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                radial_values,
            )
        });
        let mut block_grad_radial = block_radial
            .as_ref()
            .map(|block_radial_values| vec![0.0; block_radial_values.len()]);
        if order == 0 {
            let block_config =
                EsenSo2LinearBlockConfig::new(m_size, config.in_channels, config.out_channels, 1)?;
            if let (Some(block_radial_values), Some(block_grad_radial_values)) =
                (block_radial.as_deref(), block_grad_radial.as_deref_mut())
            {
                fairchem_esen_so2_linear_block_with_radial_vjp(
                    block_config,
                    &block_input,
                    block_radial_values,
                    m0_weight,
                    &block_grad_out,
                    &mut block_grad_input,
                    block_grad_radial_values,
                )?;
            } else {
                fairchem_esen_so2_linear_block_vjp(
                    block_config,
                    m0_weight,
                    &block_grad_out,
                    &mut block_grad_input,
                )?;
            }
        } else {
            let block_config =
                EsenSo2LinearBlockConfig::new(m_size, config.in_channels, config.out_channels, 2)?;
            let weight = m_weights[order - 1];
            if let (Some(block_radial_values), Some(block_grad_radial_values)) =
                (block_radial.as_deref(), block_grad_radial.as_deref_mut())
            {
                fairchem_esen_so2_linear_block_with_radial_vjp(
                    block_config,
                    &block_input,
                    block_radial_values,
                    weight,
                    &block_grad_out,
                    &mut block_grad_input,
                    block_grad_radial_values,
                )?;
            } else {
                fairchem_esen_so2_linear_block_vjp(
                    block_config,
                    weight,
                    &block_grad_out,
                    &mut block_grad_input,
                )?;
            }
        }
        scatter_add_m_major_block(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            coeff_offset,
            copies * m_size,
            &block_grad_input,
            grad_input_m_major,
        );
        if let (Some(block_grad_radial_values), Some(grad_radial_values)) =
            (block_grad_radial.as_deref(), grad_radial.as_deref_mut())
        {
            scatter_add_radial_block(
                nodes,
                config.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                block_grad_radial_values,
                grad_radial_values,
            );
        }
        coeff_offset += copies * m_size;
        radial_offset += input_feature_dim;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_linear_assembly_forward_l_major_direct_parallel(
    config: EsenSo2LinearConfig,
    nodes: usize,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_l_major: &mut [f64],
    l_to_m_permutation: &[usize],
) -> Result<()> {
    let mut coeff_offsets = Vec::with_capacity(config.m_max + 1);
    let mut radial_offsets = Vec::with_capacity(config.m_max + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.in_channels;
    }

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
            let mut block_out = vec![0.0; nodes * copies * output_feature_dim];
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                input_l_major,
                &mut block_input,
            );
            let block_radial = radial.map(|radial_values| {
                gather_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                )
            });
            let block_config = EsenSo2LinearBlockConfig::new(
                m_size,
                config.in_channels,
                config.out_channels,
                copies,
            )?;
            let (weight, bias) = if order == 0 {
                (m0_weight, Some(m0_bias))
            } else {
                (m_weights[order - 1], None)
            };
            if let Some(block_radial_values) = block_radial.as_deref() {
                fairchem_esen_so2_linear_block_with_radial_forward(
                    block_config,
                    &block_input,
                    block_radial_values,
                    weight,
                    bias,
                    &mut block_out,
                )?;
            } else {
                fairchem_esen_so2_linear_block_forward(
                    block_config,
                    &block_input,
                    weight,
                    bias,
                    &mut block_out,
                )?;
            }
            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: copies * output_feature_dim,
                source_feature_offset: 0,
                extra_m0_feature_offset: None,
                block_out,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_l_major_block_from_m_permutation_strided(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            result.coeff_offset,
            result.block_coefficients,
            result.source_node_stride,
            result.source_feature_offset,
            l_to_m_permutation,
            &result.block_out,
            out_l_major,
        );
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_linear_assembly_vjp_l_major_direct_parallel(
    config: EsenSo2LinearConfig,
    nodes: usize,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_l_major: &[f64],
    grad_input_l_major: &mut [f64],
    mut grad_radial: Option<&mut [f64]>,
    l_to_m_permutation: &[usize],
) -> Result<()> {
    let mut coeff_offsets = Vec::with_capacity(config.m_max + 1);
    let mut radial_offsets = Vec::with_capacity(config.m_max + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.in_channels;
    }

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let block_len = nodes * copies * input_feature_dim;
            let mut block_grad_out = vec![0.0; nodes * copies * output_feature_dim];
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                grad_out_l_major,
                &mut block_grad_out,
            );
            let block_config = EsenSo2LinearBlockConfig::new(
                m_size,
                config.in_channels,
                config.out_channels,
                copies,
            )?;
            let weight = if order == 0 {
                m0_weight
            } else {
                m_weights[order - 1]
            };
            let mut compact_grad = vec![0.0; block_len];
            fairchem_esen_so2_linear_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut compact_grad,
            )?;
            let (block_grad_input, compact_grad_modulated) = if radial.is_some() {
                (Vec::new(), Some(compact_grad))
            } else {
                (compact_grad, None)
            };
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input,
                block_grad_radial: None,
                compact_grad_modulated,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        if let Some(compact_grad_modulated) = result.compact_grad_modulated.as_deref() {
            let (Some(radial_values), Some(grad_radial_values)) =
                (radial, grad_radial.as_deref_mut())
            else {
                return Err(invalid_so2(
                    "validated radial values and radial grad are missing",
                ));
            };
            scatter_add_l_major_radial_vjp_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                config.radial_node_stride(),
                result.coeff_offset,
                result.block_coefficients,
                result.input_feature_dim,
                result.radial_offset,
                l_to_m_permutation,
                input_l_major,
                radial_values,
                compact_grad_modulated,
                grad_input_l_major,
                grad_radial_values,
            );
        } else {
            scatter_add_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                result.coeff_offset,
                result.block_coefficients,
                l_to_m_permutation,
                &result.block_grad_input,
                grad_input_l_major,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_m_major(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_m_major: &mut [f64],
    mut extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        return run_so2_convolution_assembly_forward_m_major_parallel(
            config,
            nodes,
            input_m_major,
            radial,
            m0_weight,
            m0_bias,
            m_weights,
            out_m_major,
            extra_m0_out,
        );
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        let input_feature_dim = m_size * config.in_channels;
        let output_feature_dim = m_size * config.out_channels;
        let mut block_input = Vec::new();
        let mut block_out = Vec::new();
        let modulated_input = if let Some(radial_values) = radial {
            let mut modulated = vec![0.0; nodes * copies * input_feature_dim];
            gather_m_major_radial_modulated_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                config.radial_node_stride(),
                coeff_offset,
                copies * m_size,
                m_size,
                radial_offset,
                input_m_major,
                radial_values,
                &mut modulated,
            );
            Some(modulated)
        } else {
            block_input = vec![0.0; nodes * copies * input_feature_dim];
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offset,
                copies * m_size,
                input_m_major,
                &mut block_input,
            );
            None
        };
        if order == 0 {
            let dense_input = modulated_input.as_deref().unwrap_or(&block_input);
            let mut dense_out = vec![0.0; nodes * config.m0_output_feature_dim()];
            dense_forward_with_bias_rows(
                nodes,
                input_feature_dim,
                config.m0_output_feature_dim(),
                dense_input,
                m0_weight,
                m0_bias,
                &mut dense_out,
            );
            block_out = dense_out;
        } else {
            block_out.resize(nodes * copies * output_feature_dim, 0.0);
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            if let Some(modulated) = modulated_input.as_deref() {
                fairchem_esen_so2_m_conv_block_forward(
                    block_config,
                    modulated,
                    weight,
                    &mut block_out,
                )?;
            } else {
                fairchem_esen_so2_m_conv_block_forward(
                    block_config,
                    &block_input,
                    weight,
                    &mut block_out,
                )?;
            }
        }
        let source_node_stride = if order == 0 {
            config.m0_output_feature_dim()
        } else {
            copies * output_feature_dim
        };
        let source_feature_offset = if order == 0 {
            config.extra_m0_output_channels
        } else {
            0
        };
        scatter_m_major_block_strided(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            coeff_offset,
            copies * m_size,
            source_node_stride,
            source_feature_offset,
            &block_out,
            out_m_major,
        );
        if order == 0 && config.extra_m0_output_channels > 0 {
            let Some(extra_values) = extra_m0_out.as_mut() else {
                return Err(invalid_so2("validated extra m0 output is missing"));
            };
            copy_extra_m0_from_strided_source(
                nodes,
                config.extra_m0_output_channels,
                source_node_stride,
                0,
                &block_out,
                extra_values,
            );
        }
        coeff_offset += copies * m_size;
        radial_offset += input_feature_dim;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_m_major(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_m_major: &mut [f64],
    mut grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && nodes >= so2_parallel_node_threshold()
        && config.m_max > 0
    {
        return run_so2_convolution_assembly_vjp_m_major_parallel(
            config,
            nodes,
            input_m_major,
            radial,
            m0_weight,
            m_weights,
            grad_out_m_major,
            grad_extra_m0,
            grad_input_m_major,
            grad_radial,
        );
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        let input_feature_dim = m_size * config.in_channels;
        let output_feature_dim = m_size * config.out_channels;

        if order == 0 {
            let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
            for node in 0..nodes {
                let full_start = node * config.m0_output_feature_dim();
                if config.extra_m0_output_channels > 0 {
                    let Some(extra_values) = grad_extra_m0 else {
                        return Err(invalid_so2("validated extra m0 grad is missing"));
                    };
                    let extra_start = node * config.extra_m0_output_channels;
                    full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                        .copy_from_slice(
                            &extra_values
                                [extra_start..extra_start + config.extra_m0_output_channels],
                        );
                }
                let main_start = full_start + config.extra_m0_output_channels;
                let block_start =
                    node * config.output_node_stride() + coeff_offset * config.out_channels;
                full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                    &grad_out_m_major[block_start..block_start + output_feature_dim],
                );
            }
            let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
            dense_vjp_input_no_bias_rows_add(
                nodes,
                input_feature_dim,
                config.m0_output_feature_dim(),
                m0_weight,
                &full_grad_out,
                &mut grad_modulated,
            );

            if let Some(radial_values) = radial {
                let Some(grad_radial_values) = grad_radial.as_deref_mut() else {
                    return Err(invalid_so2(
                        "validated radial values and radial grad are missing",
                    ));
                };
                scatter_add_m_major_radial_vjp_block(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    config.radial_node_stride(),
                    coeff_offset,
                    m_size,
                    input_feature_dim,
                    radial_offset,
                    input_m_major,
                    radial_values,
                    &grad_modulated,
                    grad_input_m_major,
                    grad_radial_values,
                );
            } else {
                scatter_add_m_major_block(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offset,
                    m_size,
                    &grad_modulated,
                    grad_input_m_major,
                );
            }
            coeff_offset += copies * m_size;
            radial_offset += input_feature_dim;
            continue;
        }

        let mut block_grad_out = vec![0.0; nodes * copies * output_feature_dim];
        gather_m_major_block(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            coeff_offset,
            copies * m_size,
            grad_out_m_major,
            &mut block_grad_out,
        );

        let block_config =
            EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
        let weight = m_weights[order - 1];
        let mut block_grad_input = vec![0.0; nodes * copies * input_feature_dim];
        fairchem_esen_so2_m_conv_block_vjp(
            block_config,
            weight,
            &block_grad_out,
            &mut block_grad_input,
        )?;
        if let Some(radial_values) = radial {
            let Some(grad_radial_values) = grad_radial.as_deref_mut() else {
                return Err(invalid_so2(
                    "validated radial values and radial grad are missing",
                ));
            };
            scatter_add_m_major_radial_vjp_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                config.radial_node_stride(),
                coeff_offset,
                copies * m_size,
                input_feature_dim,
                radial_offset,
                input_m_major,
                radial_values,
                &block_grad_input,
                grad_input_m_major,
                grad_radial_values,
            );
        } else {
            scatter_add_m_major_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offset,
                copies * m_size,
                &block_grad_input,
                grad_input_m_major,
            );
        }
        coeff_offset += copies * m_size;
        radial_offset += input_feature_dim;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
struct So2ConvForwardOrderResult {
    coeff_offset: usize,
    block_coefficients: usize,
    source_node_stride: usize,
    source_feature_offset: usize,
    extra_m0_feature_offset: Option<usize>,
    block_out: Vec<f64>,
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_m_major_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    out_m_major: &mut [f64],
    mut extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let mut coeff_offsets = Vec::with_capacity(config.m_max + 1);
    let mut radial_offsets = Vec::with_capacity(config.m_max + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.in_channels;
    }

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
            let mut block_out = Vec::new();
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offsets[order],
                copies * m_size,
                input_m_major,
                &mut block_input,
            );
            let block_radial = radial.map(|radial_values| {
                gather_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                )
            });
            if order == 0 {
                let mut modulated = block_input;
                if let Some(block_radial_values) = block_radial.as_deref() {
                    let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 1)?;
                    let mut modulated_with_radial = vec![0.0; modulated.len()];
                    fairchem_esen_so2_radial_modulation_forward(
                        radial_config,
                        &modulated,
                        block_radial_values,
                        &mut modulated_with_radial,
                    )?;
                    modulated = modulated_with_radial;
                }
                let mut dense_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                block_out = dense_out;
            } else {
                block_out.resize(nodes * copies * output_feature_dim, 0.0);
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let weight = m_weights[order - 1];
                if let Some(block_radial_values) = block_radial.as_deref() {
                    fairchem_esen_so2_m_conv_block_with_radial_forward(
                        block_config,
                        &block_input,
                        block_radial_values,
                        weight,
                        &mut block_out,
                    )?;
                } else {
                    fairchem_esen_so2_m_conv_block_forward(
                        block_config,
                        &block_input,
                        weight,
                        &mut block_out,
                    )?;
                }
            }
            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: if order == 0 {
                    config.m0_output_feature_dim()
                } else {
                    copies * output_feature_dim
                },
                source_feature_offset: if order == 0 {
                    config.extra_m0_output_channels
                } else {
                    0
                },
                extra_m0_feature_offset: (order == 0 && config.extra_m0_output_channels > 0)
                    .then_some(0),
                block_out,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_m_major_block_strided(
            nodes,
            config.coefficient_count(),
            config.out_channels,
            result.coeff_offset,
            result.block_coefficients,
            result.source_node_stride,
            result.source_feature_offset,
            &result.block_out,
            out_m_major,
        );
        if let Some(extra_offset) = result.extra_m0_feature_offset {
            let Some(target) = extra_m0_out.as_mut() else {
                return Err(invalid_so2("validated extra m0 output is missing"));
            };
            copy_extra_m0_from_strided_source(
                nodes,
                config.extra_m0_output_channels,
                result.source_node_stride,
                extra_offset,
                &result.block_out,
                target,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_l_major_direct_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    output_scales: Option<&[f64]>,
) -> Result<()> {
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                if let Some(radial_values) = radial {
                    let mut modulated = uninit_f64_vec(nodes * input_feature_dim);
                    gather_l_major_radial_modulated_block_from_m_permutation(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        coeff_offsets[order],
                        m_size,
                        config.radial_node_stride(),
                        radial_offsets[order],
                        l_to_m_permutation,
                        input_l_major,
                        radial_values,
                        &mut modulated,
                    );
                    let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                    dense_forward_with_bias_rows(
                        nodes,
                        input_feature_dim,
                        config.m0_output_feature_dim(),
                        &modulated,
                        m0_weight,
                        m0_bias,
                        &mut dense_out,
                    );
                    return Ok(So2ConvForwardOrderResult {
                        coeff_offset: coeff_offsets[order],
                        block_coefficients: m_size,
                        source_node_stride: config.m0_output_feature_dim(),
                        source_feature_offset: config.extra_m0_output_channels,
                        extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                        block_out: dense_out,
                    });
                }

                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                {
                    let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                    dense_forward_l_major_m0_from_permutation_with_bias_rows(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        input_l_major,
                        m0_weight,
                        m0_bias,
                        config.m0_output_feature_dim(),
                        &mut dense_out,
                    );
                    return Ok(So2ConvForwardOrderResult {
                        coeff_offset: coeff_offsets[order],
                        block_coefficients: m_size,
                        source_node_stride: config.m0_output_feature_dim(),
                        source_feature_offset: config.extra_m0_output_channels,
                        extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                        block_out: dense_out,
                    });
                }
            }

            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            if order > 0 {
                let fused_weight =
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied());
                if let Some(radial_values) = radial {
                    let mut block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
                    fairchem_esen_so2_m_conv_block_with_radial_forward_l_major_direct_to_block(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        m_size,
                        coeff_offsets[order],
                        config.radial_node_stride(),
                        radial_offsets[order],
                        l_to_m_permutation,
                        input_l_major,
                        radial_values,
                        m_weights[order - 1],
                        fused_weight,
                        &mut block_out,
                    )?;
                    return Ok(So2ConvForwardOrderResult {
                        coeff_offset: coeff_offsets[order],
                        block_coefficients: copies * m_size,
                        source_node_stride: copies * output_feature_dim,
                        source_feature_offset: 0,
                        extra_m0_feature_offset: None,
                        block_out,
                    });
                }
            }

            let mut block_input = uninit_f64_vec(nodes * copies * input_feature_dim);
            let mut block_out: Vec<f64>;
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                input_l_major,
                &mut block_input,
            );
            let block_radial = radial.map(|radial_values| {
                gather_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                )
            });
            if order == 0 {
                let mut modulated = block_input;
                if let Some(block_radial_values) = block_radial.as_deref() {
                    let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 1)?;
                    let mut modulated_with_radial = uninit_f64_vec(modulated.len());
                    fairchem_esen_so2_radial_modulation_forward(
                        radial_config,
                        &modulated,
                        block_radial_values,
                        &mut modulated_with_radial,
                    )?;
                    modulated = modulated_with_radial;
                }
                let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                block_out = dense_out;
            } else {
                block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let weight = m_weights[order - 1];
                if let Some(block_radial_values) = block_radial.as_deref() {
                    fairchem_esen_so2_m_conv_block_with_radial_forward(
                        block_config,
                        &block_input,
                        block_radial_values,
                        weight,
                        &mut block_out,
                    )?;
                } else {
                    #[cfg(any(
                        all(feature = "accelerate-blas", target_os = "macos"),
                        all(
                            feature = "openblas",
                            target_os = "linux",
                            any(target_arch = "x86_64", target_arch = "aarch64")
                        )
                    ))]
                    if let Some(fused_weight) =
                        m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
                    {
                        fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                            block_config,
                            &block_input,
                            weight,
                            fused_weight,
                            &mut block_out,
                        )?;
                    } else {
                        fairchem_esen_so2_m_conv_block_forward(
                            block_config,
                            &block_input,
                            weight,
                            &mut block_out,
                        )?;
                    }
                    #[cfg(not(any(
                        all(feature = "accelerate-blas", target_os = "macos"),
                        all(
                            feature = "openblas",
                            target_os = "linux",
                            any(target_arch = "x86_64", target_arch = "aarch64")
                        )
                    )))]
                    fairchem_esen_so2_m_conv_block_forward(
                        block_config,
                        &block_input,
                        weight,
                        &mut block_out,
                    )?;
                }
            }
            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: if order == 0 {
                    config.m0_output_feature_dim()
                } else {
                    copies * output_feature_dim
                },
                source_feature_offset: if order == 0 {
                    config.extra_m0_output_channels
                } else {
                    0
                },
                extra_m0_feature_offset: (order == 0 && config.extra_m0_output_channels > 0)
                    .then_some(0),
                block_out,
            })
        })
        .collect::<Vec<_>>();

    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    scatter_l_major_order_results_from_m_permutation_strided(
        nodes,
        config.coefficient_count(),
        config.out_channels,
        l_to_m_permutation,
        output_scales,
        &results,
        config.extra_m0_output_channels,
        out_l_major,
        extra_m0_out,
    )
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_m_major_input_l_major_out_direct_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    output_scales: &[f64],
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
) -> Result<()> {
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let mut block_input = uninit_f64_vec(nodes * copies * input_feature_dim);
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offsets[order],
                copies * m_size,
                input_m_major,
                &mut block_input,
            );
            let block_radial = radial.map(|radial_values| {
                gather_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                )
            });

            if order == 0 {
                let mut modulated = block_input;
                if let Some(block_radial_values) = block_radial.as_deref() {
                    let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 1)?;
                    let mut modulated_with_radial = uninit_f64_vec(modulated.len());
                    fairchem_esen_so2_radial_modulation_forward(
                        radial_config,
                        &modulated,
                        block_radial_values,
                        &mut modulated_with_radial,
                    )?;
                    modulated = modulated_with_radial;
                }
                let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                return Ok(So2ConvForwardOrderResult {
                    coeff_offset: coeff_offsets[order],
                    block_coefficients: m_size,
                    source_node_stride: config.m0_output_feature_dim(),
                    source_feature_offset: config.extra_m0_output_channels,
                    extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                    block_out: dense_out,
                });
            }

            let mut block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            if let Some(block_radial_values) = block_radial.as_deref() {
                fairchem_esen_so2_m_conv_block_with_radial_forward(
                    block_config,
                    &block_input,
                    block_radial_values,
                    weight,
                    &mut block_out,
                )?;
            } else {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if let Some(fused_weight) =
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied())
                {
                    fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
                        block_config,
                        &block_input,
                        weight,
                        fused_weight,
                        &mut block_out,
                    )?;
                } else {
                    fairchem_esen_so2_m_conv_block_forward(
                        block_config,
                        &block_input,
                        weight,
                        &mut block_out,
                    )?;
                }
                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                fairchem_esen_so2_m_conv_block_forward(
                    block_config,
                    &block_input,
                    weight,
                    &mut block_out,
                )?;
            }

            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: copies * output_feature_dim,
                source_feature_offset: 0,
                extra_m0_feature_offset: None,
                block_out,
            })
        })
        .collect::<Vec<_>>();

    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    scatter_l_major_order_results_from_m_permutation_strided(
        nodes,
        config.coefficient_count(),
        config.out_channels,
        l_to_m_permutation,
        Some(output_scales),
        &results,
        config.extra_m0_output_channels,
        out_l_major,
        extra_m0_out,
    )
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_l_major_direct_parallel_with_radial_block_source<F>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_block_source: &F,
) -> Result<()>
where
    F: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let mut radial_block = uninit_f64_vec(nodes * input_feature_dim);
            radial_block_source(radial_offsets[order], input_feature_dim, &mut radial_block)?;

            if order == 0 {
                let mut modulated = uninit_f64_vec(nodes * input_feature_dim);
                gather_l_major_radial_modulated_block_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    m_size,
                    input_feature_dim,
                    0,
                    l_to_m_permutation,
                    input_l_major,
                    &radial_block,
                    &mut modulated,
                );
                let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                return Ok(So2ConvForwardOrderResult {
                    coeff_offset: coeff_offsets[order],
                    block_coefficients: m_size,
                    source_node_stride: config.m0_output_feature_dim(),
                    source_feature_offset: config.extra_m0_output_channels,
                    extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                    block_out: dense_out,
                });
            }

            let mut block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            let weight = m_weights[order - 1];

            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            {
                let fused_weight =
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied());
                fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    config.out_channels,
                    m_size,
                    coeff_offsets[order],
                    l_to_m_permutation,
                    input_l_major,
                    &radial_block,
                    weight,
                    fused_weight,
                    &mut block_out,
                )?;
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            {
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
                gather_l_major_block_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    copies * m_size,
                    l_to_m_permutation,
                    input_l_major,
                    &mut block_input,
                );
                fairchem_esen_so2_m_conv_block_with_radial_forward(
                    block_config,
                    &block_input,
                    &radial_block,
                    weight,
                    &mut block_out,
                )?;
            }

            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: copies * output_feature_dim,
                source_feature_offset: 0,
                extra_m0_feature_offset: None,
                block_out,
            })
        })
        .collect::<Vec<_>>();

    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    scatter_l_major_order_results_from_m_permutation_strided(
        nodes,
        config.coefficient_count(),
        config.out_channels,
        l_to_m_permutation,
        None,
        &results,
        config.extra_m0_output_channels,
        out_l_major,
        extra_m0_out,
    )
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_m_major_out_direct_parallel_with_radial_block_source<F>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_m_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_block_source: &F,
) -> Result<()>
where
    F: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let mut radial_block = uninit_f64_vec(nodes * input_feature_dim);
            radial_block_source(radial_offsets[order], input_feature_dim, &mut radial_block)?;

            if order == 0 {
                let mut modulated = uninit_f64_vec(nodes * input_feature_dim);
                gather_l_major_radial_modulated_block_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    m_size,
                    input_feature_dim,
                    0,
                    l_to_m_permutation,
                    input_l_major,
                    &radial_block,
                    &mut modulated,
                );
                let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                return Ok(So2ConvForwardOrderResult {
                    coeff_offset: coeff_offsets[order],
                    block_coefficients: m_size,
                    source_node_stride: config.m0_output_feature_dim(),
                    source_feature_offset: config.extra_m0_output_channels,
                    extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                    block_out: dense_out,
                });
            }

            let mut block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            let weight = m_weights[order - 1];

            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            {
                let fused_weight =
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied());
                fairchem_esen_so2_m_conv_block_with_radial_block_forward_l_major_direct_to_block(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    config.out_channels,
                    m_size,
                    coeff_offsets[order],
                    l_to_m_permutation,
                    input_l_major,
                    &radial_block,
                    weight,
                    fused_weight,
                    &mut block_out,
                )?;
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            {
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
                gather_l_major_block_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    copies * m_size,
                    l_to_m_permutation,
                    input_l_major,
                    &mut block_input,
                );
                fairchem_esen_so2_m_conv_block_with_radial_forward(
                    block_config,
                    &block_input,
                    &radial_block,
                    weight,
                    &mut block_out,
                )?;
            }

            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: copies * output_feature_dim,
                source_feature_offset: 0,
                extra_m0_feature_offset: None,
                block_out,
            })
        })
        .collect::<Vec<_>>();

    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    scatter_m_major_order_results_strided(
        nodes,
        config.coefficient_count(),
        config.out_channels,
        &results,
        config.extra_m0_output_channels,
        out_m_major,
        extra_m0_out,
    )
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_forward_l_major_direct_parallel_with_radial_strided_block_source<
    F,
>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_strided_block_source: &F,
) -> Result<()>
where
    F: Fn(usize, usize, usize, usize, &mut [f64]) -> Result<()> + Sync,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                let mut modulated = uninit_f64_vec(nodes * input_feature_dim);
                radial_strided_block_source(
                    radial_offsets[order],
                    input_feature_dim,
                    input_feature_dim,
                    0,
                    &mut modulated,
                )?;
                modulate_l_major_radial_block_in_place_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    m_size,
                    l_to_m_permutation,
                    input_l_major,
                    &mut modulated,
                );
                let mut dense_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                dense_forward_with_bias_rows(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    &modulated,
                    m0_weight,
                    m0_bias,
                    &mut dense_out,
                );
                return Ok(So2ConvForwardOrderResult {
                    coeff_offset: coeff_offsets[order],
                    block_coefficients: m_size,
                    source_node_stride: config.m0_output_feature_dim(),
                    source_feature_offset: config.extra_m0_output_channels,
                    extra_m0_feature_offset: (config.extra_m0_output_channels > 0).then_some(0),
                    block_out: dense_out,
                });
            }

            let mut block_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            let weight = m_weights[order - 1];

            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            {
                let mut compact_input = uninit_f64_vec(nodes * copies * input_feature_dim);
                radial_strided_block_source(
                    radial_offsets[order],
                    input_feature_dim,
                    copies * input_feature_dim,
                    0,
                    &mut compact_input,
                )?;
                modulate_l_major_radial_real_block_to_m_conv_compact_in_place_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    m_size,
                    l_to_m_permutation,
                    input_l_major,
                    &mut compact_input,
                );
                let fused_weight =
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied());
                if let Some(fused_weight) = fused_weight {
                    check_len(
                        "fairchem_esen_so2_convolution radial strided fused weight",
                        4 * input_feature_dim * output_feature_dim,
                        fused_weight.len(),
                    )?;
                    accelerate_blas::so2_mconv_forward_precomputed_fused_weight(
                        &compact_input,
                        nodes,
                        input_feature_dim,
                        fused_weight,
                        output_feature_dim,
                        &mut block_out,
                    );
                } else {
                    accelerate_blas::so2_mconv_forward_fused_weight(
                        &compact_input,
                        nodes,
                        input_feature_dim,
                        weight,
                        output_feature_dim,
                        &mut block_out,
                    );
                }
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            {
                let mut radial_block = uninit_f64_vec(nodes * input_feature_dim);
                radial_strided_block_source(
                    radial_offsets[order],
                    input_feature_dim,
                    input_feature_dim,
                    0,
                    &mut radial_block,
                )?;
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
                gather_l_major_block_from_m_permutation(
                    nodes,
                    config.coefficient_count(),
                    config.in_channels,
                    coeff_offsets[order],
                    copies * m_size,
                    l_to_m_permutation,
                    input_l_major,
                    &mut block_input,
                );
                fairchem_esen_so2_m_conv_block_with_radial_forward(
                    block_config,
                    &block_input,
                    &radial_block,
                    weight,
                    &mut block_out,
                )?;
            }

            Ok(So2ConvForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_coefficients: copies * m_size,
                source_node_stride: copies * output_feature_dim,
                source_feature_offset: 0,
                extra_m0_feature_offset: None,
                block_out,
            })
        })
        .collect::<Vec<_>>();

    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    scatter_l_major_order_results_from_m_permutation_strided(
        nodes,
        config.coefficient_count(),
        config.out_channels,
        l_to_m_permutation,
        None,
        &results,
        config.extra_m0_output_channels,
        out_l_major,
        extra_m0_out,
    )
}

#[cfg(feature = "parallel")]
struct So2ConvVjpOrderResult {
    coeff_offset: usize,
    radial_offset: usize,
    block_coefficients: usize,
    input_feature_dim: usize,
    block_grad_input: Vec<f64>,
    block_grad_radial: Option<Vec<f64>>,
    compact_grad_modulated: Option<Vec<f64>>,
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_m_major_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_m_major: &mut [f64],
    mut grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    let mut coeff_offsets = Vec::with_capacity(config.m_max + 1);
    let mut radial_offsets = Vec::with_capacity(config.m_max + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.m_max {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.m_size_unchecked(order);
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.in_channels;
    }

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;
            let mut block_input = vec![0.0; nodes * copies * input_feature_dim];
            let mut block_grad_out = vec![0.0; nodes * copies * output_feature_dim];
            let mut block_grad_input = vec![0.0; block_input.len()];
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                coeff_offsets[order],
                copies * m_size,
                input_m_major,
                &mut block_input,
            );
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                grad_out_m_major,
                &mut block_grad_out,
            );
            let block_radial = radial.map(|radial_values| {
                gather_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                )
            });
            let mut block_grad_radial = block_radial
                .as_ref()
                .map(|block_radial_values| vec![0.0; block_radial_values.len()]);
            if order == 0 {
                let mut full_grad_out = uninit_f64_vec(nodes * config.m0_output_feature_dim());
                for node in 0..nodes {
                    let full_start = node * config.m0_output_feature_dim();
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        let extra_start = node * config.extra_m0_output_channels;
                        full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                            .copy_from_slice(
                                &extra_values
                                    [extra_start..extra_start + config.extra_m0_output_channels],
                            );
                    }
                    let main_start = full_start + config.extra_m0_output_channels;
                    let block_start = node * output_feature_dim;
                    full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                        &block_grad_out[block_start..block_start + output_feature_dim],
                    );
                }
                let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                dense_vjp_input_no_bias_rows_add(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    m0_weight,
                    &full_grad_out,
                    &mut grad_modulated,
                );
                if let (Some(block_radial_values), Some(block_grad_radial_values)) =
                    (block_radial.as_deref(), block_grad_radial.as_deref_mut())
                {
                    let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 1)?;
                    fairchem_esen_so2_radial_modulation_vjp(
                        radial_config,
                        &block_input,
                        block_radial_values,
                        &grad_modulated,
                        &mut block_grad_input,
                        block_grad_radial_values,
                    )?;
                } else {
                    block_grad_input.copy_from_slice(&grad_modulated);
                }
            } else {
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let weight = m_weights[order - 1];
                if let (Some(block_radial_values), Some(block_grad_radial_values)) =
                    (block_radial.as_deref(), block_grad_radial.as_deref_mut())
                {
                    fairchem_esen_so2_m_conv_block_with_radial_vjp(
                        block_config,
                        &block_input,
                        block_radial_values,
                        weight,
                        &block_grad_out,
                        &mut block_grad_input,
                        block_grad_radial_values,
                    )?;
                } else {
                    fairchem_esen_so2_m_conv_block_vjp(
                        block_config,
                        weight,
                        &block_grad_out,
                        &mut block_grad_input,
                    )?;
                }
            }
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input,
                block_grad_radial,
                compact_grad_modulated: None,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_add_m_major_block(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            result.coeff_offset,
            result.block_coefficients,
            &result.block_grad_input,
            grad_input_m_major,
        );
        if let (Some(block_grad_radial_values), Some(grad_radial_values)) = (
            result.block_grad_radial.as_deref(),
            grad_radial.as_deref_mut(),
        ) {
            scatter_add_radial_block(
                nodes,
                config.radial_node_stride(),
                result.radial_offset,
                result.input_feature_dim,
                block_grad_radial_values,
                grad_radial_values,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_l_major_direct_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    mut grad_radial: Option<&mut [f64]>,
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
) -> Result<()> {
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if radial.is_none() && config.extra_m0_output_channels == 0 {
                    let mut block_grad_input = vec![0.0; nodes * input_feature_dim];
                    dense_vjp_l_major_m0_from_permutation_no_bias_rows_add(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        m0_weight,
                        grad_out_l_major,
                        &mut block_grad_input,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input,
                        block_grad_radial: None,
                        compact_grad_modulated: None,
                    });
                }

                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if radial.is_some() && config.extra_m0_output_channels > 0 {
                    let Some(extra_values) = grad_extra_m0 else {
                        return Err(invalid_so2("validated extra m0 grad is missing"));
                    };
                    let mut grad_modulated = uninit_f64_vec(nodes * input_feature_dim);
                    dense_vjp_l_major_m0_with_extra_from_permutation_no_bias_rows_set(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        config.extra_m0_output_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        m0_weight,
                        grad_out_l_major,
                        extra_values,
                        &mut grad_modulated,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }

                let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                for node in 0..nodes {
                    let full_start = node * config.m0_output_feature_dim();
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        let extra_start = node * config.extra_m0_output_channels;
                        full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                            .copy_from_slice(
                                &extra_values
                                    [extra_start..extra_start + config.extra_m0_output_channels],
                            );
                    }

                    let grad_out_node_start = node * config.output_node_stride();
                    let main_start = full_start + config.extra_m0_output_channels;
                    for block_coefficient in 0..m_size {
                        let l_major_coefficient =
                            l_to_m_permutation[coeff_offsets[order] + block_coefficient];
                        let source_start =
                            grad_out_node_start + l_major_coefficient * config.out_channels;
                        let target_start = main_start + block_coefficient * config.out_channels;
                        full_grad_out[target_start..target_start + config.out_channels]
                            .copy_from_slice(
                                &grad_out_l_major[source_start..source_start + config.out_channels],
                            );
                    }
                }

                let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                dense_vjp_input_no_bias_rows_add(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    m0_weight,
                    &full_grad_out,
                    &mut grad_modulated,
                );

                if radial.is_some() {
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }

                return Ok(So2ConvVjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_coefficients: m_size,
                    input_feature_dim,
                    block_grad_input: grad_modulated,
                    block_grad_radial: None,
                    compact_grad_modulated: None,
                });
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                grad_out_l_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            let compact_grad_modulated = if radial.is_some() {
                let block_config =
                    EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
                let weight = m_weights[order - 1];
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                let mut compact_grad = uninit_f64_vec(block_len);
                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                let mut compact_grad = vec![0.0; block_len];
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                    block_config,
                    weight,
                    m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                    &block_grad_out,
                    &mut compact_grad,
                )?;
                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                fairchem_esen_so2_m_conv_block_vjp(
                    block_config,
                    weight,
                    &block_grad_out,
                    &mut compact_grad,
                )?;
                Some(compact_grad)
            } else {
                None
            };
            if let Some(compact_grad_modulated) = compact_grad_modulated {
                return Ok(So2ConvVjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_coefficients: copies * m_size,
                    input_feature_dim,
                    block_grad_input: Vec::new(),
                    block_grad_radial: None,
                    compact_grad_modulated: Some(compact_grad_modulated),
                });
            }

            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut block_grad_input = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut block_grad_input = vec![0.0; block_len];
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                &block_grad_out,
                &mut block_grad_input,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input,
                block_grad_radial: None,
                compact_grad_modulated,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        if let Some(compact_grad_modulated) = result.compact_grad_modulated.as_deref() {
            let (Some(radial_values), Some(grad_radial_values)) =
                (radial, grad_radial.as_deref_mut())
            else {
                return Err(invalid_so2(
                    "validated radial values and radial grad are missing",
                ));
            };
            scatter_add_l_major_radial_vjp_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                config.radial_node_stride(),
                result.coeff_offset,
                result.block_coefficients,
                result.input_feature_dim,
                result.radial_offset,
                l_to_m_permutation,
                input_l_major,
                radial_values,
                compact_grad_modulated,
                grad_input_l_major,
                grad_radial_values,
            );
        } else {
            scatter_add_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.in_channels,
                result.coeff_offset,
                result.block_coefficients,
                l_to_m_permutation,
                &result.block_grad_input,
                grad_input_l_major,
            );
            if let (Some(block_grad_radial_values), Some(grad_radial_values)) = (
                result.block_grad_radial.as_deref(),
                grad_radial.as_deref_mut(),
            ) {
                scatter_add_radial_block(
                    nodes,
                    config.radial_node_stride(),
                    result.radial_offset,
                    result.input_feature_dim,
                    block_grad_radial_values,
                    grad_radial_values,
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_m_major_input_l_major_grad_out_direct_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_m_major: &mut [f64],
    coeff_offsets: &[usize],
    l_to_m_permutation: &[usize],
) -> Result<()> {
    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if config.extra_m0_output_channels == 0 {
                    let mut block_grad_input = vec![0.0; nodes * input_feature_dim];
                    dense_vjp_l_major_m0_from_permutation_no_bias_rows_add(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        m0_weight,
                        grad_out_l_major,
                        &mut block_grad_input,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: 0,
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input,
                        block_grad_radial: None,
                        compact_grad_modulated: None,
                    });
                }

                let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                for node in 0..nodes {
                    let full_start = node * config.m0_output_feature_dim();
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        let extra_start = node * config.extra_m0_output_channels;
                        full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                            .copy_from_slice(
                                &extra_values
                                    [extra_start..extra_start + config.extra_m0_output_channels],
                            );
                    }

                    let grad_out_node_start = node * config.output_node_stride();
                    let main_start = full_start + config.extra_m0_output_channels;
                    for block_coefficient in 0..m_size {
                        let l_major_coefficient =
                            l_to_m_permutation[coeff_offsets[order] + block_coefficient];
                        let source_start =
                            grad_out_node_start + l_major_coefficient * config.out_channels;
                        let target_start = main_start + block_coefficient * config.out_channels;
                        full_grad_out[target_start..target_start + config.out_channels]
                            .copy_from_slice(
                                &grad_out_l_major[source_start..source_start + config.out_channels],
                            );
                    }
                }
                let mut block_grad_input = vec![0.0; nodes * input_feature_dim];
                dense_vjp_input_no_bias_rows_add(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    m0_weight,
                    &full_grad_out,
                    &mut block_grad_input,
                );
                return Ok(So2ConvVjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: 0,
                    block_coefficients: m_size,
                    input_feature_dim,
                    block_grad_input,
                    block_grad_radial: None,
                    compact_grad_modulated: None,
                });
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                grad_out_l_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut block_grad_input = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut block_grad_input = vec![0.0; block_len];
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                None,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: 0,
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input,
                block_grad_radial: None,
                compact_grad_modulated: None,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_add_m_major_block(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            result.coeff_offset,
            result.block_coefficients,
            &result.block_grad_input,
            grad_input_m_major,
        );
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_m_major_grad_out_l_major_input_direct_parallel(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    coeff_offsets: &[usize],
    l_to_m_permutation: &[usize],
) -> Result<()> {
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                {
                    let mut block_grad_input = vec![0.0; nodes * input_feature_dim];
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        accelerate_blas::dense_vjp_input_set(
                            extra_values,
                            nodes,
                            config.extra_m0_output_channels,
                            &m0_weight[..config.extra_m0_output_channels * input_feature_dim],
                            input_feature_dim,
                            &mut block_grad_input,
                        );
                    }
                    let grad_out_feature_offset = coeff_offsets[order] * config.out_channels;
                    let main_weight_offset = config.extra_m0_output_channels * input_feature_dim;
                    accelerate_blas::dense_vjp_input_strided_grad_out_add(
                        grad_out_m_major,
                        nodes,
                        config.output_node_stride(),
                        grad_out_feature_offset,
                        output_feature_dim,
                        &m0_weight[main_weight_offset..],
                        input_feature_dim,
                        input_feature_dim,
                        &mut block_grad_input,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: 0,
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input,
                        block_grad_radial: None,
                        compact_grad_modulated: None,
                    });
                }

                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                {
                    let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                    for node in 0..nodes {
                        let full_start = node * config.m0_output_feature_dim();
                        if config.extra_m0_output_channels > 0 {
                            let Some(extra_values) = grad_extra_m0 else {
                                return Err(invalid_so2("validated extra m0 grad is missing"));
                            };
                            let extra_start = node * config.extra_m0_output_channels;
                            full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                                .copy_from_slice(
                                    &extra_values[extra_start
                                        ..extra_start + config.extra_m0_output_channels],
                                );
                        }

                        let grad_out_node_start = node * config.output_node_stride();
                        let source_start =
                            grad_out_node_start + coeff_offsets[order] * config.out_channels;
                        let main_start = full_start + config.extra_m0_output_channels;
                        full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                            &grad_out_m_major[source_start..source_start + output_feature_dim],
                        );
                    }
                    let mut block_grad_input = vec![0.0; nodes * input_feature_dim];
                    dense_vjp_input_no_bias_rows_add(
                        nodes,
                        input_feature_dim,
                        config.m0_output_feature_dim(),
                        m0_weight,
                        &full_grad_out,
                        &mut block_grad_input,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: 0,
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input,
                        block_grad_radial: None,
                        compact_grad_modulated: None,
                    });
                }
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                grad_out_m_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut block_grad_input = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut block_grad_input = vec![0.0; block_len];
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                &block_grad_out,
                &mut block_grad_input,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut block_grad_input,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: 0,
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input,
                block_grad_radial: None,
                compact_grad_modulated: None,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_add_l_major_block_from_m_permutation(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            result.coeff_offset,
            result.block_coefficients,
            l_to_m_permutation,
            &result.block_grad_input,
            grad_input_l_major,
        );
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_l_major_direct_parallel_with_radial_block_sink<F>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    radial: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_block_sink: &mut F,
) -> Result<()>
where
    F: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if config.extra_m0_output_channels > 0 {
                    let Some(extra_values) = grad_extra_m0 else {
                        return Err(invalid_so2("validated extra m0 grad is missing"));
                    };
                    let mut grad_modulated = uninit_f64_vec(nodes * input_feature_dim);
                    dense_vjp_l_major_m0_with_extra_from_permutation_no_bias_rows_set(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        config.extra_m0_output_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        m0_weight,
                        grad_out_l_major,
                        extra_values,
                        &mut grad_modulated,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }

                let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                for node in 0..nodes {
                    let full_start = node * config.m0_output_feature_dim();
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        let extra_start = node * config.extra_m0_output_channels;
                        full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                            .copy_from_slice(
                                &extra_values
                                    [extra_start..extra_start + config.extra_m0_output_channels],
                            );
                    }

                    let grad_out_node_start = node * config.output_node_stride();
                    let main_start = full_start + config.extra_m0_output_channels;
                    for block_coefficient in 0..m_size {
                        let l_major_coefficient =
                            l_to_m_permutation[coeff_offsets[order] + block_coefficient];
                        let source_start =
                            grad_out_node_start + l_major_coefficient * config.out_channels;
                        let target_start = main_start + block_coefficient * config.out_channels;
                        full_grad_out[target_start..target_start + config.out_channels]
                            .copy_from_slice(
                                &grad_out_l_major[source_start..source_start + config.out_channels],
                            );
                    }
                }

                let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                dense_vjp_input_no_bias_rows_add(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    m0_weight,
                    &full_grad_out,
                    &mut grad_modulated,
                );
                return Ok(So2ConvVjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_coefficients: m_size,
                    input_feature_dim,
                    block_grad_input: Vec::new(),
                    block_grad_radial: None,
                    compact_grad_modulated: Some(grad_modulated),
                });
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                grad_out_l_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut compact_grad = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut compact_grad = vec![0.0; block_len];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                &block_grad_out,
                &mut compact_grad,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut compact_grad,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input: Vec::new(),
                block_grad_radial: None,
                compact_grad_modulated: Some(compact_grad),
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        let Some(compact_grad_modulated) = result.compact_grad_modulated.as_deref() else {
            return Err(invalid_so2(
                "SO2 radial block sink path expected compact radial cotangent",
            ));
        };
        let mut block_grad_radial = vec![0.0; nodes * result.input_feature_dim];
        scatter_add_l_major_radial_vjp_block_from_m_permutation(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            config.radial_node_stride(),
            result.coeff_offset,
            result.block_coefficients,
            result.input_feature_dim,
            result.radial_offset,
            l_to_m_permutation,
            input_l_major,
            radial,
            compact_grad_modulated,
            grad_input_l_major,
            &mut block_grad_radial,
        );
        radial_block_sink(
            result.radial_offset,
            result.input_feature_dim,
            &block_grad_radial,
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_l_major_direct_parallel_with_radial_block_source_and_sink<
    S,
    T,
>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_block_source: &S,
    radial_block_sink: &mut T,
) -> Result<()>
where
    S: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
    T: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                if config.extra_m0_output_channels > 0 {
                    let Some(extra_values) = grad_extra_m0 else {
                        return Err(invalid_so2("validated extra m0 grad is missing"));
                    };
                    let mut grad_modulated = uninit_f64_vec(nodes * input_feature_dim);
                    dense_vjp_l_major_m0_with_extra_from_permutation_no_bias_rows_set(
                        nodes,
                        config.coefficient_count(),
                        config.in_channels,
                        config.out_channels,
                        config.extra_m0_output_channels,
                        coeff_offsets[order],
                        m_size,
                        l_to_m_permutation,
                        m0_weight,
                        grad_out_l_major,
                        extra_values,
                        &mut grad_modulated,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }

                let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                for node in 0..nodes {
                    let full_start = node * config.m0_output_feature_dim();
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        let extra_start = node * config.extra_m0_output_channels;
                        full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                            .copy_from_slice(
                                &extra_values
                                    [extra_start..extra_start + config.extra_m0_output_channels],
                            );
                    }

                    let grad_out_node_start = node * config.output_node_stride();
                    let main_start = full_start + config.extra_m0_output_channels;
                    for block_coefficient in 0..m_size {
                        let l_major_coefficient =
                            l_to_m_permutation[coeff_offsets[order] + block_coefficient];
                        let source_start =
                            grad_out_node_start + l_major_coefficient * config.out_channels;
                        let target_start = main_start + block_coefficient * config.out_channels;
                        full_grad_out[target_start..target_start + config.out_channels]
                            .copy_from_slice(
                                &grad_out_l_major[source_start..source_start + config.out_channels],
                            );
                    }
                }

                let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                dense_vjp_input_no_bias_rows_add(
                    nodes,
                    input_feature_dim,
                    config.m0_output_feature_dim(),
                    m0_weight,
                    &full_grad_out,
                    &mut grad_modulated,
                );
                return Ok(So2ConvVjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_coefficients: m_size,
                    input_feature_dim,
                    block_grad_input: Vec::new(),
                    block_grad_radial: None,
                    compact_grad_modulated: Some(grad_modulated),
                });
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_l_major_block_from_m_permutation(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                l_to_m_permutation,
                grad_out_l_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut compact_grad = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut compact_grad = vec![0.0; block_len];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                &block_grad_out,
                &mut compact_grad,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut compact_grad,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input: Vec::new(),
                block_grad_radial: None,
                compact_grad_modulated: Some(compact_grad),
            })
        })
        .collect::<Vec<_>>();

    let mut radial_block = Vec::new();
    for result in results {
        let result = result?;
        let Some(compact_grad_modulated) = result.compact_grad_modulated.as_deref() else {
            return Err(invalid_so2(
                "SO2 radial block source/sink path expected compact radial cotangent",
            ));
        };
        reset_zeroed_f64_vec(&mut radial_block, nodes * result.input_feature_dim);
        radial_block_source(
            result.radial_offset,
            result.input_feature_dim,
            radial_block.as_mut_slice(),
        )?;
        scatter_add_l_major_radial_vjp_block_from_m_permutation_in_place(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            result.coeff_offset,
            result.block_coefficients,
            result.input_feature_dim,
            l_to_m_permutation,
            input_l_major,
            radial_block.as_mut_slice(),
            compact_grad_modulated,
            grad_input_l_major,
        );
        radial_block_sink(
            result.radial_offset,
            result.input_feature_dim,
            radial_block.as_slice(),
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_so2_convolution_assembly_vjp_m_major_grad_out_l_major_input_direct_parallel_with_radial_block_source_and_sink<
    S,
    T,
>(
    config: EsenSo2ConvolutionConfig,
    nodes: usize,
    input_l_major: &[f64],
    m0_weight: &[f64],
    m_weights: &[&[f64]],
    m_fused_weights: Option<&[&[f64]]>,
    grad_out_m_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    coeff_offsets: &[usize],
    radial_offsets: &[usize],
    l_to_m_permutation: &[usize],
    radial_block_source: &S,
    radial_block_sink: &mut T,
) -> Result<()>
where
    S: Fn(usize, usize, &mut [f64]) -> Result<()> + Sync,
    T: FnMut(usize, usize, &[f64]) -> Result<()>,
{
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = m_fused_weights;

    let results = (0..=config.m_max)
        .into_par_iter()
        .map(|order| -> Result<So2ConvVjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.m_size_unchecked(order);
            let input_feature_dim = m_size * config.in_channels;
            let output_feature_dim = m_size * config.out_channels;

            if order == 0 {
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                {
                    let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                    if config.extra_m0_output_channels > 0 {
                        let Some(extra_values) = grad_extra_m0 else {
                            return Err(invalid_so2("validated extra m0 grad is missing"));
                        };
                        accelerate_blas::dense_vjp_input_set(
                            extra_values,
                            nodes,
                            config.extra_m0_output_channels,
                            &m0_weight[..config.extra_m0_output_channels * input_feature_dim],
                            input_feature_dim,
                            &mut grad_modulated,
                        );
                    }
                    let grad_out_feature_offset = coeff_offsets[order] * config.out_channels;
                    let main_weight_offset = config.extra_m0_output_channels * input_feature_dim;
                    accelerate_blas::dense_vjp_input_strided_grad_out_add(
                        grad_out_m_major,
                        nodes,
                        config.output_node_stride(),
                        grad_out_feature_offset,
                        output_feature_dim,
                        &m0_weight[main_weight_offset..],
                        input_feature_dim,
                        input_feature_dim,
                        &mut grad_modulated,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }

                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        feature = "openblas",
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                {
                    let mut full_grad_out = vec![0.0; nodes * config.m0_output_feature_dim()];
                    for node in 0..nodes {
                        let full_start = node * config.m0_output_feature_dim();
                        if config.extra_m0_output_channels > 0 {
                            let Some(extra_values) = grad_extra_m0 else {
                                return Err(invalid_so2("validated extra m0 grad is missing"));
                            };
                            let extra_start = node * config.extra_m0_output_channels;
                            full_grad_out[full_start..full_start + config.extra_m0_output_channels]
                                .copy_from_slice(
                                    &extra_values[extra_start
                                        ..extra_start + config.extra_m0_output_channels],
                                );
                        }

                        let grad_out_node_start = node * config.output_node_stride();
                        let source_start =
                            grad_out_node_start + coeff_offsets[order] * config.out_channels;
                        let main_start = full_start + config.extra_m0_output_channels;
                        full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                            &grad_out_m_major[source_start..source_start + output_feature_dim],
                        );
                    }
                    let mut grad_modulated = vec![0.0; nodes * input_feature_dim];
                    dense_vjp_input_no_bias_rows_add(
                        nodes,
                        input_feature_dim,
                        config.m0_output_feature_dim(),
                        m0_weight,
                        &full_grad_out,
                        &mut grad_modulated,
                    );
                    return Ok(So2ConvVjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_coefficients: m_size,
                        input_feature_dim,
                        block_grad_input: Vec::new(),
                        block_grad_radial: None,
                        compact_grad_modulated: Some(grad_modulated),
                    });
                }
            }

            let mut block_grad_out = uninit_f64_vec(nodes * copies * output_feature_dim);
            gather_m_major_block(
                nodes,
                config.coefficient_count(),
                config.out_channels,
                coeff_offsets[order],
                copies * m_size,
                grad_out_m_major,
                &mut block_grad_out,
            );
            let block_len = nodes * copies * input_feature_dim;
            let block_config =
                EsenSo2MConvBlockConfig::new(m_size, config.in_channels, config.out_channels)?;
            let weight = m_weights[order - 1];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            let mut compact_grad = uninit_f64_vec(block_len);
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            let mut compact_grad = vec![0.0; block_len];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
                block_config,
                weight,
                m_fused_weights.and_then(|weights| weights.get(order - 1).copied()),
                &block_grad_out,
                &mut compact_grad,
            )?;
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    feature = "openblas",
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            fairchem_esen_so2_m_conv_block_vjp(
                block_config,
                weight,
                &block_grad_out,
                &mut compact_grad,
            )?;
            Ok(So2ConvVjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_coefficients: copies * m_size,
                input_feature_dim,
                block_grad_input: Vec::new(),
                block_grad_radial: None,
                compact_grad_modulated: Some(compact_grad),
            })
        })
        .collect::<Vec<_>>();

    let mut radial_block = Vec::new();
    for result in results {
        let result = result?;
        let Some(compact_grad_modulated) = result.compact_grad_modulated.as_deref() else {
            return Err(invalid_so2(
                "SO2 m-major grad-out radial block source/sink path expected compact radial cotangent",
            ));
        };
        reset_zeroed_f64_vec(&mut radial_block, nodes * result.input_feature_dim);
        radial_block_source(
            result.radial_offset,
            result.input_feature_dim,
            radial_block.as_mut_slice(),
        )?;
        scatter_add_l_major_radial_vjp_block_from_m_permutation_in_place(
            nodes,
            config.coefficient_count(),
            config.in_channels,
            result.coeff_offset,
            result.block_coefficients,
            result.input_feature_dim,
            l_to_m_permutation,
            input_l_major,
            radial_block.as_mut_slice(),
            compact_grad_modulated,
            grad_input_l_major,
        );
        radial_block_sink(
            result.radial_offset,
            result.input_feature_dim,
            radial_block.as_slice(),
        )?;
    }
    Ok(())
}

/// FairChem eSEN `SO2_m_Conv` block after m-major coefficient remapping.
///
/// `input` is flat `[nodes, 2, coefficient_count * in_channels]`, where the
/// two copies are the real and imaginary order-`m` components. `weight` is flat
/// `[2 * coefficient_count * out_channels, coefficient_count * in_channels]`.
/// The output is flat `[nodes, 2, coefficient_count * out_channels]`.
pub fn fairchem_esen_so2_m_conv_block_forward(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_m_conv_block_shapes(
        "fairchem_esen_so2_m_conv_block_forward",
        config,
        input,
        weight,
        out,
    )?;
    let input_len = config.input_copy_stride();
    let output_len = config.output_copy_stride();

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        fairchem_esen_so2_m_conv_block_forward_blas(
            nodes, config, input_len, output_len, input, weight, out,
        )
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        for node_index in 0..nodes {
            let input_node_start = node_index * config.input_node_stride();
            let out_node_start = node_index * config.output_node_stride();
            let real_input = &input[input_node_start..input_node_start + input_len];
            let imag_input = &input[input_node_start + input_len..input_node_start + 2 * input_len];
            for out_index in 0..output_len {
                let direct_weight_start = out_index * input_len;
                let mixed_weight_start = (output_len + out_index) * input_len;
                let mut real_value = 0.0;
                let mut imag_value = 0.0;
                for in_index in 0..input_len {
                    let direct_weight = weight[direct_weight_start + in_index];
                    let mixed_weight = weight[mixed_weight_start + in_index];
                    real_value +=
                        direct_weight * real_input[in_index] - mixed_weight * imag_input[in_index];
                    imag_value +=
                        direct_weight * imag_input[in_index] + mixed_weight * real_input[in_index];
                }
                out[out_node_start + out_index] = real_value;
                out[out_node_start + output_len + out_index] = imag_value;
            }
        }
        Ok(())
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
fn fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    weight: &[f64],
    fused_weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_m_conv_block_shapes(
        "fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight",
        config,
        input,
        weight,
        out,
    )?;
    let input_len = config.input_copy_stride();
    let output_len = config.output_copy_stride();
    check_len(
        "fairchem_esen_so2_m_conv_block_forward_with_precomputed_fused_weight fused weight",
        4 * input_len * output_len,
        fused_weight.len(),
    )?;
    accelerate_blas::so2_mconv_forward_precomputed_fused_weight(
        input,
        nodes,
        input_len,
        fused_weight,
        output_len,
        out,
    );
    Ok(())
}

/// Accumulate the input VJP of [`fairchem_esen_so2_m_conv_block_forward`] into
/// `grad_input`.
pub fn fairchem_esen_so2_m_conv_block_vjp(
    config: EsenSo2MConvBlockConfig,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_node_stride = config.input_node_stride();
    let output_node_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_m_conv_block_vjp grad_input length {} is not a multiple of node stride {input_node_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_node_stride;
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp grad_out",
        nodes * output_node_stride,
        grad_out.len(),
    )?;

    let input_len = config.input_copy_stride();
    let output_len = config.output_copy_stride();

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        fairchem_esen_so2_m_conv_block_vjp_blas(
            nodes, config, input_len, output_len, weight, grad_out, grad_input,
        )
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        for node_index in 0..nodes {
            let grad_out_node_start = node_index * output_node_stride;
            let grad_input_node_start = node_index * input_node_stride;
            let (grad_real_input, grad_imag_input) = grad_input
                [grad_input_node_start..grad_input_node_start + 2 * input_len]
                .split_at_mut(input_len);
            for out_index in 0..output_len {
                let grad_real = grad_out[grad_out_node_start + out_index];
                let grad_imag = grad_out[grad_out_node_start + output_len + out_index];
                let direct_weight_start = out_index * input_len;
                let mixed_weight_start = (output_len + out_index) * input_len;
                for in_index in 0..input_len {
                    let direct_weight = weight[direct_weight_start + in_index];
                    let mixed_weight = weight[mixed_weight_start + in_index];
                    grad_real_input[in_index] +=
                        grad_real * direct_weight + grad_imag * mixed_weight;
                    grad_imag_input[in_index] +=
                        grad_imag * direct_weight - grad_real * mixed_weight;
                }
            }
        }
        Ok(())
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
fn fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight(
    config: EsenSo2MConvBlockConfig,
    weight: &[f64],
    fused_weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_node_stride = config.input_node_stride();
    let output_node_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight grad_input length {} is not a multiple of node stride {input_node_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_node_stride;
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight grad_out",
        nodes * output_node_stride,
        grad_out.len(),
    )?;
    let input_len = config.input_copy_stride();
    let output_len = config.output_copy_stride();
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight fused weight",
        4 * input_len * output_len,
        fused_weight.len(),
    )?;
    accelerate_blas::so2_mconv_vjp_input_add_precomputed_fused(
        grad_out,
        nodes,
        output_len,
        fused_weight,
        input_len,
        grad_input,
        input_node_stride,
    );
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
fn fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight_write(
    config: EsenSo2MConvBlockConfig,
    weight: &[f64],
    fused_weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    grad_input.fill(0.0);
    fairchem_esen_so2_m_conv_block_vjp_with_precomputed_fused_weight(
        config,
        weight,
        fused_weight,
        grad_out,
        grad_input,
    )
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
fn fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight(
    config: EsenSo2MConvBlockConfig,
    weight: &[f64],
    fused_weight: Option<&[f64]>,
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_node_stride = config.input_node_stride();
    let output_node_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight grad_input length {} is not a multiple of node stride {input_node_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_node_stride;
    check_len(
        "fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight grad_out",
        nodes * output_node_stride,
        grad_out.len(),
    )?;
    let input_len = config.input_copy_stride();
    let output_len = config.output_copy_stride();
    if let Some(fused_weight) = fused_weight {
        check_len(
            "fairchem_esen_so2_m_conv_block_vjp_set_zeroed_with_optional_fused_weight fused weight",
            4 * input_len * output_len,
            fused_weight.len(),
        )?;
        accelerate_blas::so2_mconv_vjp_input_set_precomputed_fused(
            grad_out,
            nodes,
            output_len,
            fused_weight,
            input_len,
            grad_input,
            input_node_stride,
        );
    } else {
        accelerate_blas::so2_mconv_vjp_input_set(
            grad_out,
            nodes,
            output_len,
            weight,
            input_len,
            grad_input,
            input_node_stride,
        );
    }
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_so2_m_conv_block_forward_blas(
    nodes: usize,
    config: EsenSo2MConvBlockConfig,
    input_len: usize,
    output_len: usize,
    input: &[f64],
    weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    if nodes >= so2_parallel_node_threshold() {
        accelerate_blas::so2_mconv_forward_fused_weight(
            input, nodes, input_len, weight, output_len, out,
        );
        return Ok(());
    }

    let fc_len = 2 * output_len;
    let mut fc = vec![0.0; 2 * nodes * fc_len];
    accelerate_blas::dense_forward_transposed_weight(
        input,
        2 * nodes,
        input_len,
        weight,
        fc_len,
        &mut fc,
    );

    for node_index in 0..nodes {
        let real_fc_start = (2 * node_index) * fc_len;
        let imag_fc_start = (2 * node_index + 1) * fc_len;
        let out_node_start = node_index * config.output_node_stride();
        for out_index in 0..output_len {
            out[out_node_start + out_index] =
                fc[real_fc_start + out_index] - fc[imag_fc_start + output_len + out_index];
            out[out_node_start + output_len + out_index] =
                fc[imag_fc_start + out_index] + fc[real_fc_start + output_len + out_index];
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_so2_m_conv_block_with_radial_forward_blas(
    nodes: usize,
    config: EsenSo2MConvBlockConfig,
    input_len: usize,
    output_len: usize,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let mut compact_input = uninit_f64_vec(nodes * 2 * input_len);
    for node_index in 0..nodes {
        let input_node_start = node_index * config.input_node_stride();
        let radial_start = node_index * input_len;
        let real_start = node_index * 2 * input_len;
        let imag_start = real_start + input_len;
        for in_index in 0..input_len {
            let radial_value = radial[radial_start + in_index];
            compact_input[real_start + in_index] =
                input[input_node_start + in_index] * radial_value;
            compact_input[imag_start + in_index] =
                input[input_node_start + input_len + in_index] * radial_value;
        }
    }

    accelerate_blas::so2_mconv_forward_fused_weight(
        &compact_input,
        nodes,
        input_len,
        weight,
        output_len,
        out,
    );
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_so2_m_conv_block_vjp_blas(
    nodes: usize,
    config: EsenSo2MConvBlockConfig,
    input_len: usize,
    output_len: usize,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    accelerate_blas::so2_mconv_vjp_input_add(
        grad_out,
        nodes,
        output_len,
        weight,
        input_len,
        grad_input,
        config.input_node_stride(),
    );
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_so2_m_conv_block_with_radial_vjp_blas(
    nodes: usize,
    config: EsenSo2MConvBlockConfig,
    input_len: usize,
    output_len: usize,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
    grad_radial: &mut [f64],
) -> Result<()> {
    let mut compact_grad_modulated = vec![0.0; 2 * nodes * input_len];
    accelerate_blas::so2_mconv_vjp_input_add(
        grad_out,
        nodes,
        output_len,
        weight,
        input_len,
        &mut compact_grad_modulated,
        config.input_node_stride(),
    );

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        let input_node_stride = config.input_node_stride();
        grad_input
            .par_chunks_mut(input_node_stride)
            .zip(grad_radial.par_chunks_mut(input_len))
            .zip(input.par_chunks(input_node_stride))
            .zip(radial.par_chunks(input_len))
            .zip(compact_grad_modulated.par_chunks(input_node_stride))
            .for_each(
                |(
                    (((grad_input_node, grad_radial_node), input_node), radial_node),
                    compact_node,
                )| {
                    let compact_real = &compact_node[..input_len];
                    let compact_imag = &compact_node[input_len..input_node_stride];
                    for in_index in 0..input_len {
                        let radial_value = radial_node[in_index];
                        let grad_real = compact_real[in_index];
                        let grad_imag = compact_imag[in_index];
                        grad_input_node[in_index] += radial_value * grad_real;
                        grad_input_node[input_len + in_index] += radial_value * grad_imag;
                        grad_radial_node[in_index] += input_node[in_index] * grad_real
                            + input_node[input_len + in_index] * grad_imag;
                    }
                },
            );
        return Ok(());
    }

    for node_index in 0..nodes {
        let input_node_start = node_index * config.input_node_stride();
        let radial_start = node_index * input_len;
        let real_start = (2 * node_index) * input_len;
        let imag_start = (2 * node_index + 1) * input_len;
        for in_index in 0..input_len {
            let radial_value = radial[radial_start + in_index];
            let grad_real = compact_grad_modulated[real_start + in_index];
            let grad_imag = compact_grad_modulated[imag_start + in_index];
            grad_input[input_node_start + in_index] += radial_value * grad_real;
            grad_input[input_node_start + input_len + in_index] += radial_value * grad_imag;
            grad_radial[radial_start + in_index] += input[input_node_start + in_index] * grad_real
                + input[input_node_start + input_len + in_index] * grad_imag;
        }
    }
    Ok(())
}

/// FairChem eSEN SO2 linear block for one order `m`, after coefficient
/// remapping into m-major layout.
///
/// For `m = 0`, use `copies = 1` and pass the dense bias. For `m > 0`, use
/// `copies = 2` for the real and imaginary blocks and pass no bias; the same
/// dense matrix is applied independently to both blocks.
pub fn fairchem_esen_so2_linear_block_forward(
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so2_linear_block_shapes(
        "fairchem_esen_so2_linear_block_forward",
        config,
        input,
        weight,
        bias,
        out,
    )?;
    let input_copy_stride = config.input_copy_stride();
    let output_copy_stride = config.output_copy_stride();

    for node_index in 0..nodes {
        let input_node_start = node_index * config.input_node_stride();
        let out_node_start = node_index * config.output_node_stride();
        for copy_index in 0..config.copies {
            let input_copy_start = input_node_start + copy_index * input_copy_stride;
            let out_copy_start = out_node_start + copy_index * output_copy_stride;
            for out_index in 0..output_copy_stride {
                let mut value = bias.map_or(0.0, |bias_values| bias_values[out_index]);
                let weight_row_start = out_index * input_copy_stride;
                for in_index in 0..input_copy_stride {
                    value +=
                        weight[weight_row_start + in_index] * input[input_copy_start + in_index];
                }
                out[out_copy_start + out_index] = value;
            }
        }
    }
    Ok(())
}

/// Accumulate the input VJP of [`fairchem_esen_so2_linear_block_forward`] into
/// `grad_input`.
pub fn fairchem_esen_so2_linear_block_vjp(
    config: EsenSo2LinearBlockConfig,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_so2_linear_block_vjp weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_node_stride = config.input_node_stride();
    let output_node_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "fairchem_esen_so2_linear_block_vjp grad_input length {} is not a multiple of node stride {input_node_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_node_stride;
    check_len(
        "fairchem_esen_so2_linear_block_vjp grad_out",
        nodes * output_node_stride,
        grad_out.len(),
    )?;

    let input_copy_stride = config.input_copy_stride();
    let output_copy_stride = config.output_copy_stride();

    for node_index in 0..nodes {
        let grad_input_node_start = node_index * input_node_stride;
        let grad_out_node_start = node_index * output_node_stride;
        for copy_index in 0..config.copies {
            let grad_input_copy_start = grad_input_node_start + copy_index * input_copy_stride;
            let grad_out_copy_start = grad_out_node_start + copy_index * output_copy_stride;
            for out_index in 0..output_copy_stride {
                let grad = grad_out[grad_out_copy_start + out_index];
                let weight_row_start = out_index * input_copy_stride;
                for in_index in 0..input_copy_stride {
                    grad_input[grad_input_copy_start + in_index] +=
                        weight[weight_row_start + in_index] * grad;
                }
            }
        }
    }
    Ok(())
}

#[cfg(not(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
fn dense_forward_with_bias(
    input_len: usize,
    output_len: usize,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    out: &mut [f64],
) {
    for (out_index, out_value) in out.iter_mut().enumerate().take(output_len) {
        let mut value = bias[out_index];
        let weight_row_start = out_index * input_len;
        for in_index in 0..input_len {
            value += weight[weight_row_start + in_index] * input[in_index];
        }
        *out_value = value;
    }
}

fn dense_forward_with_bias_rows(
    rows: usize,
    input_len: usize,
    output_len: usize,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    out: &mut [f64],
) {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        accelerate_blas::dense_forward_transposed_weight(
            input, rows, input_len, weight, output_len, out,
        );
        add_bias_rows(rows, output_len, bias, out);
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        for row in 0..rows {
            let input_start = row * input_len;
            let out_start = row * output_len;
            dense_forward_with_bias(
                input_len,
                output_len,
                &input[input_start..input_start + input_len],
                weight,
                bias,
                &mut out[out_start..out_start + output_len],
            );
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_forward_l_major_m0_from_permutation_with_bias_rows(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    weight: &[f64],
    bias: &[f64],
    output_len: usize,
    out: &mut [f64],
) {
    let input_node_stride = coefficient_count * channels;
    let input_len = block_coefficients * channels;
    for block_coefficient in 0..block_coefficients {
        let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
        let input_feature_offset = l_major_coefficient * channels;
        let weight_feature_offset = block_coefficient * channels;
        let beta = if block_coefficient == 0 { 0.0 } else { 1.0 };
        accelerate_blas::dense_forward_strided_input_transposed_weight_add(
            input_l_major,
            rows,
            input_node_stride,
            input_feature_offset,
            channels,
            &weight[weight_feature_offset..],
            input_len,
            output_len,
            beta,
            out,
        );
    }
    add_bias_rows(rows, output_len, bias, out);
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn add_bias_rows(rows: usize, row_len: usize, bias: &[f64], out: &mut [f64]) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= crate::esen_parallel_min_nodes() {
        out.par_chunks_mut(row_len).for_each(|row| {
            for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                *value += *bias_value;
            }
        });
        return;
    }

    for row in out.chunks_exact_mut(row_len).take(rows) {
        for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
            *value += *bias_value;
        }
    }
}

#[cfg(not(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
fn dense_vjp_input_no_bias(
    input_len: usize,
    output_len: usize,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    for (out_index, grad) in grad_out.iter().copied().enumerate().take(output_len) {
        let weight_row_start = out_index * input_len;
        for in_index in 0..input_len {
            grad_input[in_index] += weight[weight_row_start + in_index] * grad;
        }
    }
}

fn dense_vjp_input_no_bias_rows_add(
    rows: usize,
    input_len: usize,
    output_len: usize,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        accelerate_blas::dense_vjp_input_add(
            grad_out, rows, output_len, weight, input_len, grad_input,
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        for row in 0..rows {
            let input_start = row * input_len;
            let out_start = row * output_len;
            dense_vjp_input_no_bias(
                input_len,
                output_len,
                weight,
                &grad_out[out_start..out_start + output_len],
                &mut grad_input[input_start..input_start + input_len],
            );
        }
    }
}

fn dense_vjp_input_no_bias_rows_write(
    rows: usize,
    input_len: usize,
    output_len: usize,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    grad_input.fill(0.0);
    dense_vjp_input_no_bias_rows_add(rows, input_len, output_len, weight, grad_out, grad_input);
}

fn gather_m_major_block(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let target_node_stride = block_coefficients * channels;
    for node in 0..nodes {
        let source_start = node * source_node_stride + coefficient_offset * channels;
        let target_start = node * target_node_stride;
        target[target_start..target_start + target_node_stride]
            .copy_from_slice(&source[source_start..source_start + target_node_stride]);
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_m_major_radial_modulated_block(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    radial_node_stride: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    coefficients_per_copy: usize,
    radial_offset: usize,
    source: &[f64],
    radial: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let target_node_stride = block_coefficients * channels;
    for node in 0..nodes {
        let source_node_start = node * source_node_stride + coefficient_offset * channels;
        let radial_node_start = node * radial_node_stride + radial_offset;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let radial_coefficient = block_coefficient % coefficients_per_copy;
            let source_start = source_node_start + block_coefficient * channels;
            let radial_start = radial_node_start + radial_coefficient * channels;
            let target_start = target_node_start + block_coefficient * channels;
            for channel in 0..channels {
                target[target_start + channel] =
                    source[source_start + channel] * radial[radial_start + channel];
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn gather_l_major_block_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let target_node_stride = block_coefficients * channels;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(target_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(target_node, source_node)| {
                for block_coefficient in 0..block_coefficients {
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let source_start = l_major_coefficient * channels;
                    let target_start = block_coefficient * channels;
                    target_node[target_start..target_start + channels]
                        .copy_from_slice(&source_node[source_start..source_start + channels]);
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let source_start = source_node_start + l_major_coefficient * channels;
            let target_start = target_node_start + block_coefficient * channels;
            target[target_start..target_start + channels]
                .copy_from_slice(&source[source_start..source_start + channels]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "parallel")]
fn gather_l_major_radial_modulated_block_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    radial_node_stride: usize,
    radial_offset: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    radial: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let target_node_stride = block_coefficients * channels;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(target_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .zip(radial.par_chunks(radial_node_stride))
            .for_each(|((target_node, source_node), radial_node)| {
                for block_coefficient in 0..block_coefficients {
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let source_start = l_major_coefficient * channels;
                    let radial_start = radial_offset + block_coefficient * channels;
                    let target_start = block_coefficient * channels;
                    for channel in 0..channels {
                        target_node[target_start + channel] = source_node[source_start + channel]
                            * radial_node[radial_start + channel];
                    }
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let radial_node_start = node * radial_node_stride + radial_offset;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let source_start = source_node_start + l_major_coefficient * channels;
            let radial_start = radial_node_start + block_coefficient * channels;
            let target_start = target_node_start + block_coefficient * channels;
            for channel in 0..channels {
                target[target_start + channel] =
                    source[source_start + channel] * radial[radial_start + channel];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "parallel")]
fn modulate_l_major_radial_block_in_place_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let target_node_stride = block_coefficients * channels;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(target_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(target_node, source_node)| {
                for block_coefficient in 0..block_coefficients {
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let source_start = l_major_coefficient * channels;
                    let target_start = block_coefficient * channels;
                    for channel in 0..channels {
                        target_node[target_start + channel] *= source_node[source_start + channel];
                    }
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let source_start = source_node_start + l_major_coefficient * channels;
            let target_start = target_node_start + block_coefficient * channels;
            for channel in 0..channels {
                target[target_start + channel] *= source[source_start + channel];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(
    feature = "parallel",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )
))]
fn modulate_l_major_radial_real_block_to_m_conv_compact_in_place_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    compact: &mut [f64],
) {
    let source_node_stride = coefficient_count * channels;
    let input_len = block_coefficients * channels;
    let compact_node_stride = 2 * input_len;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        compact
            .par_chunks_mut(compact_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(compact_node, source_node)| {
                let (compact_real, compact_imag) = compact_node.split_at_mut(input_len);
                for block_coefficient in 0..block_coefficients {
                    let real_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let imag_coefficient = l_to_m_permutation
                        [coefficient_offset + block_coefficients + block_coefficient];
                    let input_real_start = real_coefficient * channels;
                    let input_imag_start = imag_coefficient * channels;
                    let feature_start = block_coefficient * channels;
                    for channel in 0..channels {
                        let radial_value = compact_real[feature_start + channel];
                        compact_real[feature_start + channel] =
                            source_node[input_real_start + channel] * radial_value;
                        compact_imag[feature_start + channel] =
                            source_node[input_imag_start + channel] * radial_value;
                    }
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let compact_node_start = node * compact_node_stride;
        let compact_imag_start = compact_node_start + input_len;
        for block_coefficient in 0..block_coefficients {
            let real_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let imag_coefficient =
                l_to_m_permutation[coefficient_offset + block_coefficients + block_coefficient];
            let input_real_start = source_node_start + real_coefficient * channels;
            let input_imag_start = source_node_start + imag_coefficient * channels;
            let feature_start = block_coefficient * channels;
            for channel in 0..channels {
                let radial_index = compact_node_start + feature_start + channel;
                let radial_value = compact[radial_index];
                compact[radial_index] = source[input_real_start + channel] * radial_value;
                compact[compact_imag_start + feature_start + channel] =
                    source[input_imag_start + channel] * radial_value;
            }
        }
    }
}

fn scatter_m_major_block(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let target_node_stride = coefficient_count * channels;
    let source_node_stride = block_coefficients * channels;
    for node in 0..nodes {
        let source_start = node * source_node_stride;
        let target_start = node * target_node_stride + coefficient_offset * channels;
        target[target_start..target_start + source_node_stride]
            .copy_from_slice(&source[source_start..source_start + source_node_stride]);
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_m_major_block_strided(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source_node_stride: usize,
    source_feature_offset: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let target_node_stride = coefficient_count * channels;
    let copy_len = block_coefficients * channels;
    for node in 0..nodes {
        let source_start = node * source_node_stride + source_feature_offset;
        let target_start = node * target_node_stride + coefficient_offset * channels;
        target[target_start..target_start + copy_len]
            .copy_from_slice(&source[source_start..source_start + copy_len]);
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_add_l_major_block_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let target_node_stride = coefficient_count * channels;
    let source_node_stride = block_coefficients * channels;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(target_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(target_node, source_node)| {
                for block_coefficient in 0..block_coefficients {
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let source_start = block_coefficient * channels;
                    let target_start = l_major_coefficient * channels;
                    for channel in 0..channels {
                        target_node[target_start + channel] += source_node[source_start + channel];
                    }
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let source_start = source_node_start + block_coefficient * channels;
            let target_start = target_node_start + l_major_coefficient * channels;
            for channel in 0..channels {
                target[target_start + channel] += source[source_start + channel];
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_l_major_order_results_from_m_permutation_strided(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    l_to_m_permutation: &[usize],
    scales: Option<&[f64]>,
    results: &[So2ConvForwardOrderResult],
    extra_channels: usize,
    target: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let needs_extra_m0 = results
        .iter()
        .any(|result| result.extra_m0_feature_offset.is_some());
    if needs_extra_m0 && extra_m0_out.is_none() {
        return Err(invalid_so2("validated extra m0 output is missing"));
    }
    if needs_extra_m0 && extra_channels == 0 {
        return Err(invalid_so2(
            "validated extra m0 output has zero channel count",
        ));
    }
    let extra_m0_out = if needs_extra_m0 { extra_m0_out } else { None };

    let target_node_stride = coefficient_count * channels;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        match (scales, extra_m0_out) {
            (Some(scales), Some(extra_out)) => target
                .par_chunks_mut(target_node_stride)
                .zip(extra_out.par_chunks_mut(extra_channels))
                .zip(scales.par_iter())
                .enumerate()
                .for_each(|(node, ((target_node, extra_node), scale))| {
                    scatter_l_major_order_results_node_from_m_permutation_strided(
                        node,
                        channels,
                        l_to_m_permutation,
                        Some(*scale),
                        results,
                        target_node,
                        Some(extra_node),
                    );
                }),
            (Some(scales), None) => target
                .par_chunks_mut(target_node_stride)
                .zip(scales.par_iter())
                .enumerate()
                .for_each(|(node, (target_node, scale))| {
                    scatter_l_major_order_results_node_from_m_permutation_strided(
                        node,
                        channels,
                        l_to_m_permutation,
                        Some(*scale),
                        results,
                        target_node,
                        None,
                    );
                }),
            (None, Some(extra_out)) => target
                .par_chunks_mut(target_node_stride)
                .zip(extra_out.par_chunks_mut(extra_channels))
                .enumerate()
                .for_each(|(node, (target_node, extra_node))| {
                    scatter_l_major_order_results_node_from_m_permutation_strided(
                        node,
                        channels,
                        l_to_m_permutation,
                        None,
                        results,
                        target_node,
                        Some(extra_node),
                    );
                }),
            (None, None) => target
                .par_chunks_mut(target_node_stride)
                .enumerate()
                .for_each(|(node, target_node)| {
                    scatter_l_major_order_results_node_from_m_permutation_strided(
                        node,
                        channels,
                        l_to_m_permutation,
                        None,
                        results,
                        target_node,
                        None,
                    );
                }),
        }
        return Ok(());
    }

    match (scales, extra_m0_out) {
        (Some(scales), Some(extra_out)) => {
            for (node, (target_node, extra_node)) in target
                .chunks_mut(target_node_stride)
                .zip(extra_out.chunks_mut(extra_channels))
                .enumerate()
                .take(nodes)
            {
                scatter_l_major_order_results_node_from_m_permutation_strided(
                    node,
                    channels,
                    l_to_m_permutation,
                    Some(scales[node]),
                    results,
                    target_node,
                    Some(extra_node),
                );
            }
        }
        (Some(scales), None) => {
            for (node, target_node) in target
                .chunks_mut(target_node_stride)
                .enumerate()
                .take(nodes)
            {
                scatter_l_major_order_results_node_from_m_permutation_strided(
                    node,
                    channels,
                    l_to_m_permutation,
                    Some(scales[node]),
                    results,
                    target_node,
                    None,
                );
            }
        }
        (None, Some(extra_out)) => {
            for (node, (target_node, extra_node)) in target
                .chunks_mut(target_node_stride)
                .zip(extra_out.chunks_mut(extra_channels))
                .enumerate()
                .take(nodes)
            {
                scatter_l_major_order_results_node_from_m_permutation_strided(
                    node,
                    channels,
                    l_to_m_permutation,
                    None,
                    results,
                    target_node,
                    Some(extra_node),
                );
            }
        }
        (None, None) => {
            for (node, target_node) in target
                .chunks_mut(target_node_stride)
                .enumerate()
                .take(nodes)
            {
                scatter_l_major_order_results_node_from_m_permutation_strided(
                    node,
                    channels,
                    l_to_m_permutation,
                    None,
                    results,
                    target_node,
                    None,
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn scatter_l_major_order_results_node_from_m_permutation_strided(
    node: usize,
    channels: usize,
    l_to_m_permutation: &[usize],
    scale: Option<f64>,
    results: &[So2ConvForwardOrderResult],
    target_node: &mut [f64],
    mut extra_m0_node: Option<&mut [f64]>,
) {
    for result in results {
        let source_node_start = node * result.source_node_stride + result.source_feature_offset;
        for block_coefficient in 0..result.block_coefficients {
            let l_major_coefficient = l_to_m_permutation[result.coeff_offset + block_coefficient];
            let source_start = source_node_start + block_coefficient * channels;
            let target_start = l_major_coefficient * channels;
            let source_values = &result.block_out[source_start..source_start + channels];
            let target_values = &mut target_node[target_start..target_start + channels];
            if let Some(scale) = scale {
                for channel in 0..channels {
                    target_values[channel] = source_values[channel] * scale;
                }
            } else {
                target_values.copy_from_slice(source_values);
            }
        }

        if let (Some(extra_offset), Some(extra_node)) =
            (result.extra_m0_feature_offset, extra_m0_node.as_deref_mut())
        {
            let source_start = node * result.source_node_stride + extra_offset;
            extra_node
                .copy_from_slice(&result.block_out[source_start..source_start + extra_node.len()]);
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_m_major_order_results_strided(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    results: &[So2ConvForwardOrderResult],
    extra_channels: usize,
    target: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let needs_extra_m0 = results
        .iter()
        .any(|result| result.extra_m0_feature_offset.is_some());
    if needs_extra_m0 && extra_m0_out.is_none() {
        return Err(invalid_so2("validated extra m0 output is missing"));
    }
    if needs_extra_m0 && extra_channels == 0 {
        return Err(invalid_so2(
            "validated extra m0 output has zero channel count",
        ));
    }
    let extra_m0_out = if needs_extra_m0 { extra_m0_out } else { None };
    let target_node_stride = coefficient_count * channels;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        match extra_m0_out {
            Some(extra_out) => target
                .par_chunks_mut(target_node_stride)
                .zip(extra_out.par_chunks_mut(extra_channels))
                .enumerate()
                .for_each(|(node, (target_node, extra_node))| {
                    scatter_m_major_order_results_node_strided(
                        node,
                        channels,
                        results,
                        target_node,
                        Some(extra_node),
                    );
                }),
            None => target
                .par_chunks_mut(target_node_stride)
                .enumerate()
                .for_each(|(node, target_node)| {
                    scatter_m_major_order_results_node_strided(
                        node,
                        channels,
                        results,
                        target_node,
                        None,
                    );
                }),
        }
        return Ok(());
    }

    match extra_m0_out {
        Some(extra_out) => {
            for (node, (target_node, extra_node)) in target
                .chunks_mut(target_node_stride)
                .zip(extra_out.chunks_mut(extra_channels))
                .enumerate()
                .take(nodes)
            {
                scatter_m_major_order_results_node_strided(
                    node,
                    channels,
                    results,
                    target_node,
                    Some(extra_node),
                );
            }
        }
        None => {
            for (node, target_node) in target
                .chunks_mut(target_node_stride)
                .enumerate()
                .take(nodes)
            {
                scatter_m_major_order_results_node_strided(
                    node,
                    channels,
                    results,
                    target_node,
                    None,
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn scatter_m_major_order_results_node_strided(
    node: usize,
    channels: usize,
    results: &[So2ConvForwardOrderResult],
    target_node: &mut [f64],
    mut extra_m0_node: Option<&mut [f64]>,
) {
    for result in results {
        let source_node_start = node * result.source_node_stride + result.source_feature_offset;
        for block_coefficient in 0..result.block_coefficients {
            let source_start = source_node_start + block_coefficient * channels;
            let target_start = (result.coeff_offset + block_coefficient) * channels;
            target_node[target_start..target_start + channels]
                .copy_from_slice(&result.block_out[source_start..source_start + channels]);
        }

        if let (Some(extra_offset), Some(extra_node)) =
            (result.extra_m0_feature_offset, extra_m0_node.as_deref_mut())
        {
            let source_start = node * result.source_node_stride + extra_offset;
            extra_node
                .copy_from_slice(&result.block_out[source_start..source_start + extra_node.len()]);
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_l_major_block_from_m_permutation_strided(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source_node_stride: usize,
    source_feature_offset: usize,
    l_to_m_permutation: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let target_node_stride = coefficient_count * channels;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(target_node_stride)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(target_node, source_node)| {
                for block_coefficient in 0..block_coefficients {
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let source_start = source_feature_offset + block_coefficient * channels;
                    let target_start = l_major_coefficient * channels;
                    target_node[target_start..target_start + channels]
                        .copy_from_slice(&source_node[source_start..source_start + channels]);
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_node_start = node * source_node_stride + source_feature_offset;
        let target_node_start = node * target_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let source_start = source_node_start + block_coefficient * channels;
            let target_start = target_node_start + l_major_coefficient * channels;
            target[target_start..target_start + channels]
                .copy_from_slice(&source[source_start..source_start + channels]);
        }
    }
}

fn scale_l_major_rows(nodes: usize, node_stride: usize, scales: &[f64], target: &mut [f64]) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(node_stride)
            .zip(scales.par_iter())
            .for_each(|(target_node, scale)| {
                for value in target_node {
                    *value *= *scale;
                }
            });
        return;
    }

    for (node, scale) in scales.iter().copied().enumerate().take(nodes) {
        let start = node * node_stride;
        for value in &mut target[start..start + node_stride] {
            *value *= scale;
        }
    }
}

fn copy_extra_m0_from_strided_source(
    nodes: usize,
    extra_channels: usize,
    source_node_stride: usize,
    source_feature_offset: usize,
    source: &[f64],
    target: &mut [f64],
) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(extra_channels)
            .zip(source.par_chunks(source_node_stride))
            .for_each(|(target_node, source_node)| {
                let source_start = source_feature_offset;
                target_node
                    .copy_from_slice(&source_node[source_start..source_start + extra_channels]);
            });
        return;
    }

    for node in 0..nodes {
        let source_start = node * source_node_stride + source_feature_offset;
        let target_start = node * extra_channels;
        target[target_start..target_start + extra_channels]
            .copy_from_slice(&source[source_start..source_start + extra_channels]);
    }
}

#[allow(clippy::uninit_vec, dead_code, unsafe_code)]
fn uninit_f64_vec(len: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(len);
    // Invariant: callers use this only for f64 buffers that are fully
    // overwritten by gather loops or BLAS calls before any read.
    unsafe {
        values.set_len(len);
    }
    values
}

fn reset_zeroed_f64_vec(values: &mut Vec<f64>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_vjp_l_major_m0_from_permutation_no_bias_rows_add(
    rows: usize,
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    weight: &[f64],
    grad_out_l_major: &[f64],
    grad_input: &mut [f64],
) {
    let grad_out_node_stride = coefficient_count * out_channels;
    let input_len = block_coefficients * in_channels;
    for block_coefficient in 0..block_coefficients {
        let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
        let grad_out_feature_offset = l_major_coefficient * out_channels;
        let weight_row_offset = block_coefficient * out_channels * input_len;
        accelerate_blas::dense_vjp_input_strided_grad_out_add(
            grad_out_l_major,
            rows,
            grad_out_node_stride,
            grad_out_feature_offset,
            out_channels,
            &weight[weight_row_offset..],
            input_len,
            input_len,
            grad_input,
        );
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_vjp_l_major_m0_with_extra_from_permutation_no_bias_rows_set(
    rows: usize,
    coefficient_count: usize,
    in_channels: usize,
    out_channels: usize,
    extra_m0_output_channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    l_to_m_permutation: &[usize],
    weight: &[f64],
    grad_out_l_major: &[f64],
    grad_extra_m0: &[f64],
    grad_input: &mut [f64],
) {
    let grad_out_node_stride = coefficient_count * out_channels;
    let input_len = block_coefficients * in_channels;

    accelerate_blas::dense_vjp_input_set(
        grad_extra_m0,
        rows,
        extra_m0_output_channels,
        &weight[..extra_m0_output_channels * input_len],
        input_len,
        grad_input,
    );

    for block_coefficient in 0..block_coefficients {
        let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
        let grad_out_feature_offset = l_major_coefficient * out_channels;
        let weight_row_offset =
            (extra_m0_output_channels + block_coefficient * out_channels) * input_len;
        accelerate_blas::dense_vjp_input_strided_grad_out_add(
            grad_out_l_major,
            rows,
            grad_out_node_stride,
            grad_out_feature_offset,
            out_channels,
            &weight[weight_row_offset..],
            input_len,
            input_len,
            grad_input,
        );
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_add_l_major_radial_vjp_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    radial_node_stride: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    input_feature_dim: usize,
    radial_offset: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial: &[f64],
    compact_grad_modulated: &[f64],
    grad_input_l_major: &mut [f64],
    grad_radial: &mut [f64],
) {
    let l_node_stride = coefficient_count * channels;
    let compact_node_stride = block_coefficients * channels;
    let coefficients_per_copy = input_feature_dim / channels;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        grad_input_l_major
            .par_chunks_mut(l_node_stride)
            .zip(grad_radial.par_chunks_mut(radial_node_stride))
            .zip(input_l_major.par_chunks(l_node_stride))
            .zip(radial.par_chunks(radial_node_stride))
            .zip(compact_grad_modulated.par_chunks(compact_node_stride))
            .for_each(
                |(
                    (((grad_input_node, grad_radial_node), input_node), radial_node),
                    compact_node,
                )| {
                    for block_coefficient in 0..block_coefficients {
                        let l_major_coefficient =
                            l_to_m_permutation[coefficient_offset + block_coefficient];
                        let radial_coefficient = block_coefficient % coefficients_per_copy;
                        let input_start = l_major_coefficient * channels;
                        let compact_start = block_coefficient * channels;
                        let radial_start = radial_offset + radial_coefficient * channels;
                        for channel in 0..channels {
                            let grad = compact_node[compact_start + channel];
                            let radial_value = radial_node[radial_start + channel];
                            grad_input_node[input_start + channel] += radial_value * grad;
                            grad_radial_node[radial_start + channel] +=
                                input_node[input_start + channel] * grad;
                        }
                    }
                },
            );
        return;
    }

    for node in 0..nodes {
        let input_node_start = node * l_node_stride;
        let radial_node_start = node * radial_node_stride;
        let compact_node_start = node * compact_node_stride;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let radial_coefficient = block_coefficient % coefficients_per_copy;
            let input_start = input_node_start + l_major_coefficient * channels;
            let compact_start = compact_node_start + block_coefficient * channels;
            let radial_start = radial_node_start + radial_offset + radial_coefficient * channels;
            for channel in 0..channels {
                let grad = compact_grad_modulated[compact_start + channel];
                let radial_value = radial[radial_start + channel];
                grad_input_l_major[input_start + channel] += radial_value * grad;
                grad_radial[radial_start + channel] += input_l_major[input_start + channel] * grad;
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_add_l_major_radial_vjp_block_from_m_permutation(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    radial_node_stride: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    input_feature_dim: usize,
    radial_offset: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial: &[f64],
    compact_grad_modulated: &[f64],
    grad_input_l_major: &mut [f64],
    block_grad_radial: &mut [f64],
) {
    let l_node_stride = coefficient_count * channels;
    let compact_node_stride = block_coefficients * channels;
    let coefficients_per_copy = input_feature_dim / channels;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        grad_input_l_major
            .par_chunks_mut(l_node_stride)
            .zip(block_grad_radial.par_chunks_mut(input_feature_dim))
            .zip(input_l_major.par_chunks(l_node_stride))
            .zip(radial.par_chunks(radial_node_stride))
            .zip(compact_grad_modulated.par_chunks(compact_node_stride))
            .for_each(
                |(
                    (((grad_input_node, block_grad_radial_node), input_node), radial_node),
                    compact_node,
                )| {
                    for block_coefficient in 0..block_coefficients {
                        let l_major_coefficient =
                            l_to_m_permutation[coefficient_offset + block_coefficient];
                        let radial_coefficient = block_coefficient % coefficients_per_copy;
                        let input_start = l_major_coefficient * channels;
                        let compact_start = block_coefficient * channels;
                        let radial_start = radial_offset + radial_coefficient * channels;
                        let block_radial_start = radial_coefficient * channels;
                        for channel in 0..channels {
                            let grad = compact_node[compact_start + channel];
                            let radial_value = radial_node[radial_start + channel];
                            grad_input_node[input_start + channel] += radial_value * grad;
                            block_grad_radial_node[block_radial_start + channel] +=
                                input_node[input_start + channel] * grad;
                        }
                    }
                },
            );
        return;
    }

    for node in 0..nodes {
        let input_node_start = node * l_node_stride;
        let radial_node_start = node * radial_node_stride;
        let compact_node_start = node * compact_node_stride;
        let block_grad_radial_start = node * input_feature_dim;
        for block_coefficient in 0..block_coefficients {
            let l_major_coefficient = l_to_m_permutation[coefficient_offset + block_coefficient];
            let radial_coefficient = block_coefficient % coefficients_per_copy;
            let input_start = input_node_start + l_major_coefficient * channels;
            let compact_start = compact_node_start + block_coefficient * channels;
            let radial_start = radial_node_start + radial_offset + radial_coefficient * channels;
            let block_radial_start = block_grad_radial_start + radial_coefficient * channels;
            for channel in 0..channels {
                let grad = compact_grad_modulated[compact_start + channel];
                let radial_value = radial[radial_start + channel];
                grad_input_l_major[input_start + channel] += radial_value * grad;
                block_grad_radial[block_radial_start + channel] +=
                    input_l_major[input_start + channel] * grad;
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn scatter_add_l_major_radial_vjp_block_from_m_permutation_in_place(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    input_feature_dim: usize,
    l_to_m_permutation: &[usize],
    input_l_major: &[f64],
    radial_then_grad_radial: &mut [f64],
    compact_grad_modulated: &[f64],
    grad_input_l_major: &mut [f64],
) {
    let l_node_stride = coefficient_count * channels;
    let compact_node_stride = block_coefficients * channels;
    let coefficients_per_copy = input_feature_dim / channels;
    let copies = block_coefficients / coefficients_per_copy;

    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        grad_input_l_major
            .par_chunks_mut(l_node_stride)
            .zip(radial_then_grad_radial.par_chunks_mut(input_feature_dim))
            .zip(input_l_major.par_chunks(l_node_stride))
            .zip(compact_grad_modulated.par_chunks(compact_node_stride))
            .for_each(
                |(((grad_input_node, radial_grad_node), input_node), compact_node)| {
                    for radial_coefficient in 0..coefficients_per_copy {
                        let radial_start = radial_coefficient * channels;
                        for channel in 0..channels {
                            let radial_value = radial_grad_node[radial_start + channel];
                            let mut grad_radial = 0.0;
                            for copy in 0..copies {
                                let block_coefficient =
                                    copy * coefficients_per_copy + radial_coefficient;
                                let l_major_coefficient =
                                    l_to_m_permutation[coefficient_offset + block_coefficient];
                                let input_start = l_major_coefficient * channels;
                                let compact_start = block_coefficient * channels;
                                let grad = compact_node[compact_start + channel];
                                grad_input_node[input_start + channel] += radial_value * grad;
                                grad_radial += input_node[input_start + channel] * grad;
                            }
                            radial_grad_node[radial_start + channel] = grad_radial;
                        }
                    }
                },
            );
        return;
    }

    for node in 0..nodes {
        let input_node_start = node * l_node_stride;
        let compact_node_start = node * compact_node_stride;
        let radial_node_start = node * input_feature_dim;
        for radial_coefficient in 0..coefficients_per_copy {
            let radial_start = radial_node_start + radial_coefficient * channels;
            for channel in 0..channels {
                let radial_value = radial_then_grad_radial[radial_start + channel];
                let mut grad_radial = 0.0;
                for copy in 0..copies {
                    let block_coefficient = copy * coefficients_per_copy + radial_coefficient;
                    let l_major_coefficient =
                        l_to_m_permutation[coefficient_offset + block_coefficient];
                    let input_start = input_node_start + l_major_coefficient * channels;
                    let compact_start = compact_node_start + block_coefficient * channels;
                    let grad = compact_grad_modulated[compact_start + channel];
                    grad_input_l_major[input_start + channel] += radial_value * grad;
                    grad_radial += input_l_major[input_start + channel] * grad;
                }
                radial_then_grad_radial[radial_start + channel] = grad_radial;
            }
        }
    }
}

fn scatter_add_m_major_block(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let target_node_stride = coefficient_count * channels;
    let source_node_stride = block_coefficients * channels;
    for node in 0..nodes {
        let source_start = node * source_node_stride;
        let target_start = node * target_node_stride + coefficient_offset * channels;
        for index in 0..source_node_stride {
            target[target_start + index] += source[source_start + index];
        }
    }
}

fn gather_radial_block(
    nodes: usize,
    radial_node_stride: usize,
    radial_offset: usize,
    feature_dim: usize,
    source: &[f64],
) -> Vec<f64> {
    let mut target = vec![0.0; nodes * feature_dim];

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(feature_dim)
            .zip(source.par_chunks(radial_node_stride))
            .for_each(|(target_node, source_node)| {
                target_node
                    .copy_from_slice(&source_node[radial_offset..radial_offset + feature_dim]);
            });
        return target;
    }

    for node in 0..nodes {
        let source_start = node * radial_node_stride + radial_offset;
        let target_start = node * feature_dim;
        target[target_start..target_start + feature_dim]
            .copy_from_slice(&source[source_start..source_start + feature_dim]);
    }
    target
}

fn scatter_add_radial_block(
    nodes: usize,
    radial_node_stride: usize,
    radial_offset: usize,
    feature_dim: usize,
    source: &[f64],
    target: &mut [f64],
) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= so2_parallel_node_threshold() {
        target
            .par_chunks_mut(radial_node_stride)
            .zip(source.par_chunks(feature_dim))
            .for_each(|(target_node, source_node)| {
                let target_start = radial_offset;
                for index in 0..feature_dim {
                    target_node[target_start + index] += source_node[index];
                }
            });
        return;
    }

    for node in 0..nodes {
        let source_start = node * feature_dim;
        let target_start = node * radial_node_stride + radial_offset;
        for index in 0..feature_dim {
            target[target_start + index] += source[source_start + index];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_add_m_major_radial_vjp_block(
    nodes: usize,
    coefficient_count: usize,
    channels: usize,
    radial_node_stride: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    input_feature_dim: usize,
    radial_offset: usize,
    input_m_major: &[f64],
    radial: &[f64],
    compact_grad_modulated: &[f64],
    grad_input_m_major: &mut [f64],
    grad_radial: &mut [f64],
) {
    let m_node_stride = coefficient_count * channels;
    let compact_node_stride = block_coefficients * channels;
    let coefficients_per_copy = input_feature_dim / channels;

    for node in 0..nodes {
        let input_node_start = node * m_node_stride;
        let radial_node_start = node * radial_node_stride;
        let compact_node_start = node * compact_node_stride;
        for block_coefficient in 0..block_coefficients {
            let m_major_coefficient = coefficient_offset + block_coefficient;
            let radial_coefficient = block_coefficient % coefficients_per_copy;
            let input_start = input_node_start + m_major_coefficient * channels;
            let compact_start = compact_node_start + block_coefficient * channels;
            let radial_start = radial_node_start + radial_offset + radial_coefficient * channels;
            for channel in 0..channels {
                let grad = compact_grad_modulated[compact_start + channel];
                let radial_value = radial[radial_start + channel];
                grad_input_m_major[input_start + channel] += radial_value * grad;
                grad_radial[radial_start + channel] += input_m_major[input_start + channel] * grad;
            }
        }
    }
}

fn reduced_coefficient_count(l_max: usize, m_max: usize) -> Option<usize> {
    let mut count = 0usize;
    for degree in 0..=l_max {
        count = count.checked_add(2 * m_max.min(degree) + 1)?;
    }
    Some(count)
}

#[allow(clippy::too_many_arguments)]
fn check_so2_linear_assembly_shapes(
    op: &'static str,
    config: EsenSo2LinearConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: Option<&[f64]>,
    m_weights: &[&[f64]],
    output_like: &[f64],
) -> Result<usize> {
    let input_stride = config.input_node_stride();
    if !input_l_major.len().is_multiple_of(input_stride) {
        return Err(invalid_so2(format!(
            "{op} input length {} is not a multiple of node stride {input_stride}",
            input_l_major.len()
        )));
    }
    let nodes = input_l_major.len() / input_stride;
    check_len(op, nodes * config.output_node_stride(), output_like.len())?;
    check_len(
        "fairchem_esen_so2_linear m0_weight",
        config.m_weight_len(0),
        m0_weight.len(),
    )?;
    if let Some(m0_bias_values) = m0_bias {
        check_len(
            "fairchem_esen_so2_linear m0_bias",
            config.m0_bias_len(),
            m0_bias_values.len(),
        )?;
    }
    check_len(
        "fairchem_esen_so2_linear m_weights",
        config.m_max,
        m_weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "fairchem_esen_so2_linear m_weight",
            config.m_weight_len(order),
            m_weights[order - 1].len(),
        )?;
    }
    if let Some(radial_values) = radial {
        check_len(
            "fairchem_esen_so2_linear radial",
            nodes * config.radial_node_stride(),
            radial_values.len(),
        )?;
    }
    Ok(nodes)
}

fn fill_l2_m2_gate_activation_m0_input(
    nodes: usize,
    channels: usize,
    input_l_major: &[f64],
    gating_scalars: &mut [f64],
    block_input: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
) {
    let input_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let gate_node_stride = 2 * channels;
    let block_node_stride = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels;
    for node in 0..nodes {
        let input_node_start = node * input_node_stride;
        let gate_node_start = node * gate_node_stride;
        let block_node_start = node * block_node_stride;
        let derivative_node_start = node * channels;

        for channel in 0..channels {
            let gate1 = so2_gate_sigmoid(gating_scalars[gate_node_start + channel]);
            let gate2 = so2_gate_sigmoid(gating_scalars[gate_node_start + channels + channel]);
            gating_scalars[gate_node_start + channel] = gate1;
            gating_scalars[gate_node_start + channels + channel] = gate2;

            let scalar_index = input_node_start + channel;
            let (scalar_value, scalar_derivative) =
                so2_silu_value_and_derivative(input_l_major[scalar_index]);
            block_input[block_node_start + channel] = scalar_value;
            scalar_silu_derivatives[derivative_node_start + channel] = scalar_derivative;

            block_input[block_node_start + channels + channel] =
                input_l_major[input_node_start + 2 * channels + channel] * gate1;
            block_input[block_node_start + 2 * channels + channel] =
                input_l_major[input_node_start + 6 * channels + channel] * gate2;
        }
    }
}

fn fill_l2_m2_gate_activation_gated_order_input(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    input_l_major: &[f64],
    gate_sigmoids: &[f64],
    block_input: &mut [f64],
) {
    let input_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let gate_node_stride = 2 * channels;
    let block_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate1 = gate_sigmoids[gate_node_start + channel];
                let gate2 = gate_sigmoids[gate_node_start + channels + channel];
                block_input[block_node_start + channel] =
                    input_l_major[input_node_start + 3 * channels + channel] * gate1;
                block_input[block_node_start + channels + channel] =
                    input_l_major[input_node_start + 7 * channels + channel] * gate2;
                block_input[block_node_start + 2 * channels + channel] =
                    input_l_major[input_node_start + channels + channel] * gate1;
                block_input[block_node_start + 3 * channels + channel] =
                    input_l_major[input_node_start + 5 * channels + channel] * gate2;
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate2 = gate_sigmoids[gate_node_start + channels + channel];
                block_input[block_node_start + channel] =
                    input_l_major[input_node_start + 8 * channels + channel] * gate2;
                block_input[block_node_start + channels + channel] =
                    input_l_major[input_node_start + 4 * channels + channel] * gate2;
            }
        }
        return;
    }
    for node in 0..nodes {
        let input_node_start = node * input_node_stride;
        let gate_node_start = node * gate_node_stride;
        let block_node_start = node * block_node_stride;
        for (source_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let gate_offset = if source_coefficient <= 3 { 0 } else { channels };
            let input_start = input_node_start + source_coefficient * channels;
            let block_start = block_node_start + source_offset * channels;
            let gate_start = gate_node_start + gate_offset;
            for channel in 0..channels {
                block_input[block_start + channel] =
                    input_l_major[input_start + channel] * gate_sigmoids[gate_start + channel];
            }
        }
    }
}

fn fill_l2_m2_radial_order_input(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    radial_offset: usize,
    input_l_major: &[f64],
    radial: &[f64],
    block_input: &mut [f64],
) {
    let input_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let radial_node_stride = 6 * channels;
    let block_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                block_input[block_node_start + channel] =
                    input_l_major[input_node_start + channel] * radial[radial_node_start + channel];
                block_input[block_node_start + channels + channel] = input_l_major
                    [input_node_start + 2 * channels + channel]
                    * radial[radial_node_start + channels + channel];
                block_input[block_node_start + 2 * channels + channel] = input_l_major
                    [input_node_start + 6 * channels + channel]
                    * radial[radial_node_start + 2 * channels + channel];
            }
        }
        return;
    }
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let radial0 = radial[radial_node_start + channel];
                let radial1 = radial[radial_node_start + channels + channel];
                block_input[block_node_start + channel] =
                    input_l_major[input_node_start + 3 * channels + channel] * radial0;
                block_input[block_node_start + channels + channel] =
                    input_l_major[input_node_start + 7 * channels + channel] * radial1;
                block_input[block_node_start + 2 * channels + channel] =
                    input_l_major[input_node_start + channels + channel] * radial0;
                block_input[block_node_start + 3 * channels + channel] =
                    input_l_major[input_node_start + 5 * channels + channel] * radial1;
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let radial0 = radial[radial_node_start + channel];
                block_input[block_node_start + channel] =
                    input_l_major[input_node_start + 8 * channels + channel] * radial0;
                block_input[block_node_start + channels + channel] =
                    input_l_major[input_node_start + 4 * channels + channel] * radial0;
            }
        }
        return;
    }
    for node in 0..nodes {
        let input_node_start = node * input_node_stride;
        let radial_node_start = node * radial_node_stride + radial_offset;
        let block_node_start = node * block_node_stride;
        let radial_m_size = if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
            sources.len()
        } else {
            sources.len() / 2
        };
        for (source_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let input_start = input_node_start + source_coefficient * channels;
            let block_start = block_node_start + source_offset * channels;
            let radial_start = radial_node_start + (source_offset % radial_m_size) * channels;
            for channel in 0..channels {
                block_input[block_start + channel] =
                    input_l_major[input_start + channel] * radial[radial_start + channel];
            }
        }
    }
}

fn scatter_l2_m2_order_block(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = sources.len() * channels;
    let target_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start..target_node_start + channels]
                .copy_from_slice(&source[source_node_start..source_node_start + channels]);
            target[target_node_start + 2 * channels..target_node_start + 3 * channels]
                .copy_from_slice(
                    &source[source_node_start + channels..source_node_start + 2 * channels],
                );
            target[target_node_start + 6 * channels..target_node_start + 7 * channels]
                .copy_from_slice(
                    &source[source_node_start + 2 * channels..source_node_start + 3 * channels],
                );
        }
        return;
    }
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start + 3 * channels..target_node_start + 4 * channels]
                .copy_from_slice(&source[source_node_start..source_node_start + channels]);
            target[target_node_start + 7 * channels..target_node_start + 8 * channels]
                .copy_from_slice(
                    &source[source_node_start + channels..source_node_start + 2 * channels],
                );
            target[target_node_start + channels..target_node_start + 2 * channels].copy_from_slice(
                &source[source_node_start + 2 * channels..source_node_start + 3 * channels],
            );
            target[target_node_start + 5 * channels..target_node_start + 6 * channels]
                .copy_from_slice(
                    &source[source_node_start + 3 * channels..source_node_start + 4 * channels],
                );
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start + 8 * channels..target_node_start + 9 * channels]
                .copy_from_slice(&source[source_node_start..source_node_start + channels]);
            target[target_node_start + 4 * channels..target_node_start + 5 * channels]
                .copy_from_slice(
                    &source[source_node_start + channels..source_node_start + 2 * channels],
                );
        }
        return;
    }
    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for (source_offset, target_coefficient) in sources.iter().copied().enumerate() {
            let source_start = source_node_start + source_offset * channels;
            let target_start = target_node_start + target_coefficient * channels;
            target[target_start..target_start + channels]
                .copy_from_slice(&source[source_start..source_start + channels]);
        }
    }
}

fn scatter_l2_m2_m0_dense_out(
    nodes: usize,
    channels: usize,
    extra_channels: usize,
    dense_out: &[f64],
    out_l_major: &mut [f64],
    mut extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let dense_node_stride = extra_channels + L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels;
    let out_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    for node in 0..nodes {
        let dense_node_start = node * dense_node_stride;
        let out_node_start = node * out_node_stride;
        if extra_channels > 0 {
            let Some(extra) = extra_m0_out.as_deref_mut() else {
                return Err(invalid_so2("validated extra m0 output is missing"));
            };
            let extra_start = node * extra_channels;
            extra[extra_start..extra_start + extra_channels]
                .copy_from_slice(&dense_out[dense_node_start..dense_node_start + extra_channels]);
        }
        let dense_main_start = dense_node_start + extra_channels;
        out_l_major[out_node_start..out_node_start + channels]
            .copy_from_slice(&dense_out[dense_main_start..dense_main_start + channels]);
        out_l_major[out_node_start + 2 * channels..out_node_start + 3 * channels].copy_from_slice(
            &dense_out[dense_main_start + channels..dense_main_start + 2 * channels],
        );
        out_l_major[out_node_start + 6 * channels..out_node_start + 7 * channels].copy_from_slice(
            &dense_out[dense_main_start + 2 * channels..dense_main_start + 3 * channels],
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_l2_m2_gate_conv2_m0_input_from_conv1_dense_out(
    nodes: usize,
    channels: usize,
    extra_channels: usize,
    conv1_dense_out: &[f64],
    conv1_out_l_major: &mut [f64],
    gate_sigmoids: &mut [f64],
    conv2_m0_input: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
) {
    let dense_node_stride = extra_channels + L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels;
    let conv1_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let gate_node_stride = 2 * channels;
    let conv2_input_node_stride = L2_M2_ORDER0_L_MAJOR_SOURCES.len() * channels;
    for node in 0..nodes {
        let dense_node_start = node * dense_node_stride;
        let dense_main_start = dense_node_start + extra_channels;
        let conv1_node_start = node * conv1_node_stride;
        let gate_node_start = node * gate_node_stride;
        let conv2_input_node_start = node * conv2_input_node_stride;
        let derivative_node_start = node * channels;
        for channel in 0..channels {
            let gate1 = so2_gate_sigmoid(conv1_dense_out[dense_node_start + channel]);
            let gate2 = so2_gate_sigmoid(conv1_dense_out[dense_node_start + channels + channel]);
            gate_sigmoids[gate_node_start + channel] = gate1;
            gate_sigmoids[gate_node_start + channels + channel] = gate2;

            let scalar = conv1_dense_out[dense_main_start + channel];
            conv1_out_l_major[conv1_node_start + channel] = scalar;
            let (scalar_value, scalar_derivative) = so2_silu_value_and_derivative(scalar);
            conv2_m0_input[conv2_input_node_start + channel] = scalar_value;
            scalar_silu_derivatives[derivative_node_start + channel] = scalar_derivative;

            let value_l1 = conv1_dense_out[dense_main_start + channels + channel];
            conv1_out_l_major[conv1_node_start + 2 * channels + channel] = value_l1;
            conv2_m0_input[conv2_input_node_start + channels + channel] = value_l1 * gate1;

            let value_l2 = conv1_dense_out[dense_main_start + 2 * channels + channel];
            conv1_out_l_major[conv1_node_start + 6 * channels + channel] = value_l2;
            conv2_m0_input[conv2_input_node_start + 2 * channels + channel] = value_l2 * gate2;
        }
    }
}

fn scatter_l2_m2_order_block_and_fill_gate_input(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    conv1_block_out: &[f64],
    gate_sigmoids: &[f64],
    conv1_out_l_major: &mut [f64],
    conv2_block_input: &mut [f64],
) {
    let conv1_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let gate_node_stride = 2 * channels;
    let block_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let conv1_node_start = node * conv1_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate1 = gate_sigmoids[gate_node_start + channel];
                let gate2 = gate_sigmoids[gate_node_start + channels + channel];

                let value3 = conv1_block_out[block_node_start + channel];
                conv1_out_l_major[conv1_node_start + 3 * channels + channel] = value3;
                conv2_block_input[block_node_start + channel] = value3 * gate1;

                let value7 = conv1_block_out[block_node_start + channels + channel];
                conv1_out_l_major[conv1_node_start + 7 * channels + channel] = value7;
                conv2_block_input[block_node_start + channels + channel] = value7 * gate2;

                let value1 = conv1_block_out[block_node_start + 2 * channels + channel];
                conv1_out_l_major[conv1_node_start + channels + channel] = value1;
                conv2_block_input[block_node_start + 2 * channels + channel] = value1 * gate1;

                let value5 = conv1_block_out[block_node_start + 3 * channels + channel];
                conv1_out_l_major[conv1_node_start + 5 * channels + channel] = value5;
                conv2_block_input[block_node_start + 3 * channels + channel] = value5 * gate2;
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let conv1_node_start = node * conv1_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate2 = gate_sigmoids[gate_node_start + channels + channel];

                let value8 = conv1_block_out[block_node_start + channel];
                conv1_out_l_major[conv1_node_start + 8 * channels + channel] = value8;
                conv2_block_input[block_node_start + channel] = value8 * gate2;

                let value4 = conv1_block_out[block_node_start + channels + channel];
                conv1_out_l_major[conv1_node_start + 4 * channels + channel] = value4;
                conv2_block_input[block_node_start + channels + channel] = value4 * gate2;
            }
        }
        return;
    }
    for node in 0..nodes {
        let conv1_node_start = node * conv1_node_stride;
        let gate_node_start = node * gate_node_stride;
        let block_node_start = node * block_node_stride;
        for (source_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let block_start = block_node_start + source_offset * channels;
            let conv1_start = conv1_node_start + source_coefficient * channels;
            let gate_offset = if source_coefficient <= 3 { 0 } else { channels };
            let gate_start = gate_node_start + gate_offset;
            for channel in 0..channels {
                let value = conv1_block_out[block_start + channel];
                conv1_out_l_major[conv1_start + channel] = value;
                conv2_block_input[block_start + channel] =
                    value * gate_sigmoids[gate_start + channel];
            }
        }
    }
}

fn gather_l2_m2_order_block(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let target_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start..target_node_start + channels]
                .copy_from_slice(&source[source_node_start..source_node_start + channels]);
            target[target_node_start + channels..target_node_start + 2 * channels].copy_from_slice(
                &source[source_node_start + 2 * channels..source_node_start + 3 * channels],
            );
            target[target_node_start + 2 * channels..target_node_start + 3 * channels]
                .copy_from_slice(
                    &source[source_node_start + 6 * channels..source_node_start + 7 * channels],
                );
        }
        return;
    }
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start..target_node_start + channels].copy_from_slice(
                &source[source_node_start + 3 * channels..source_node_start + 4 * channels],
            );
            target[target_node_start + channels..target_node_start + 2 * channels].copy_from_slice(
                &source[source_node_start + 7 * channels..source_node_start + 8 * channels],
            );
            target[target_node_start + 2 * channels..target_node_start + 3 * channels]
                .copy_from_slice(
                    &source[source_node_start + channels..source_node_start + 2 * channels],
                );
            target[target_node_start + 3 * channels..target_node_start + 4 * channels]
                .copy_from_slice(
                    &source[source_node_start + 5 * channels..source_node_start + 6 * channels],
                );
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            target[target_node_start..target_node_start + channels].copy_from_slice(
                &source[source_node_start + 8 * channels..source_node_start + 9 * channels],
            );
            target[target_node_start + channels..target_node_start + 2 * channels].copy_from_slice(
                &source[source_node_start + 4 * channels..source_node_start + 5 * channels],
            );
        }
        return;
    }
    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for (target_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let source_start = source_node_start + source_coefficient * channels;
            let target_start = target_node_start + target_offset * channels;
            target[target_start..target_start + channels]
                .copy_from_slice(&source[source_start..source_start + channels]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_l2_m2_m0_grad_out_with_extra(
    nodes: usize,
    channels: usize,
    extra_channels: usize,
    sources: &[usize],
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    target: &mut [f64],
) {
    let source_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let target_node_stride = extra_channels + sources.len() * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let source_node_start = node * source_node_stride;
            let target_node_start = node * target_node_stride;
            if let Some(extra) = grad_extra_m0 {
                let extra_start = node * extra_channels;
                target[target_node_start..target_node_start + extra_channels]
                    .copy_from_slice(&extra[extra_start..extra_start + extra_channels]);
            }
            let target_main_start = target_node_start + extra_channels;
            target[target_main_start..target_main_start + channels].copy_from_slice(
                &grad_out_l_major[source_node_start..source_node_start + channels],
            );
            target[target_main_start + channels..target_main_start + 2 * channels].copy_from_slice(
                &grad_out_l_major
                    [source_node_start + 2 * channels..source_node_start + 3 * channels],
            );
            target[target_main_start + 2 * channels..target_main_start + 3 * channels]
                .copy_from_slice(
                    &grad_out_l_major
                        [source_node_start + 6 * channels..source_node_start + 7 * channels],
                );
        }
        return;
    }
    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        if let Some(extra) = grad_extra_m0 {
            let extra_start = node * extra_channels;
            target[target_node_start..target_node_start + extra_channels]
                .copy_from_slice(&extra[extra_start..extra_start + extra_channels]);
        }
        let target_main_start = target_node_start + extra_channels;
        for (target_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let source_start = source_node_start + source_coefficient * channels;
            let target_start = target_main_start + target_offset * channels;
            target[target_start..target_start + channels]
                .copy_from_slice(&grad_out_l_major[source_start..source_start + channels]);
        }
    }
}

fn scatter_add_l2_m2_order_block(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    source: &[f64],
    target: &mut [f64],
) {
    let source_node_stride = sources.len() * channels;
    let target_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    for node in 0..nodes {
        let source_node_start = node * source_node_stride;
        let target_node_start = node * target_node_stride;
        for (source_offset, target_coefficient) in sources.iter().copied().enumerate() {
            let source_start = source_node_start + source_offset * channels;
            let target_start = target_node_start + target_coefficient * channels;
            for channel in 0..channels {
                target[target_start + channel] += source[source_start + channel];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_l2_m2_radial_modulation_vjp_from_order_block(
    nodes: usize,
    channels: usize,
    m_size: usize,
    radial_offset: usize,
    sources: &[usize],
    grad_modulated_block: &[f64],
    input_l_major: &[f64],
    radial: &[f64],
    grad_input_l_major: &mut [f64],
    grad_radial: &mut [f64],
) {
    let input_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let radial_node_stride = 6 * channels;
    let block_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let grad0 = grad_modulated_block[block_node_start + channel];
                let input0 = input_l_major[input_node_start + channel];
                let radial0 = radial[radial_node_start + channel];
                grad_input_l_major[input_node_start + channel] += grad0 * radial0;
                grad_radial[radial_node_start + channel] += grad0 * input0;

                let input2 = input_node_start + 2 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let radial1 = radial_node_start + channels + channel;
                let grad1 = grad_modulated_block[block1];
                let input1 = input_l_major[input2];
                let radial_value1 = radial[radial1];
                grad_input_l_major[input2] += grad1 * radial_value1;
                grad_radial[radial1] += grad1 * input1;

                let input6 = input_node_start + 6 * channels + channel;
                let block2 = block_node_start + 2 * channels + channel;
                let radial2 = radial_node_start + 2 * channels + channel;
                let grad2 = grad_modulated_block[block2];
                let input2_value = input_l_major[input6];
                let radial_value2 = radial[radial2];
                grad_input_l_major[input6] += grad2 * radial_value2;
                grad_radial[radial2] += grad2 * input2_value;
            }
        }
        return;
    }
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let radial0 = radial_node_start + channel;
                let radial1 = radial_node_start + channels + channel;

                let input3 = input_node_start + 3 * channels + channel;
                let block0 = block_node_start + channel;
                let grad0 = grad_modulated_block[block0];
                let input0 = input_l_major[input3];
                let radial_value0 = radial[radial0];
                grad_input_l_major[input3] += grad0 * radial_value0;

                let input7 = input_node_start + 7 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let grad1 = grad_modulated_block[block1];
                let input1 = input_l_major[input7];
                let radial_value1 = radial[radial1];
                grad_input_l_major[input7] += grad1 * radial_value1;

                let input1_index = input_node_start + channels + channel;
                let block2 = block_node_start + 2 * channels + channel;
                let grad2 = grad_modulated_block[block2];
                let input2 = input_l_major[input1_index];
                grad_input_l_major[input1_index] += grad2 * radial_value0;

                let input5 = input_node_start + 5 * channels + channel;
                let block3 = block_node_start + 3 * channels + channel;
                let grad3 = grad_modulated_block[block3];
                let input3_value = input_l_major[input5];
                grad_input_l_major[input5] += grad3 * radial_value1;
                grad_radial[radial0] += grad0 * input0 + grad2 * input2;
                grad_radial[radial1] += grad1 * input1 + grad3 * input3_value;
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let radial_node_start = node * radial_node_stride + radial_offset;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let radial0 = radial_node_start + channel;
                let radial_value0 = radial[radial0];

                let input8 = input_node_start + 8 * channels + channel;
                let block0 = block_node_start + channel;
                let grad0 = grad_modulated_block[block0];
                let input0 = input_l_major[input8];
                grad_input_l_major[input8] += grad0 * radial_value0;

                let input4 = input_node_start + 4 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let grad1 = grad_modulated_block[block1];
                let input1 = input_l_major[input4];
                grad_input_l_major[input4] += grad1 * radial_value0;
                grad_radial[radial0] += grad0 * input0 + grad1 * input1;
            }
        }
        return;
    }
    for node in 0..nodes {
        let input_node_start = node * input_node_stride;
        let radial_node_start = node * radial_node_stride + radial_offset;
        let block_node_start = node * block_node_stride;
        for (source_offset, source_coefficient) in sources.iter().copied().enumerate() {
            let source_start = input_node_start + source_coefficient * channels;
            let block_start = block_node_start + source_offset * channels;
            let radial_start = radial_node_start + (source_offset % m_size) * channels;
            for channel in 0..channels {
                let grad = grad_modulated_block[block_start + channel];
                let input = input_l_major[source_start + channel];
                let radial_value = radial[radial_start + channel];
                grad_input_l_major[source_start + channel] += grad * radial_value;
                grad_radial[radial_start + channel] += grad * input;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_l2_m2_gate_activation_vjp_from_order_block(
    nodes: usize,
    channels: usize,
    sources: &[usize],
    grad_activated_block: &[f64],
    input_l_major: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_l_major: &mut [f64],
) {
    let input_node_stride = L2_M2_COEFFICIENT_COUNT * channels;
    let gate_node_stride = 2 * channels;
    let block_node_stride = sources.len() * channels;
    if sources == L2_M2_ORDER0_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let gate_node_start = node * gate_node_stride;
            let scalar_derivative_node_start = node * channels;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let scalar_index = input_node_start + channel;
                grad_input_l_major[scalar_index] += grad_activated_block
                    [block_node_start + channel]
                    * scalar_silu_derivatives[scalar_derivative_node_start + channel];

                let gate1_index = gate_node_start + channel;
                let gate1 = gate_sigmoids[gate1_index];
                let input2 = input_node_start + 2 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let grad1 = grad_activated_block[block1];
                grad_input_l_major[input2] += grad1 * gate1;
                grad_gating_scalars[gate1_index] += grad1 * input_l_major[input2];

                let gate2_index = gate_node_start + channels + channel;
                let gate2 = gate_sigmoids[gate2_index];
                let input6 = input_node_start + 6 * channels + channel;
                let block2 = block_node_start + 2 * channels + channel;
                let grad2 = grad_activated_block[block2];
                grad_input_l_major[input6] += grad2 * gate2;
                grad_gating_scalars[gate2_index] += grad2 * input_l_major[input6];
            }
        }
        return;
    }
    if sources == L2_M2_ORDER1_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate1_index = gate_node_start + channel;
                let gate2_index = gate_node_start + channels + channel;
                let gate1 = gate_sigmoids[gate1_index];
                let gate2 = gate_sigmoids[gate2_index];

                let input3 = input_node_start + 3 * channels + channel;
                let block0 = block_node_start + channel;
                let grad0 = grad_activated_block[block0];
                grad_input_l_major[input3] += grad0 * gate1;
                grad_gating_scalars[gate1_index] += grad0 * input_l_major[input3];

                let input7 = input_node_start + 7 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let grad1 = grad_activated_block[block1];
                grad_input_l_major[input7] += grad1 * gate2;
                grad_gating_scalars[gate2_index] += grad1 * input_l_major[input7];

                let input1 = input_node_start + channels + channel;
                let block2 = block_node_start + 2 * channels + channel;
                let grad2 = grad_activated_block[block2];
                grad_input_l_major[input1] += grad2 * gate1;
                grad_gating_scalars[gate1_index] += grad2 * input_l_major[input1];
                grad_gating_scalars[gate1_index] *= gate1 * (1.0 - gate1);

                let input5 = input_node_start + 5 * channels + channel;
                let block3 = block_node_start + 3 * channels + channel;
                let grad3 = grad_activated_block[block3];
                grad_input_l_major[input5] += grad3 * gate2;
                grad_gating_scalars[gate2_index] += grad3 * input_l_major[input5];
            }
        }
        return;
    }
    if sources == L2_M2_ORDER2_L_MAJOR_SOURCES {
        for node in 0..nodes {
            let input_node_start = node * input_node_stride;
            let gate_node_start = node * gate_node_stride;
            let block_node_start = node * block_node_stride;
            for channel in 0..channels {
                let gate2_index = gate_node_start + channels + channel;
                let gate2 = gate_sigmoids[gate2_index];

                let input8 = input_node_start + 8 * channels + channel;
                let block0 = block_node_start + channel;
                let grad0 = grad_activated_block[block0];
                grad_input_l_major[input8] += grad0 * gate2;
                grad_gating_scalars[gate2_index] += grad0 * input_l_major[input8];

                let input4 = input_node_start + 4 * channels + channel;
                let block1 = block_node_start + channels + channel;
                let grad1 = grad_activated_block[block1];
                grad_input_l_major[input4] += grad1 * gate2;
                grad_gating_scalars[gate2_index] += grad1 * input_l_major[input4];
                grad_gating_scalars[gate2_index] *= gate2 * (1.0 - gate2);
            }
        }
        return;
    }
    unreachable!("l=2,m=2 GateActivation VJP only accepts the three pinned order source blocks");
}

#[inline]
fn so2_gate_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[inline]
fn so2_silu_value_and_derivative(value: f64) -> (f64, f64) {
    let sig = so2_gate_sigmoid(value);
    (value * sig, sig * (1.0 + value * (1.0 - sig)))
}

#[allow(clippy::too_many_arguments)]
fn check_so2_convolution_assembly_shapes(
    op: &'static str,
    config: EsenSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: Option<&[f64]>,
    m_weights: &[&[f64]],
    output_like: &[f64],
    extra_m0_like: Option<&[f64]>,
) -> Result<usize> {
    let input_stride = config.input_node_stride();
    if !input_l_major.len().is_multiple_of(input_stride) {
        return Err(invalid_so2(format!(
            "{op} input length {} is not a multiple of node stride {input_stride}",
            input_l_major.len()
        )));
    }
    let nodes = input_l_major.len() / input_stride;
    check_len(op, nodes * config.output_node_stride(), output_like.len())?;
    check_len(
        "fairchem_esen_so2_convolution m0_weight",
        config.m0_weight_len(),
        m0_weight.len(),
    )?;
    if let Some(m0_bias_values) = m0_bias {
        check_len(
            "fairchem_esen_so2_convolution m0_bias",
            config.m0_bias_len(),
            m0_bias_values.len(),
        )?;
    }
    check_len(
        "fairchem_esen_so2_convolution m_weights",
        config.m_max,
        m_weights.len(),
    )?;
    for order in 1..=config.m_max {
        check_len(
            "fairchem_esen_so2_convolution m_weight",
            config.m_conv_weight_len(order),
            m_weights[order - 1].len(),
        )?;
    }
    if let Some(radial_values) = radial {
        check_len(
            "fairchem_esen_so2_convolution radial",
            nodes * config.radial_node_stride(),
            radial_values.len(),
        )?;
    }
    if config.extra_m0_output_channels > 0 {
        let Some(extra_values) = extra_m0_like else {
            return Err(invalid_so2(
                "extra m0 output or gradient is required by this SO2 convolution config",
            ));
        };
        check_len(
            "fairchem_esen_so2_convolution extra_m0",
            nodes * config.extra_m0_output_channels,
            extra_values.len(),
        )?;
    } else if let Some(extra_values) = extra_m0_like {
        check_len(
            "fairchem_esen_so2_convolution extra_m0",
            0,
            extra_values.len(),
        )?;
    }
    Ok(nodes)
}

fn check_so2_m_conv_block_shapes(
    op: &'static str,
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    weight: &[f64],
    out: &[f64],
) -> Result<usize> {
    check_len(
        "fairchem_esen_so2_m_conv_block weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_node_stride = config.input_node_stride();
    if !input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "{op} input length {} is not a multiple of node stride {input_node_stride}",
            input.len()
        )));
    }
    let nodes = input.len() / input_node_stride;
    check_len(op, nodes * config.output_node_stride(), out.len())?;
    Ok(nodes)
}

fn check_so2_radial_modulation_shapes(
    op: &'static str,
    config: EsenSo2RadialModulationConfig,
    input: &[f64],
    radial: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    check_len(op, input.len(), output_like.len())?;
    let input_node_stride = config.input_node_stride();
    if !input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "{op} input length {} is not a multiple of node stride {input_node_stride}",
            input.len()
        )));
    }
    let nodes = input.len() / input_node_stride;
    check_len(
        "fairchem_esen_so2_radial_modulation radial",
        nodes * config.feature_dim,
        radial.len(),
    )?;
    Ok(nodes)
}

fn check_so2_linear_block_shapes(
    op: &'static str,
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    out: &[f64],
) -> Result<usize> {
    check_len(
        "fairchem_esen_so2_linear_block weight",
        config.weight_len(),
        weight.len(),
    )?;
    if let Some(bias_values) = bias {
        if config.copies != 1 {
            return Err(invalid_so2(
                "SO2 m>0 real/imag blocks are bias-free in FairChem",
            ));
        }
        check_len(
            "fairchem_esen_so2_linear_block bias",
            config.output_copy_stride(),
            bias_values.len(),
        )?;
    }
    let input_node_stride = config.input_node_stride();
    if !input.len().is_multiple_of(input_node_stride) {
        return Err(invalid_so2(format!(
            "{op} input length {} is not a multiple of node stride {input_node_stride}",
            input.len()
        )));
    }
    let nodes = input.len() / input_node_stride;
    check_len(op, nodes * config.output_node_stride(), out.len())?;
    Ok(nodes)
}

fn check_len(op: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(CoreError::ShapeMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

fn invalid_so2(message: impl Into<String>) -> CoreError {
    CoreError::InvalidSo2(message.into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::activation::{
        fairchem_esen_gate_activation_forward_with_cache,
        fairchem_esen_gate_activation_vjp_with_cache, EsenGateActivationConfig,
    };

    fn patterned_values(len: usize, scale: f64, shift: f64) -> Vec<f64> {
        (0..len)
            .map(|index| {
                let base = (index % 23) as f64 - 11.0;
                base * scale + shift
            })
            .collect()
    }

    fn assert_close_slices(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
        let mut max_abs = 0.0_f64;
        let mut max_index = 0usize;
        for (index, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(actual_value.is_finite(), "{name}[{index}] is not finite");
            let abs = (actual_value - expected_value).abs();
            if abs > max_abs {
                max_abs = abs;
                max_index = index;
            }
        }
        assert!(
            max_abs <= tolerance,
            "{name} max abs {max_abs:e} at {max_index}, expected <= {tolerance:e}; actual={}, expected={}",
            actual[max_index],
            expected[max_index]
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn compact_input_parallel_and_serial_guards_are_complementary() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let threshold = so2_parallel_node_threshold();
            for nodes in [
                threshold,
                threshold.saturating_mul(2),
                threshold.saturating_mul(4),
            ] {
                assert!(
                    so2_use_parallel_compact_input(nodes),
                    "nodes={nodes} should use the lowered parallel compact-input path"
                );
                assert!(
                    !so2_use_serial_compact_input(nodes),
                    "nodes={nodes} must not also run the serial compact-input fill"
                );
            }
            for nodes in [0usize, threshold.saturating_sub(1)] {
                assert!(
                    !so2_use_parallel_compact_input(nodes),
                    "nodes={nodes} should remain below the parallel compact-input path"
                );
                assert!(
                    so2_use_serial_compact_input(nodes),
                    "nodes={nodes} should use exactly the serial compact-input fill"
                );
            }
        });
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn m_conv_block_with_radial_forward_lowered_parallel_band_matches_single_thread(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let single_thread = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        let multi_thread = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let config = EsenSo2MConvBlockConfig::new(4, 3, 2)?;
        let weight = patterned_values(config.weight_len(), 0.006, -0.014);

        for nodes in [64usize, 128, 256] {
            let input = patterned_values(nodes * config.input_node_stride(), 0.013, 0.021);
            let radial = patterned_values(nodes * config.input_copy_stride(), 0.017, 0.37);

            let run_forward = || -> Result<Vec<f64>> {
                let mut out = vec![f64::NAN; nodes * config.output_node_stride()];
                fairchem_esen_so2_m_conv_block_with_radial_forward(
                    config, &input, &radial, &weight, &mut out,
                )?;
                Ok(out)
            };

            let expected = single_thread.install(run_forward)?;
            let actual = multi_thread.install(run_forward)?;
            assert_close_slices(
                &format!("m-conv radial nodes={nodes}"),
                &actual,
                &expected,
                1.0e-14,
            );
        }
        Ok(())
    }

    #[test]
    fn l2_m2_radial_gate_conv2_fused_forward_matches_generic_replay() {
        let nodes = 2usize;
        let in_channels = 3usize;
        let channels = 2usize;

        let conv1_config =
            EsenSo2ConvolutionConfig::new(2, 2, in_channels, channels, 2 * channels).unwrap();
        let conv2_config = EsenSo2ConvolutionConfig::new(2, 2, channels, channels, 0).unwrap();

        let input_l_major =
            patterned_values(nodes * conv1_config.input_node_stride(), 0.017, -0.03);
        let radial = patterned_values(nodes * conv1_config.radial_node_stride(), 0.011, 0.08);

        let conv1_m0_weight = patterned_values(conv1_config.m0_weight_len(), 0.013, 0.01);
        let conv1_m0_bias = patterned_values(conv1_config.m0_bias_len(), 0.019, -0.02);
        let conv1_m1_weight = patterned_values(conv1_config.m_conv_weight_len(1), 0.007, 0.015);
        let conv1_m2_weight = patterned_values(conv1_config.m_conv_weight_len(2), 0.009, -0.01);
        let conv1_m_weights: [&[f64]; 2] = [&conv1_m1_weight, &conv1_m2_weight];

        let conv2_m0_weight = patterned_values(conv2_config.m0_weight_len(), 0.012, -0.005);
        let conv2_m0_bias = patterned_values(conv2_config.m0_bias_len(), 0.016, 0.02);
        let conv2_m1_weight = patterned_values(conv2_config.m_conv_weight_len(1), 0.006, -0.01);
        let conv2_m2_weight = patterned_values(conv2_config.m_conv_weight_len(2), 0.008, 0.01);
        let conv2_m_weights: [&[f64]; 2] = [&conv2_m1_weight, &conv2_m2_weight];

        let mut conv1_generic = vec![0.0; nodes * conv1_config.output_node_stride()];
        let mut gate_scalars_generic = vec![0.0; nodes * 2 * channels];
        fairchem_esen_so2_convolution_forward(
            conv1_config,
            &input_l_major,
            Some(&radial),
            &conv1_m0_weight,
            &conv1_m0_bias,
            &conv1_m_weights,
            &mut conv1_generic,
            Some(&mut gate_scalars_generic),
        )
        .unwrap();

        let gate_config = EsenGateActivationConfig::new(2, 2, channels).unwrap();
        let mut activated_generic = vec![0.0; conv1_generic.len()];
        let mut gate_sigmoids_generic = vec![0.0; gate_scalars_generic.len()];
        let mut scalar_silu_derivatives_generic = vec![0.0; nodes * channels];
        fairchem_esen_gate_activation_forward_with_cache(
            gate_config,
            &gate_scalars_generic,
            &conv1_generic,
            &mut activated_generic,
            &mut gate_sigmoids_generic,
            &mut scalar_silu_derivatives_generic,
        )
        .unwrap();

        let mut conv2_generic = vec![0.0; nodes * conv2_config.output_node_stride()];
        fairchem_esen_so2_convolution_forward(
            conv2_config,
            &activated_generic,
            None,
            &conv2_m0_weight,
            &conv2_m0_bias,
            &conv2_m_weights,
            &mut conv2_generic,
            None,
        )
        .unwrap();

        let mut conv1_fused = vec![f64::NAN; conv1_generic.len()];
        let mut gate_sigmoids_fused = vec![f64::NAN; gate_sigmoids_generic.len()];
        let mut scalar_silu_derivatives_fused =
            vec![f64::NAN; scalar_silu_derivatives_generic.len()];
        let mut conv2_fused = vec![f64::NAN; conv2_generic.len()];
        fairchem_esen_so2_convolution_forward_l2_m2_radial_gate_conv2_with_fused_m_weights(
            conv1_config,
            &input_l_major,
            &radial,
            &conv1_m0_weight,
            &conv1_m0_bias,
            &conv1_m_weights,
            None,
            conv2_config,
            &conv2_m0_weight,
            &conv2_m0_bias,
            &conv2_m_weights,
            None,
            &mut conv1_fused,
            &mut gate_sigmoids_fused,
            &mut scalar_silu_derivatives_fused,
            &mut conv2_fused,
        )
        .unwrap();

        assert_close_slices("conv1 replay", &conv1_fused, &conv1_generic, 1.0e-14);
        assert_close_slices(
            "gate sigmoids",
            &gate_sigmoids_fused,
            &gate_sigmoids_generic,
            1.0e-14,
        );
        assert_close_slices(
            "scalar silu derivatives",
            &scalar_silu_derivatives_fused,
            &scalar_silu_derivatives_generic,
            1.0e-14,
        );
        assert_close_slices("conv2 output", &conv2_fused, &conv2_generic, 1.0e-14);

        let grad_conv2_out = patterned_values(conv2_generic.len(), 0.014, -0.006);
        let mut grad_activated_generic = vec![0.0; activated_generic.len()];
        fairchem_esen_so2_convolution_vjp(
            conv2_config,
            &activated_generic,
            None,
            &conv2_m0_weight,
            &conv2_m_weights,
            &grad_conv2_out,
            None,
            &mut grad_activated_generic,
            None,
        )
        .unwrap();

        let mut grad_gate_scalars_generic = vec![0.0; gate_scalars_generic.len()];
        let mut grad_conv1_out_generic = vec![0.0; conv1_generic.len()];
        fairchem_esen_gate_activation_vjp_with_cache(
            gate_config,
            &gate_scalars_generic,
            &conv1_generic,
            &gate_sigmoids_generic,
            &scalar_silu_derivatives_generic,
            &grad_activated_generic,
            &mut grad_gate_scalars_generic,
            &mut grad_conv1_out_generic,
        )
        .unwrap();

        let mut grad_input_generic = vec![0.0; input_l_major.len()];
        let mut grad_radial_generic = vec![0.0; radial.len()];
        fairchem_esen_so2_convolution_vjp(
            conv1_config,
            &input_l_major,
            Some(&radial),
            &conv1_m0_weight,
            &conv1_m_weights,
            &grad_conv1_out_generic,
            Some(&grad_gate_scalars_generic),
            &mut grad_input_generic,
            Some(&mut grad_radial_generic),
        )
        .unwrap();

        let mut grad_input_fused = vec![0.0; input_l_major.len()];
        let mut grad_radial_fused = vec![0.0; radial.len()];
        fairchem_esen_so2_convolution_vjp_l2_m2_radial_gate_conv2_with_fused_m_weights(
            conv2_config,
            &conv2_m0_weight,
            &conv2_m_weights,
            None,
            &grad_conv2_out,
            &conv1_fused,
            &gate_sigmoids_fused,
            &scalar_silu_derivatives_fused,
            conv1_config,
            &input_l_major,
            &radial,
            &conv1_m0_weight,
            &conv1_m_weights,
            None,
            &mut grad_input_fused,
            &mut grad_radial_fused,
        )
        .unwrap();

        assert_close_slices(
            "conv1 input VJP",
            &grad_input_fused,
            &grad_input_generic,
            1.0e-12,
        );
        assert_close_slices(
            "conv1 radial VJP",
            &grad_radial_fused,
            &grad_radial_generic,
            1.0e-12,
        );
    }
}
