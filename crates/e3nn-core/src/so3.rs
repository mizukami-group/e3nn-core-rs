//! SO3 feature primitives used by fixed-weight equivariant models.

use crate::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const SO3_F32_PARALLEL_ITEMS_MIN: usize = 128;
#[cfg(feature = "parallel")]
const SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS: usize = 128;

#[cfg(e3nn_core_cblas_backend)]
#[allow(clippy::too_many_arguments, unsafe_code)]
mod accelerate_blas {
    use std::os::raw::{c_double, c_int};

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

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

    pub fn matmul_nn(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, out: &mut [f64]) {
        // Shape guard for the raw cblas_dgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_nn a is [m, k]");
        debug_assert_eq!(b.len(), k * n, "matmul_nn b is [k, n]");
        debug_assert_eq!(out.len(), m * n, "matmul_nn out is [m, n]");
        unsafe {
            // Invariant: computes a [m, k] * b [k, n] into out [m, n].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_tn_add(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, out: &mut [f64]) {
        // Shape guard for the raw cblas_dgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_tn_add a is [m, k] (a.T is [k, m])");
        debug_assert_eq!(b.len(), m * n, "matmul_tn_add b is [m, n]");
        debug_assert_eq!(out.len(), k * n, "matmul_tn_add out is [k, n]");
        unsafe {
            // Invariant: accumulates a.T [k, m] * b [m, n] into out [k, n].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                k as c_int,
                n as c_int,
                m as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                1.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_tn_write(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, out: &mut [f64]) {
        // Shape guard for the raw cblas_dgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(
            a.len(),
            m * k,
            "matmul_tn_write a is [m, k] (a.T is [k, m])"
        );
        debug_assert_eq!(b.len(), m * n, "matmul_tn_write b is [m, n]");
        debug_assert_eq!(out.len(), k * n, "matmul_tn_write out is [k, n]");
        unsafe {
            // Invariant: computes a.T [k, m] * b [m, n] into fresh out [k, n].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                k as c_int,
                n as c_int,
                m as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_nt_add(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, out: &mut [f64]) {
        // Shape guard for the raw cblas_dgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_nt_add a is [m, k]");
        debug_assert_eq!(b.len(), n * k, "matmul_nt_add b is [n, k] (b.T is [k, n])");
        debug_assert_eq!(out.len(), m * n, "matmul_nt_add out is [m, n]");
        unsafe {
            // Invariant: accumulates a [m, k] * b.T [k, n] into out [m, n].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                k as c_int,
                1.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_nt_write(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, out: &mut [f64]) {
        // Shape guard for the raw cblas_dgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_nt_write a is [m, k]");
        debug_assert_eq!(
            b.len(),
            n * k,
            "matmul_nt_write b is [n, k] (b.T is [k, n])"
        );
        debug_assert_eq!(out.len(), m * n, "matmul_nt_write out is [m, n]");
        unsafe {
            // Invariant: computes a [m, k] * b.T [k, n] into fresh out [m, n].
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                k as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo3LinearConfig {
    l_max: usize,
    in_channels: usize,
    out_channels: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenCoefficientMappingConfig {
    l_max: usize,
    m_max: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo3GridConfig {
    coefficient_count: usize,
    channels: usize,
    lat_resolution: usize,
    long_resolution: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSo3BatchedMatmulConfig {
    input_coefficients: usize,
    output_coefficients: usize,
    channels: usize,
}

impl EsenSo3BatchedMatmulConfig {
    pub fn new(
        input_coefficients: usize,
        output_coefficients: usize,
        channels: usize,
    ) -> Result<Self> {
        if input_coefficients == 0 || output_coefficients == 0 || channels == 0 {
            return Err(invalid_so3(format!(
                "batched SO3 dimensions must be nonzero, got {input_coefficients}->{output_coefficients} with {channels} channels"
            )));
        }
        input_coefficients
            .checked_mul(output_coefficients)
            .and_then(|_| input_coefficients.checked_mul(channels))
            .and_then(|_| output_coefficients.checked_mul(channels))
            .ok_or_else(|| invalid_so3("batched SO3 matmul dimensions overflow"))?;
        Ok(Self {
            input_coefficients,
            output_coefficients,
            channels,
        })
    }

    #[must_use]
    pub fn input_coefficients(&self) -> usize {
        self.input_coefficients
    }

    #[must_use]
    pub fn output_coefficients(&self) -> usize {
        self.output_coefficients
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub fn input_item_stride(&self) -> usize {
        self.input_coefficients * self.channels
    }

    #[must_use]
    pub fn output_item_stride(&self) -> usize {
        self.output_coefficients * self.channels
    }

    #[must_use]
    pub fn matrix_item_stride(&self) -> usize {
        self.output_coefficients * self.input_coefficients
    }
}

impl EsenSo3GridConfig {
    pub fn new(
        coefficient_count: usize,
        channels: usize,
        lat_resolution: usize,
        long_resolution: usize,
    ) -> Result<Self> {
        if coefficient_count == 0 || channels == 0 {
            return Err(invalid_so3(format!(
                "coefficient_count and channels must be nonzero, got {coefficient_count}, {channels}"
            )));
        }
        if lat_resolution == 0 || long_resolution == 0 {
            return Err(invalid_so3(format!(
                "grid resolutions must be nonzero, got {lat_resolution} x {long_resolution}"
            )));
        }
        lat_resolution
            .checked_mul(long_resolution)
            .and_then(|grid_points| grid_points.checked_mul(coefficient_count))
            .and_then(|_| coefficient_count.checked_mul(channels))
            .ok_or_else(|| invalid_so3("SO3 grid dimensions overflow"))?;
        Ok(Self {
            coefficient_count,
            channels,
            lat_resolution,
            long_resolution,
        })
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub fn lat_resolution(&self) -> usize {
        self.lat_resolution
    }

    #[must_use]
    pub fn long_resolution(&self) -> usize {
        self.long_resolution
    }

    #[must_use]
    pub fn grid_points(&self) -> usize {
        self.lat_resolution * self.long_resolution
    }

    #[must_use]
    pub fn coefficient_node_stride(&self) -> usize {
        self.coefficient_count * self.channels
    }

    #[must_use]
    pub fn grid_node_stride(&self) -> usize {
        self.grid_points() * self.channels
    }

    #[must_use]
    pub fn matrix_len(&self) -> usize {
        self.grid_points() * self.coefficient_count
    }
}

impl EsenCoefficientMappingConfig {
    pub fn new(l_max: usize, m_max: usize) -> Result<Self> {
        if m_max > l_max {
            return Err(invalid_so3(format!(
                "m_max must be <= l_max, got {m_max} > {l_max}"
            )));
        }
        coefficient_count(l_max, m_max).ok_or_else(|| {
            invalid_so3(format!(
                "coefficient count overflows for l_max={l_max}, m_max={m_max}"
            ))
        })?;
        Ok(Self { l_max, m_max })
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
    pub fn coefficient_count(&self) -> usize {
        (0..=self.l_max)
            .map(|degree| 2 * self.m_max.min(degree) + 1)
            .sum()
    }

    #[must_use]
    pub fn m_size(&self, order: usize) -> Option<usize> {
        if order > self.m_max {
            None
        } else {
            Some(self.l_max - order + 1)
        }
    }
}

impl EsenSo3LinearConfig {
    pub fn new(l_max: usize, in_channels: usize, out_channels: usize) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(invalid_so3(format!(
                "channel dimensions must be nonzero, got {in_channels} -> {out_channels}"
            )));
        }
        l_max
            .checked_add(1)
            .and_then(|degree_count| degree_count.checked_mul(degree_count))
            .ok_or_else(|| invalid_so3(format!("l_max is too large: {l_max}")))?;
        Ok(Self {
            l_max,
            in_channels,
            out_channels,
        })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
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
    pub fn degree_count(&self) -> usize {
        self.l_max + 1
    }

    #[must_use]
    pub fn basis_dim(&self) -> usize {
        self.degree_count() * self.degree_count()
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.basis_dim() * self.in_channels
    }

    #[must_use]
    pub fn output_node_stride(&self) -> usize {
        self.basis_dim() * self.out_channels
    }

    #[must_use]
    pub fn weight_len(&self) -> usize {
        self.degree_count() * self.out_channels * self.in_channels
    }
}

/// FairChem eSEN l-major to m-major coefficient permutation.
///
/// The returned vector has length `coefficient_count`; entry `out_index`
/// stores the source l-major coefficient index for that m-major output row.
#[must_use]
pub fn fairchem_esen_l_to_m_permutation(config: EsenCoefficientMappingConfig) -> Vec<usize> {
    let mut permutation = Vec::with_capacity(config.coefficient_count());
    for order in 0..=config.m_max {
        for degree in order..=config.l_max {
            permutation.push(l_major_index(config, degree, order as isize));
        }
        if order != 0 {
            for degree in order..=config.l_max {
                permutation.push(l_major_index(config, degree, -(order as isize)));
            }
        }
    }
    permutation
}

/// Apply FairChem eSEN l-major to m-major coefficient remapping to flat
/// `[nodes, coefficients, channels]` tensors.
pub fn fairchem_esen_l_to_m_forward(
    config: EsenCoefficientMappingConfig,
    channels: usize,
    input: &[f64],
    out: &mut [f64],
) -> Result<()> {
    apply_coefficient_permutation(
        "fairchem_esen_l_to_m_forward",
        config,
        channels,
        input,
        out,
        true,
    )
}

/// Apply the inverse m-major to l-major coefficient remapping to flat
/// `[nodes, coefficients, channels]` tensors.
pub fn fairchem_esen_m_to_l_forward(
    config: EsenCoefficientMappingConfig,
    channels: usize,
    input: &[f64],
    out: &mut [f64],
) -> Result<()> {
    apply_coefficient_permutation(
        "fairchem_esen_m_to_l_forward",
        config,
        channels,
        input,
        out,
        false,
    )
}

/// FairChem eSEN `SO3_Linear` forward pass.
///
/// `input` is flat `[nodes, (lmax + 1)^2, in_channels]`, `weight` is flat
/// `[lmax + 1, out_channels, in_channels]`, `bias` is `[out_channels]`, and
/// `out` is flat `[nodes, (lmax + 1)^2, out_channels]`. The bias is applied
/// only to the scalar `m = 0` component.
pub fn fairchem_esen_so3_linear_forward(
    config: EsenSo3LinearConfig,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so3_linear_shapes(
        "fairchem_esen_so3_linear_forward",
        config,
        input,
        weight,
        bias,
        out,
    )?;

    #[cfg(e3nn_core_cblas_backend)]
    {
        so3_linear_forward_blas(config, nodes, input, weight, bias, out);
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for node_index in 0..nodes {
            let input_node_start = node_index * config.input_node_stride();
            let out_node_start = node_index * config.output_node_stride();
            for m_index in 0..config.basis_dim() {
                let degree = degree_for_component(m_index);
                for out_channel in 0..config.out_channels {
                    let mut value = if m_index == 0 { bias[out_channel] } else { 0.0 };
                    let weight_row_start = weight_index(config, degree, out_channel, 0);
                    let input_row_start = input_node_start + m_index * config.in_channels;
                    for in_channel in 0..config.in_channels {
                        value += weight[weight_row_start + in_channel]
                            * input[input_row_start + in_channel];
                    }
                    out[out_node_start + m_index * config.out_channels + out_channel] = value;
                }
            }
        }
    }
    Ok(())
}

/// Accumulate the input VJP of [`fairchem_esen_so3_linear_forward`] into
/// `grad_input`.
pub fn fairchem_esen_so3_linear_vjp(
    config: EsenSo3LinearConfig,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_so3_linear_vjp weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_stride = config.input_node_stride();
    let output_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_linear_vjp grad_input length {} is not a multiple of node stride {input_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_linear_vjp grad_out",
        nodes * output_stride,
        grad_out.len(),
    )?;

    #[cfg(e3nn_core_cblas_backend)]
    {
        so3_linear_vjp_blas(config, nodes, weight, grad_out, grad_input);
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for node_index in 0..nodes {
            let grad_input_node_start = node_index * input_stride;
            let grad_out_node_start = node_index * output_stride;
            for m_index in 0..config.basis_dim() {
                let degree = degree_for_component(m_index);
                for out_channel in 0..config.out_channels {
                    let grad =
                        grad_out[grad_out_node_start + m_index * config.out_channels + out_channel];
                    let weight_row_start = weight_index(config, degree, out_channel, 0);
                    let grad_input_row_start = grad_input_node_start + m_index * config.in_channels;
                    for in_channel in 0..config.in_channels {
                        grad_input[grad_input_row_start + in_channel] +=
                            weight[weight_row_start + in_channel] * grad;
                    }
                }
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_so3_linear_forward`].
pub fn fairchem_esen_so3_linear_forward_f32(
    config: EsenSo3LinearConfig,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) -> Result<()> {
    check_len(
        "fairchem_esen_so3_linear_forward_f32 weight",
        config.weight_len(),
        weight.len(),
    )?;
    check_len(
        "fairchem_esen_so3_linear_forward_f32 bias",
        config.out_channels,
        bias.len(),
    )?;
    let input_stride = config.input_node_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_linear_forward_f32 input length {} is not a multiple of node stride {input_stride}",
            input.len()
        )));
    }
    let nodes = input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_linear_forward_f32",
        nodes * config.output_node_stride(),
        out.len(),
    )?;

    #[cfg(e3nn_core_cblas_backend)]
    {
        so3_linear_forward_blas_f32(config, nodes, input, weight, bias, out);
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for node_index in 0..nodes {
            let input_node_start = node_index * config.input_node_stride();
            let out_node_start = node_index * config.output_node_stride();
            for m_index in 0..config.basis_dim() {
                let degree = degree_for_component(m_index);
                for out_channel in 0..config.out_channels {
                    let mut value = if m_index == 0 {
                        bias[out_channel]
                    } else {
                        0.0_f32
                    };
                    let weight_row_start = weight_index(config, degree, out_channel, 0);
                    let input_row_start = input_node_start + m_index * config.in_channels;
                    for in_channel in 0..config.in_channels {
                        value += weight[weight_row_start + in_channel]
                            * input[input_row_start + in_channel];
                    }
                    out[out_node_start + m_index * config.out_channels + out_channel] = value;
                }
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_so3_linear_vjp`].
///
/// Accumulates the input VJP of [`fairchem_esen_so3_linear_forward_f32`] into `grad_input`.
pub fn fairchem_esen_so3_linear_vjp_f32(
    config: EsenSo3LinearConfig,
    weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    check_len(
        "fairchem_esen_so3_linear_vjp_f32 weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_stride = config.input_node_stride();
    let output_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_linear_vjp_f32 grad_input length {} is not a multiple of node stride {input_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_linear_vjp_f32 grad_out",
        nodes * output_stride,
        grad_out.len(),
    )?;

    #[cfg(e3nn_core_cblas_backend)]
    {
        so3_linear_vjp_blas_f32(config, nodes, weight, grad_out, grad_input);
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for node_index in 0..nodes {
            let grad_input_node_start = node_index * input_stride;
            let grad_out_node_start = node_index * output_stride;
            for m_index in 0..config.basis_dim() {
                let degree = degree_for_component(m_index);
                for out_channel in 0..config.out_channels {
                    let grad =
                        grad_out[grad_out_node_start + m_index * config.out_channels + out_channel];
                    let weight_row_start = weight_index(config, degree, out_channel, 0);
                    let grad_input_row_start = grad_input_node_start + m_index * config.in_channels;
                    for in_channel in 0..config.in_channels {
                        grad_input[grad_input_row_start + in_channel] +=
                            weight[weight_row_start + in_channel] * grad;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Write-only f32 input VJP of [`fairchem_esen_so3_linear_forward_f32`].
///
/// This is equivalent to calling [`fairchem_esen_so3_linear_vjp_f32`] with a
/// zeroed `grad_input`, but every input component is overwritten exactly once.
pub fn fairchem_esen_so3_linear_vjp_write_f32(
    config: EsenSo3LinearConfig,
    weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    check_len(
        "fairchem_esen_so3_linear_vjp_write_f32 weight",
        config.weight_len(),
        weight.len(),
    )?;
    let input_stride = config.input_node_stride();
    let output_stride = config.output_node_stride();
    if !grad_input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_linear_vjp_write_f32 grad_input length {} is not a multiple of node stride {input_stride}",
            grad_input.len()
        )));
    }
    let nodes = grad_input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_linear_vjp_write_f32 grad_out",
        nodes * output_stride,
        grad_out.len(),
    )?;

    #[cfg(e3nn_core_cblas_backend)]
    {
        so3_linear_vjp_blas_write_f32(config, nodes, weight, grad_out, grad_input);
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for node_index in 0..nodes {
            let grad_input_node_start = node_index * input_stride;
            let grad_out_node_start = node_index * output_stride;
            for m_index in 0..config.basis_dim() {
                let degree = degree_for_component(m_index);
                let grad_out_row_start = grad_out_node_start + m_index * config.out_channels;
                let grad_input_row_start = grad_input_node_start + m_index * config.in_channels;
                for in_channel in 0..config.in_channels {
                    let mut value = 0.0_f32;
                    for out_channel in 0..config.out_channels {
                        let weight_row_start = weight_index(config, degree, out_channel, 0);
                        value += weight[weight_row_start + in_channel]
                            * grad_out[grad_out_row_start + out_channel];
                    }
                    grad_input[grad_input_row_start + in_channel] = value;
                }
            }
        }
    }
    Ok(())
}

/// FairChem eSEN `SO3_Grid.to_grid` projection.
///
/// `to_grid_mat` is flat `[lat_resolution, long_resolution, coefficients]`.
/// `input` is flat `[nodes, coefficients, channels]`, and `out_grid` is flat
/// `[nodes, lat_resolution, long_resolution, channels]`.
pub fn fairchem_esen_so3_to_grid_forward(
    config: EsenSo3GridConfig,
    input: &[f64],
    to_grid_mat: &[f64],
    out_grid: &mut [f64],
) -> Result<()> {
    let nodes = check_so3_to_grid_shapes(
        "fairchem_esen_so3_to_grid_forward",
        config,
        input,
        to_grid_mat,
        out_grid,
    )?;
    project_coefficients_to_grid(config, nodes, input, to_grid_mat, out_grid);
    Ok(())
}

/// Accumulate input VJP of [`fairchem_esen_so3_to_grid_forward`].
pub fn fairchem_esen_so3_to_grid_vjp(
    config: EsenSo3GridConfig,
    to_grid_mat: &[f64],
    grad_grid: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    let nodes = check_so3_grid_vjp_shapes(
        "fairchem_esen_so3_to_grid_vjp",
        config,
        to_grid_mat,
        grad_grid,
        grad_input,
    )?;
    project_grid_to_coefficients_add(config, nodes, grad_grid, to_grid_mat, grad_input);
    Ok(())
}

/// FairChem eSEN `SO3_Grid.from_grid` projection.
///
/// `from_grid_mat` is flat `[lat_resolution, long_resolution, coefficients]`.
/// `grid` is flat `[nodes, lat_resolution, long_resolution, channels]`, and
/// `out` is flat `[nodes, coefficients, channels]`.
pub fn fairchem_esen_so3_from_grid_forward(
    config: EsenSo3GridConfig,
    grid: &[f64],
    from_grid_mat: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_so3_grid_vjp_shapes(
        "fairchem_esen_so3_from_grid_forward",
        config,
        from_grid_mat,
        grid,
        out,
    )?;
    project_grid_to_coefficients(config, nodes, grid, from_grid_mat, out);
    Ok(())
}

/// Accumulate grid VJP of [`fairchem_esen_so3_from_grid_forward`].
pub fn fairchem_esen_so3_from_grid_vjp(
    config: EsenSo3GridConfig,
    from_grid_mat: &[f64],
    grad_out: &[f64],
    grad_grid: &mut [f64],
) -> Result<()> {
    let nodes = check_so3_to_grid_shapes(
        "fairchem_esen_so3_from_grid_vjp",
        config,
        grad_out,
        from_grid_mat,
        grad_grid,
    )?;
    project_coefficients_to_grid_add(config, nodes, grad_out, from_grid_mat, grad_grid);
    Ok(())
}

/// FairChem eSEN batched coefficient matrix application used by Wigner
/// rotation boundaries.
///
/// `matrix` is flat `[items, output_coefficients, input_coefficients]`,
/// `input` is flat `[items, input_coefficients, channels]`, and `out` is flat
/// `[items, output_coefficients, channels]`.
pub fn fairchem_esen_so3_batched_matmul_forward(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    input: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let items = check_so3_batched_matmul_shapes(
        "fairchem_esen_so3_batched_matmul_forward",
        config,
        matrix,
        input,
        out,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        let ch = config.channels;
        out.par_chunks_mut(config.output_item_stride())
            .enumerate()
            .for_each(|(item, out_item)| {
                let matrix_item =
                    &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
                let input_item =
                    &input[item * config.input_item_stride()..][..config.input_item_stride()];
                for out_coefficient in 0..config.output_coefficients {
                    let out_row = &mut out_item[out_coefficient * ch..out_coefficient * ch + ch];
                    // Write first in_coeff (avoids fill(0) + accumulate in separate passes)
                    let w0 = matrix_item[out_coefficient * config.input_coefficients];
                    let in0 = &input_item[0..ch];
                    for (o, &i) in out_row.iter_mut().zip(in0.iter()) {
                        *o = w0 * i;
                    }
                    // Accumulate remaining in_coefficients
                    for in_coefficient in 1..config.input_coefficients {
                        let weight = matrix_item
                            [out_coefficient * config.input_coefficients + in_coefficient];
                        let in_row = &input_item[in_coefficient * ch..in_coefficient * ch + ch];
                        for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
                            *o += weight * i;
                        }
                    }
                }
            });
        return Ok(());
    }

    #[cfg(e3nn_core_cblas_backend)]
    {
        for item in 0..items {
            let matrix_item_start = item * config.matrix_item_stride();
            let input_item_start = item * config.input_item_stride();
            let out_item_start = item * config.output_item_stride();
            accelerate_blas::matmul_nn(
                &matrix[matrix_item_start..matrix_item_start + config.matrix_item_stride()],
                config.output_coefficients,
                config.input_coefficients,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.channels,
                &mut out[out_item_start..out_item_start + config.output_item_stride()],
            );
        }
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        let ch = config.channels;
        for item in 0..items {
            let matrix_item =
                &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
            let input_item =
                &input[item * config.input_item_stride()..][..config.input_item_stride()];
            let out_item =
                &mut out[item * config.output_item_stride()..][..config.output_item_stride()];
            for out_coefficient in 0..config.output_coefficients {
                let out_row = &mut out_item[out_coefficient * ch..out_coefficient * ch + ch];
                // Write first in_coeff (avoids fill(0) + accumulate in separate passes)
                let w0 = matrix_item[out_coefficient * config.input_coefficients];
                let in0 = &input_item[0..ch];
                for (o, &i) in out_row.iter_mut().zip(in0.iter()) {
                    *o = w0 * i;
                }
                // Accumulate remaining in_coefficients
                for in_coefficient in 1..config.input_coefficients {
                    let weight =
                        matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                    let in_row = &input_item[in_coefficient * ch..in_coefficient * ch + ch];
                    for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
                        *o += weight * i;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Accumulate input VJP of [`fairchem_esen_so3_batched_matmul_forward`].
pub fn fairchem_esen_so3_batched_matmul_vjp_input(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    let items = check_so3_batched_matmul_vjp_shapes(
        "fairchem_esen_so3_batched_matmul_vjp_input",
        config,
        matrix,
        grad_out,
        grad_input,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        let ch = config.channels;
        grad_input
            .par_chunks_mut(config.input_item_stride())
            .enumerate()
            .for_each(|(item, grad_input_item)| {
                let matrix_item =
                    &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
                let grad_out_item =
                    &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
                for out_coefficient in 0..config.output_coefficients {
                    let grad_out_row =
                        &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
                    for in_coefficient in 0..config.input_coefficients {
                        let weight = matrix_item
                            [out_coefficient * config.input_coefficients + in_coefficient];
                        let grad_input_row =
                            &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                        for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                            *gi += weight * go;
                        }
                    }
                }
            });
        return Ok(());
    }

    #[cfg(e3nn_core_cblas_backend)]
    {
        for item in 0..items {
            let matrix_item_start = item * config.matrix_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_input_item_start = item * config.input_item_stride();
            accelerate_blas::matmul_tn_add(
                &matrix[matrix_item_start..matrix_item_start + config.matrix_item_stride()],
                config.output_coefficients,
                config.input_coefficients,
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.channels,
                &mut grad_input
                    [grad_input_item_start..grad_input_item_start + config.input_item_stride()],
            );
        }
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        let ch = config.channels;
        for item in 0..items {
            let matrix_item =
                &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
            let grad_out_item =
                &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
            let grad_input_item =
                &mut grad_input[item * config.input_item_stride()..][..config.input_item_stride()];
            for out_coefficient in 0..config.output_coefficients {
                let grad_out_row = &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
                for in_coefficient in 0..config.input_coefficients {
                    let weight =
                        matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                    let grad_input_row =
                        &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                    for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                        *gi += weight * go;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Compute input VJP of [`fairchem_esen_so3_batched_matmul_forward`] into a
/// fresh output buffer.
pub fn fairchem_esen_so3_batched_matmul_vjp_input_write(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    let _items = check_so3_batched_matmul_vjp_shapes(
        "fairchem_esen_so3_batched_matmul_vjp_input_write",
        config,
        matrix,
        grad_out,
        grad_input,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && _items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        let ch = config.channels;
        grad_input
            .par_chunks_mut(config.input_item_stride())
            .enumerate()
            .for_each(|(item, grad_input_item)| {
                let matrix_item =
                    &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
                let grad_out_item =
                    &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
                // Write-first: initialise with out_coeff=0, in_coeff=0..in_coefficients
                for in_coefficient in 0..config.input_coefficients {
                    let w0 = matrix_item[in_coefficient];
                    let grad_out_row0 = &grad_out_item[0..ch];
                    let grad_input_row =
                        &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                    for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row0.iter()) {
                        *gi = w0 * go;
                    }
                }
                // Accumulate remaining out_coefficients
                for out_coefficient in 1..config.output_coefficients {
                    let grad_out_row =
                        &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
                    for in_coefficient in 0..config.input_coefficients {
                        let weight = matrix_item
                            [out_coefficient * config.input_coefficients + in_coefficient];
                        let grad_input_row =
                            &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                        for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                            *gi += weight * go;
                        }
                    }
                }
            });
        return Ok(());
    }

    #[cfg(e3nn_core_cblas_backend)]
    {
        for item in 0.._items {
            let matrix_item_start = item * config.matrix_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_input_item_start = item * config.input_item_stride();
            accelerate_blas::matmul_tn_write(
                &matrix[matrix_item_start..matrix_item_start + config.matrix_item_stride()],
                config.output_coefficients,
                config.input_coefficients,
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.channels,
                &mut grad_input
                    [grad_input_item_start..grad_input_item_start + config.input_item_stride()],
            );
        }
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        let ch = config.channels;
        let items = _items;
        for item in 0..items {
            let matrix_item =
                &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
            let grad_out_item =
                &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
            let grad_input_item =
                &mut grad_input[item * config.input_item_stride()..][..config.input_item_stride()];
            // Write-first: initialise with out_coeff=0, in_coeff=0..in_coefficients
            for in_coefficient in 0..config.input_coefficients {
                let w0 = matrix_item[in_coefficient];
                let grad_out_row0 = &grad_out_item[0..ch];
                let grad_input_row =
                    &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row0.iter()) {
                    *gi = w0 * go;
                }
            }
            // Accumulate remaining out_coefficients
            for out_coefficient in 1..config.output_coefficients {
                let grad_out_row = &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
                for in_coefficient in 0..config.input_coefficients {
                    let weight =
                        matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                    let grad_input_row =
                        &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                    for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                        *gi += weight * go;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Accumulate matrix VJP of [`fairchem_esen_so3_batched_matmul_forward`].
///
/// This is needed for conservative force propagation through geometry-derived
/// Wigner matrices. It does not compute model parameter gradients.
pub fn fairchem_esen_so3_batched_matmul_vjp_matrix(
    config: EsenSo3BatchedMatmulConfig,
    input: &[f64],
    grad_out: &[f64],
    grad_matrix: &mut [f64],
) -> Result<()> {
    let items = check_so3_batched_matmul_matrix_vjp_shapes(
        "fairchem_esen_so3_batched_matmul_vjp_matrix",
        config,
        input,
        grad_out,
        grad_matrix,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        grad_matrix
            .par_chunks_mut(config.matrix_item_stride())
            .enumerate()
            .for_each(|(item, grad_matrix_item)| {
                let input_item_start = item * config.input_item_stride();
                let grad_out_item_start = item * config.output_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let grad_out_row_start =
                        grad_out_item_start + out_coefficient * config.channels;
                    let grad_matrix_row_start = out_coefficient * config.input_coefficients;
                    for in_coefficient in 0..config.input_coefficients {
                        let input_row_start = input_item_start + in_coefficient * config.channels;
                        let mut value = 0.0;
                        for channel in 0..config.channels {
                            value += grad_out[grad_out_row_start + channel]
                                * input[input_row_start + channel];
                        }
                        grad_matrix_item[grad_matrix_row_start + in_coefficient] += value;
                    }
                }
            });
        return Ok(());
    }

    #[cfg(e3nn_core_cblas_backend)]
    {
        for item in 0..items {
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_matrix_item_start = item * config.matrix_item_stride();
            accelerate_blas::matmul_nt_add(
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.output_coefficients,
                config.channels,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.input_coefficients,
                &mut grad_matrix
                    [grad_matrix_item_start..grad_matrix_item_start + config.matrix_item_stride()],
            );
        }
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for item in 0..items {
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_matrix_item_start = item * config.matrix_item_stride();
            for out_coefficient in 0..config.output_coefficients {
                let grad_out_row_start = grad_out_item_start + out_coefficient * config.channels;
                let grad_matrix_row_start =
                    grad_matrix_item_start + out_coefficient * config.input_coefficients;
                for in_coefficient in 0..config.input_coefficients {
                    let input_row_start = input_item_start + in_coefficient * config.channels;
                    let mut value = 0.0;
                    for channel in 0..config.channels {
                        value += grad_out[grad_out_row_start + channel]
                            * input[input_row_start + channel];
                    }
                    grad_matrix[grad_matrix_row_start + in_coefficient] += value;
                }
            }
        }
    }
    Ok(())
}

fn apply_coefficient_permutation(
    op: &'static str,
    config: EsenCoefficientMappingConfig,
    channels: usize,
    input: &[f64],
    out: &mut [f64],
    l_to_m: bool,
) -> Result<()> {
    if channels == 0 {
        return Err(invalid_so3("channels must be nonzero"));
    }
    check_len(op, input.len(), out.len())?;
    let node_stride = config
        .coefficient_count()
        .checked_mul(channels)
        .ok_or_else(|| invalid_so3("coefficient remap node stride overflow"))?;
    if !input.len().is_multiple_of(node_stride) {
        return Err(invalid_so3(format!(
            "{op} input length {} is not a multiple of node stride {node_stride}",
            input.len()
        )));
    }
    let permutation = fairchem_esen_l_to_m_permutation(config);
    for node_start in (0..input.len()).step_by(node_stride) {
        for (m_major_index, l_major_index) in permutation.iter().copied().enumerate() {
            for channel in 0..channels {
                let m_offset = node_start + m_major_index * channels + channel;
                let l_offset = node_start + l_major_index * channels + channel;
                if l_to_m {
                    out[m_offset] = input[l_offset];
                } else {
                    out[l_offset] = input[m_offset];
                }
            }
        }
    }
    Ok(())
}

/// Compute matrix VJP of [`fairchem_esen_so3_batched_matmul_forward`] into a
/// fresh output buffer.
pub fn fairchem_esen_so3_batched_matmul_vjp_matrix_write(
    config: EsenSo3BatchedMatmulConfig,
    input: &[f64],
    grad_out: &[f64],
    grad_matrix: &mut [f64],
) -> Result<()> {
    let items = check_so3_batched_matmul_matrix_vjp_shapes(
        "fairchem_esen_so3_batched_matmul_vjp_matrix_write",
        config,
        input,
        grad_out,
        grad_matrix,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        grad_matrix.fill(0.0);
        return fairchem_esen_so3_batched_matmul_vjp_matrix(config, input, grad_out, grad_matrix);
    }

    #[cfg(e3nn_core_cblas_backend)]
    {
        for item in 0..items {
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_matrix_item_start = item * config.matrix_item_stride();
            accelerate_blas::matmul_nt_write(
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.output_coefficients,
                config.channels,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.input_coefficients,
                &mut grad_matrix
                    [grad_matrix_item_start..grad_matrix_item_start + config.matrix_item_stride()],
            );
        }
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    {
        for item in 0..items {
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_matrix_item_start = item * config.matrix_item_stride();
            for out_coefficient in 0..config.output_coefficients {
                let grad_out_row_start = grad_out_item_start + out_coefficient * config.channels;
                let grad_matrix_row_start =
                    grad_matrix_item_start + out_coefficient * config.input_coefficients;
                for in_coefficient in 0..config.input_coefficients {
                    let input_row_start = input_item_start + in_coefficient * config.channels;
                    let mut value = 0.0;
                    for channel in 0..config.channels {
                        value += grad_out[grad_out_row_start + channel]
                            * input[input_row_start + channel];
                    }
                    grad_matrix[grad_matrix_row_start + in_coefficient] = value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(e3nn_core_cblas_backend)]
fn so3_linear_forward_blas(
    config: EsenSo3LinearConfig,
    nodes: usize,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    out: &mut [f64],
) {
    for degree in 0..config.degree_count() {
        let component_start = degree * degree;
        let component_count = 2 * degree + 1;
        let rows = nodes * component_count;
        let mut degree_input = vec![0.0; rows * config.in_channels];
        for node in 0..nodes {
            let input_start =
                node * config.input_node_stride() + component_start * config.in_channels;
            let degree_start = node * component_count * config.in_channels;
            degree_input[degree_start..degree_start + component_count * config.in_channels]
                .copy_from_slice(
                    &input[input_start..input_start + component_count * config.in_channels],
                );
        }

        let mut degree_out = vec![0.0; rows * config.out_channels];
        let weight_start = weight_index(config, degree, 0, 0);
        accelerate_blas::dense_forward_transposed_weight(
            &degree_input,
            rows,
            config.in_channels,
            &weight[weight_start..weight_start + config.out_channels * config.in_channels],
            config.out_channels,
            &mut degree_out,
        );
        if degree == 0 {
            for row in degree_out.chunks_exact_mut(config.out_channels).take(rows) {
                for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                    *value += *bias_value;
                }
            }
        }

        for node in 0..nodes {
            let out_start =
                node * config.output_node_stride() + component_start * config.out_channels;
            let degree_start = node * component_count * config.out_channels;
            out[out_start..out_start + component_count * config.out_channels].copy_from_slice(
                &degree_out[degree_start..degree_start + component_count * config.out_channels],
            );
        }
    }
}

#[cfg(e3nn_core_cblas_backend)]
fn so3_linear_vjp_blas(
    config: EsenSo3LinearConfig,
    nodes: usize,
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    for degree in 0..config.degree_count() {
        let component_start = degree * degree;
        let component_count = 2 * degree + 1;
        let rows = nodes * component_count;
        let mut degree_grad_out = vec![0.0; rows * config.out_channels];
        for node in 0..nodes {
            let grad_out_start =
                node * config.output_node_stride() + component_start * config.out_channels;
            let degree_start = node * component_count * config.out_channels;
            degree_grad_out[degree_start..degree_start + component_count * config.out_channels]
                .copy_from_slice(
                    &grad_out
                        [grad_out_start..grad_out_start + component_count * config.out_channels],
                );
        }

        let mut degree_grad_input = vec![0.0; rows * config.in_channels];
        let weight_start = weight_index(config, degree, 0, 0);
        accelerate_blas::dense_vjp_input_add(
            &degree_grad_out,
            rows,
            config.out_channels,
            &weight[weight_start..weight_start + config.out_channels * config.in_channels],
            config.in_channels,
            &mut degree_grad_input,
        );

        for node in 0..nodes {
            let grad_input_start =
                node * config.input_node_stride() + component_start * config.in_channels;
            let degree_start = node * component_count * config.in_channels;
            for offset in 0..component_count * config.in_channels {
                grad_input[grad_input_start + offset] += degree_grad_input[degree_start + offset];
            }
        }
    }
}

#[cfg(e3nn_core_cblas_backend)]
fn so3_linear_forward_blas_f32(
    config: EsenSo3LinearConfig,
    nodes: usize,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    for degree in 0..config.degree_count() {
        let component_start = degree * degree;
        let component_count = 2 * degree + 1;
        let rows = nodes * component_count;
        let mut degree_input = vec![0.0_f32; rows * config.in_channels];
        for node in 0..nodes {
            let input_start =
                node * config.input_node_stride() + component_start * config.in_channels;
            let degree_start = node * component_count * config.in_channels;
            degree_input[degree_start..degree_start + component_count * config.in_channels]
                .copy_from_slice(
                    &input[input_start..input_start + component_count * config.in_channels],
                );
        }

        let mut degree_out = vec![0.0_f32; rows * config.out_channels];
        let weight_start = weight_index(config, degree, 0, 0);
        accelerate_blas_f32::dense_forward_transposed_weight(
            &degree_input,
            rows,
            config.in_channels,
            &weight[weight_start..weight_start + config.out_channels * config.in_channels],
            config.out_channels,
            &mut degree_out,
        );
        if degree == 0 {
            for row in degree_out.chunks_exact_mut(config.out_channels).take(rows) {
                for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                    *value += *bias_value;
                }
            }
        }

        for node in 0..nodes {
            let out_start =
                node * config.output_node_stride() + component_start * config.out_channels;
            let degree_start = node * component_count * config.out_channels;
            out[out_start..out_start + component_count * config.out_channels].copy_from_slice(
                &degree_out[degree_start..degree_start + component_count * config.out_channels],
            );
        }
    }
}

#[cfg(e3nn_core_cblas_backend)]
fn so3_linear_vjp_blas_f32(
    config: EsenSo3LinearConfig,
    nodes: usize,
    weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    for degree in 0..config.degree_count() {
        let component_start = degree * degree;
        let component_count = 2 * degree + 1;
        let rows = nodes * component_count;
        let mut degree_grad_out = vec![0.0_f32; rows * config.out_channels];
        for node in 0..nodes {
            let grad_out_start =
                node * config.output_node_stride() + component_start * config.out_channels;
            let degree_start = node * component_count * config.out_channels;
            degree_grad_out[degree_start..degree_start + component_count * config.out_channels]
                .copy_from_slice(
                    &grad_out
                        [grad_out_start..grad_out_start + component_count * config.out_channels],
                );
        }

        let mut degree_grad_input = vec![0.0_f32; rows * config.in_channels];
        let weight_start = weight_index(config, degree, 0, 0);
        accelerate_blas_f32::dense_vjp_input_add(
            &degree_grad_out,
            rows,
            config.out_channels,
            &weight[weight_start..weight_start + config.out_channels * config.in_channels],
            config.in_channels,
            &mut degree_grad_input,
        );

        for node in 0..nodes {
            let grad_input_start =
                node * config.input_node_stride() + component_start * config.in_channels;
            let degree_start = node * component_count * config.in_channels;
            for offset in 0..component_count * config.in_channels {
                grad_input[grad_input_start + offset] += degree_grad_input[degree_start + offset];
            }
        }
    }
}

#[cfg(e3nn_core_cblas_backend)]
fn so3_linear_vjp_blas_write_f32(
    config: EsenSo3LinearConfig,
    nodes: usize,
    weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    for degree in 0..config.degree_count() {
        let component_start = degree * degree;
        let component_count = 2 * degree + 1;
        let rows = nodes * component_count;
        let mut degree_grad_out = vec![0.0_f32; rows * config.out_channels];
        for node in 0..nodes {
            let grad_out_start =
                node * config.output_node_stride() + component_start * config.out_channels;
            let degree_start = node * component_count * config.out_channels;
            degree_grad_out[degree_start..degree_start + component_count * config.out_channels]
                .copy_from_slice(
                    &grad_out
                        [grad_out_start..grad_out_start + component_count * config.out_channels],
                );
        }

        let mut degree_grad_input = vec![0.0_f32; rows * config.in_channels];
        let weight_start = weight_index(config, degree, 0, 0);
        accelerate_blas_f32::dense_vjp_input_add(
            &degree_grad_out,
            rows,
            config.out_channels,
            &weight[weight_start..weight_start + config.out_channels * config.in_channels],
            config.in_channels,
            &mut degree_grad_input,
        );

        for node in 0..nodes {
            let grad_input_start =
                node * config.input_node_stride() + component_start * config.in_channels;
            let degree_start = node * component_count * config.in_channels;
            grad_input[grad_input_start..grad_input_start + component_count * config.in_channels]
                .copy_from_slice(
                    &degree_grad_input
                        [degree_start..degree_start + component_count * config.in_channels],
                );
        }
    }
}

fn project_coefficients_to_grid(
    config: EsenSo3GridConfig,
    nodes: usize,
    input: &[f64],
    matrix: &[f64],
    out_grid: &mut [f64],
) {
    for node in 0..nodes {
        let input_node_start = node * config.coefficient_node_stride();
        let grid_node_start = node * config.grid_node_stride();
        for grid_index in 0..config.grid_points() {
            let matrix_start = grid_index * config.coefficient_count;
            let grid_start = grid_node_start + grid_index * config.channels;
            for channel in 0..config.channels {
                let mut value = 0.0;
                for coefficient in 0..config.coefficient_count {
                    value += matrix[matrix_start + coefficient]
                        * input[input_node_start + coefficient * config.channels + channel];
                }
                out_grid[grid_start + channel] = value;
            }
        }
    }
}

fn project_coefficients_to_grid_add(
    config: EsenSo3GridConfig,
    nodes: usize,
    input: &[f64],
    matrix: &[f64],
    out_grid: &mut [f64],
) {
    for node in 0..nodes {
        let input_node_start = node * config.coefficient_node_stride();
        let grid_node_start = node * config.grid_node_stride();
        for grid_index in 0..config.grid_points() {
            let matrix_start = grid_index * config.coefficient_count;
            let grid_start = grid_node_start + grid_index * config.channels;
            for channel in 0..config.channels {
                let mut value = 0.0;
                for coefficient in 0..config.coefficient_count {
                    value += matrix[matrix_start + coefficient]
                        * input[input_node_start + coefficient * config.channels + channel];
                }
                out_grid[grid_start + channel] += value;
            }
        }
    }
}

fn project_grid_to_coefficients(
    config: EsenSo3GridConfig,
    nodes: usize,
    grid: &[f64],
    matrix: &[f64],
    out: &mut [f64],
) {
    for node in 0..nodes {
        let grid_node_start = node * config.grid_node_stride();
        let out_node_start = node * config.coefficient_node_stride();
        for coefficient in 0..config.coefficient_count {
            let out_coeff_start = out_node_start + coefficient * config.channels;
            for channel in 0..config.channels {
                out[out_coeff_start + channel] = 0.0;
            }
            for grid_index in 0..config.grid_points() {
                let matrix_value = matrix[grid_index * config.coefficient_count + coefficient];
                let grid_start = grid_node_start + grid_index * config.channels;
                for channel in 0..config.channels {
                    out[out_coeff_start + channel] += matrix_value * grid[grid_start + channel];
                }
            }
        }
    }
}

fn project_grid_to_coefficients_add(
    config: EsenSo3GridConfig,
    nodes: usize,
    grid: &[f64],
    matrix: &[f64],
    out: &mut [f64],
) {
    for node in 0..nodes {
        let grid_node_start = node * config.grid_node_stride();
        let out_node_start = node * config.coefficient_node_stride();
        for coefficient in 0..config.coefficient_count {
            for channel in 0..config.channels {
                let mut value = 0.0;
                for grid_index in 0..config.grid_points() {
                    value += matrix[grid_index * config.coefficient_count + coefficient]
                        * grid[grid_node_start + grid_index * config.channels + channel];
                }
                out[out_node_start + coefficient * config.channels + channel] += value;
            }
        }
    }
}

fn coefficient_count(l_max: usize, m_max: usize) -> Option<usize> {
    let mut count = 0usize;
    for degree in 0..=l_max {
        let order_count = 2usize.checked_mul(m_max.min(degree))?.checked_add(1)?;
        count = count.checked_add(order_count)?;
    }
    Some(count)
}

fn l_major_index(config: EsenCoefficientMappingConfig, degree: usize, order: isize) -> usize {
    let mut offset = 0usize;
    for previous_degree in 0..degree {
        offset += 2 * config.m_max.min(previous_degree) + 1;
    }
    let degree_m_max = config.m_max.min(degree) as isize;
    debug_assert!((-degree_m_max..=degree_m_max).contains(&order));
    offset + (order + degree_m_max) as usize
}

fn check_so3_linear_shapes(
    op: &'static str,
    config: EsenSo3LinearConfig,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    out: &[f64],
) -> Result<usize> {
    check_len(
        "fairchem_esen_so3_linear weight",
        config.weight_len(),
        weight.len(),
    )?;
    check_len(
        "fairchem_esen_so3_linear bias",
        config.out_channels,
        bias.len(),
    )?;
    let input_stride = config.input_node_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} input length {} is not a multiple of node stride {input_stride}",
            input.len()
        )));
    }
    let nodes = input.len() / input_stride;
    check_len(op, nodes * config.output_node_stride(), out.len())?;
    Ok(nodes)
}

fn check_so3_to_grid_shapes(
    op: &'static str,
    config: EsenSo3GridConfig,
    coefficient_like: &[f64],
    matrix: &[f64],
    grid_like: &[f64],
) -> Result<usize> {
    check_len(
        "fairchem_esen_so3_grid matrix",
        config.matrix_len(),
        matrix.len(),
    )?;
    let coefficient_stride = config.coefficient_node_stride();
    if !coefficient_like.len().is_multiple_of(coefficient_stride) {
        return Err(invalid_so3(format!(
            "{op} coefficient tensor length {} is not a multiple of node stride {coefficient_stride}",
            coefficient_like.len()
        )));
    }
    let nodes = coefficient_like.len() / coefficient_stride;
    check_len(op, nodes * config.grid_node_stride(), grid_like.len())?;
    Ok(nodes)
}

fn check_so3_grid_vjp_shapes(
    op: &'static str,
    config: EsenSo3GridConfig,
    matrix: &[f64],
    grid_like: &[f64],
    coefficient_like: &[f64],
) -> Result<usize> {
    check_len(
        "fairchem_esen_so3_grid matrix",
        config.matrix_len(),
        matrix.len(),
    )?;
    let grid_stride = config.grid_node_stride();
    if !grid_like.len().is_multiple_of(grid_stride) {
        return Err(invalid_so3(format!(
            "{op} grid tensor length {} is not a multiple of node stride {grid_stride}",
            grid_like.len()
        )));
    }
    let nodes = grid_like.len() / grid_stride;
    check_len(
        op,
        nodes * config.coefficient_node_stride(),
        coefficient_like.len(),
    )?;
    Ok(nodes)
}

fn check_so3_batched_matmul_shapes(
    op: &'static str,
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    input: &[f64],
    out: &[f64],
) -> Result<usize> {
    let input_stride = config.input_item_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} input length {} is not a multiple of item stride {input_stride}",
            input.len()
        )));
    }
    let items = input.len() / input_stride;
    check_len(op, items * config.output_item_stride(), out.len())?;
    check_len(
        "fairchem_esen_so3_batched_matmul matrix",
        items * config.matrix_item_stride(),
        matrix.len(),
    )?;
    Ok(items)
}

fn check_so3_batched_matmul_vjp_shapes(
    op: &'static str,
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    grad_out: &[f64],
    grad_input: &[f64],
) -> Result<usize> {
    let input_stride = config.input_item_stride();
    if !grad_input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} grad_input length {} is not a multiple of item stride {input_stride}",
            grad_input.len()
        )));
    }
    let items = grad_input.len() / input_stride;
    check_len(op, items * config.output_item_stride(), grad_out.len())?;
    check_len(
        "fairchem_esen_so3_batched_matmul matrix",
        items * config.matrix_item_stride(),
        matrix.len(),
    )?;
    Ok(items)
}

fn check_so3_batched_matmul_matrix_vjp_shapes(
    op: &'static str,
    config: EsenSo3BatchedMatmulConfig,
    input: &[f64],
    grad_out: &[f64],
    grad_matrix: &[f64],
) -> Result<usize> {
    let input_stride = config.input_item_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} input length {} is not a multiple of item stride {input_stride}",
            input.len()
        )));
    }
    let items = input.len() / input_stride;
    check_len(op, items * config.output_item_stride(), grad_out.len())?;
    check_len(
        "fairchem_esen_so3_batched_matmul grad_matrix",
        items * config.matrix_item_stride(),
        grad_matrix.len(),
    )?;
    Ok(items)
}

#[cfg(not(all(feature = "accelerate-blas", target_os = "macos")))]
#[allow(dead_code)]
fn degree_for_component(m_index: usize) -> usize {
    let mut degree = 0;
    while (degree + 1) * (degree + 1) <= m_index {
        degree += 1;
    }
    degree
}

fn weight_index(
    config: EsenSo3LinearConfig,
    degree: usize,
    out_channel: usize,
    in_channel: usize,
) -> usize {
    (degree * config.out_channels + out_channel) * config.in_channels + in_channel
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

fn invalid_so3(message: impl Into<String>) -> CoreError {
    CoreError::InvalidSo3(message.into())
}

/// f32 twin of [`fairchem_esen_so3_batched_matmul_forward`].
pub fn fairchem_esen_so3_batched_matmul_forward_f32(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f32],
    input: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let items = check_so3_batched_matmul_shapes_f32(
        "fairchem_esen_so3_batched_matmul_forward_f32",
        config,
        matrix,
        input,
        out,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_F32_PARALLEL_ITEMS_MIN {
        out.par_chunks_mut(config.output_item_stride())
            .enumerate()
            .for_each(|(item, out_item)| {
                let matrix_item_start = item * config.matrix_item_stride();
                let input_item_start = item * config.input_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let matrix_row_start =
                        matrix_item_start + out_coefficient * config.input_coefficients;
                    let out_row_start = out_coefficient * config.channels;
                    out_item[out_row_start..out_row_start + config.channels].fill(0.0_f32);
                    for in_coefficient in 0..config.input_coefficients {
                        let weight = matrix[matrix_row_start + in_coefficient];
                        let input_row_start = input_item_start + in_coefficient * config.channels;
                        for channel in 0..config.channels {
                            out_item[out_row_start + channel] +=
                                weight * input[input_row_start + channel];
                        }
                    }
                }
            });
        return Ok(());
    }

    // Use SGEMM for large Wigner matrices where BLAS amortises per-call
    // overhead over enough FLOPs.  Small matrices (e.g. 9×9 for lmax=2)
    // are faster through the auto-vectorised scalar loop below because the
    // SGEMM call overhead exceeds the compute time on Apple Silicon.
    // Threshold: matrix area >= 256 (i.e. 16×16 or larger).
    #[cfg(e3nn_core_cblas_backend)]
    if config.output_coefficients * config.input_coefficients >= 256 {
        for item in 0..items {
            let matrix_item_start = item * config.matrix_item_stride();
            let input_item_start = item * config.input_item_stride();
            let out_item_start = item * config.output_item_stride();
            accelerate_blas_f32::matmul_nn(
                &matrix[matrix_item_start..matrix_item_start + config.matrix_item_stride()],
                config.output_coefficients,
                config.input_coefficients,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.channels,
                &mut out[out_item_start..out_item_start + config.output_item_stride()],
            );
        }
        return Ok(());
    }

    let ch = config.channels;
    for item in 0..items {
        let matrix_item =
            &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
        let input_item = &input[item * config.input_item_stride()..][..config.input_item_stride()];
        let out_item =
            &mut out[item * config.output_item_stride()..][..config.output_item_stride()];
        for out_coefficient in 0..config.output_coefficients {
            let out_row = &mut out_item[out_coefficient * ch..out_coefficient * ch + ch];
            // Write first in_coeff (avoids fill(0) + accumulate in separate passes)
            let w0 = matrix_item[out_coefficient * config.input_coefficients];
            let in0 = &input_item[0..ch];
            for (o, &i) in out_row.iter_mut().zip(in0.iter()) {
                *o = w0 * i;
            }
            // Accumulate remaining in_coefficients
            for in_coefficient in 1..config.input_coefficients {
                let weight =
                    matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                let in_row = &input_item[in_coefficient * ch..in_coefficient * ch + ch];
                for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
                    *o += weight * i;
                }
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_so3_batched_matmul_vjp_input`].
pub fn fairchem_esen_so3_batched_matmul_vjp_input_f32(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    let items = check_so3_batched_matmul_vjp_shapes_f32(
        "fairchem_esen_so3_batched_matmul_vjp_input_f32",
        config,
        matrix,
        grad_out,
        grad_input,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_F32_PARALLEL_ITEMS_MIN {
        grad_input
            .par_chunks_mut(config.input_item_stride())
            .enumerate()
            .for_each(|(item, grad_input_item)| {
                let matrix_item_start = item * config.matrix_item_stride();
                let grad_out_item_start = item * config.output_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let matrix_row_start =
                        matrix_item_start + out_coefficient * config.input_coefficients;
                    let grad_out_row_start =
                        grad_out_item_start + out_coefficient * config.channels;
                    for in_coefficient in 0..config.input_coefficients {
                        let weight = matrix[matrix_row_start + in_coefficient];
                        let grad_input_row_start = in_coefficient * config.channels;
                        for channel in 0..config.channels {
                            grad_input_item[grad_input_row_start + channel] +=
                                weight * grad_out[grad_out_row_start + channel];
                        }
                    }
                }
            });
        return Ok(());
    }

    // Use SGEMM for large Wigner matrices where BLAS amortises per-call
    // overhead over enough FLOPs.  Small matrices (e.g. 9×9 for lmax=2)
    // are faster through the auto-vectorised scalar loop below because the
    // SGEMM call overhead exceeds the compute time on Apple Silicon.
    // Threshold: matrix area >= 256 (i.e. 16×16 or larger).
    #[cfg(e3nn_core_cblas_backend)]
    if config.output_coefficients * config.input_coefficients >= 256 {
        for item in 0..items {
            let matrix_item_start = item * config.matrix_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            let grad_input_item_start = item * config.input_item_stride();
            accelerate_blas_f32::matmul_tn_add(
                &matrix[matrix_item_start..matrix_item_start + config.matrix_item_stride()],
                config.output_coefficients,
                config.input_coefficients,
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.channels,
                &mut grad_input
                    [grad_input_item_start..grad_input_item_start + config.input_item_stride()],
            );
        }
        return Ok(());
    }

    let ch = config.channels;
    for item in 0..items {
        let matrix_item =
            &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
        let grad_out_item =
            &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
        let grad_input_item =
            &mut grad_input[item * config.input_item_stride()..][..config.input_item_stride()];
        for out_coefficient in 0..config.output_coefficients {
            let grad_out_row = &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
            for in_coefficient in 0..config.input_coefficients {
                let weight =
                    matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                let grad_input_row =
                    &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                    *gi += weight * go;
                }
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_so3_batched_matmul_vjp_input_write`].
pub fn fairchem_esen_so3_batched_matmul_vjp_input_write_f32(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    let items = check_so3_batched_matmul_vjp_shapes_f32(
        "fairchem_esen_so3_batched_matmul_vjp_input_write_f32",
        config,
        matrix,
        grad_out,
        grad_input,
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_F32_PARALLEL_ITEMS_MIN {
        let ch = config.channels;
        grad_input
            .par_chunks_mut(config.input_item_stride())
            .enumerate()
            .for_each(|(item, grad_input_item)| {
                let matrix_item =
                    &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
                let grad_out_item =
                    &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
                for in_coefficient in 0..config.input_coefficients {
                    let weight = matrix_item[in_coefficient];
                    let grad_out_row = &grad_out_item[0..ch];
                    let grad_input_row =
                        &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                    for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                        *gi = weight * go;
                    }
                }
                for out_coefficient in 1..config.output_coefficients {
                    let grad_out_row =
                        &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
                    for in_coefficient in 0..config.input_coefficients {
                        let weight = matrix_item
                            [out_coefficient * config.input_coefficients + in_coefficient];
                        let grad_input_row =
                            &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                        for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                            *gi += weight * go;
                        }
                    }
                }
            });
        return Ok(());
    }

    let ch = config.channels;
    for item in 0..items {
        let matrix_item =
            &matrix[item * config.matrix_item_stride()..][..config.matrix_item_stride()];
        let grad_out_item =
            &grad_out[item * config.output_item_stride()..][..config.output_item_stride()];
        let grad_input_item =
            &mut grad_input[item * config.input_item_stride()..][..config.input_item_stride()];
        for in_coefficient in 0..config.input_coefficients {
            let weight = matrix_item[in_coefficient];
            let grad_out_row = &grad_out_item[0..ch];
            let grad_input_row =
                &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
            for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                *gi = weight * go;
            }
        }
        for out_coefficient in 1..config.output_coefficients {
            let grad_out_row = &grad_out_item[out_coefficient * ch..out_coefficient * ch + ch];
            for in_coefficient in 0..config.input_coefficients {
                let weight =
                    matrix_item[out_coefficient * config.input_coefficients + in_coefficient];
                let grad_input_row =
                    &mut grad_input_item[in_coefficient * ch..in_coefficient * ch + ch];
                for (gi, &go) in grad_input_row.iter_mut().zip(grad_out_row.iter()) {
                    *gi += weight * go;
                }
            }
        }
    }
    Ok(())
}
/// f32-input twin of [`fairchem_esen_so3_batched_matmul_vjp_matrix`].
///
/// Takes f32 `input` and `grad_out`, accumulates the matrix gradient into the
/// f64 `grad_matrix` buffer. Used in the f32 forward pass where the Wigner
/// gradient must remain in f64 for the force pipeline.
pub fn fairchem_esen_so3_batched_matmul_vjp_matrix_f32(
    config: EsenSo3BatchedMatmulConfig,
    input: &[f32],
    grad_out: &[f32],
    grad_matrix: &mut [f64],
) -> Result<()> {
    // Validate sizes using the f32 input and grad_out shapes
    let input_stride = config.input_item_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_batched_matmul_vjp_matrix_f32 input length {} is not a multiple of item stride {input_stride}",
            input.len()
        )));
    }
    let items = input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_batched_matmul_vjp_matrix_f32 grad_out",
        items * config.output_item_stride(),
        grad_out.len(),
    )?;
    check_len(
        "fairchem_esen_so3_batched_matmul_vjp_matrix_f32 grad_matrix",
        items * config.matrix_item_stride(),
        grad_matrix.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        grad_matrix
            .par_chunks_mut(config.matrix_item_stride())
            .enumerate()
            .for_each(|(item, grad_matrix_item)| {
                let input_item_start = item * config.input_item_stride();
                let grad_out_item_start = item * config.output_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let grad_out_row_start =
                        grad_out_item_start + out_coefficient * config.channels;
                    let grad_matrix_row_start = out_coefficient * config.input_coefficients;
                    for in_coefficient in 0..config.input_coefficients {
                        let input_row_start = input_item_start + in_coefficient * config.channels;
                        let mut dot = 0.0_f64;
                        for channel in 0..config.channels {
                            dot += (grad_out[grad_out_row_start + channel] as f64)
                                * (input[input_row_start + channel] as f64);
                        }
                        grad_matrix_item[grad_matrix_row_start + in_coefficient] += dot;
                    }
                }
            });
        return Ok(());
    }

    #[cfg(all(feature = "accelerate-blas", target_os = "macos"))]
    {
        // Compute grad_matrix[out_coeff, in_coeff] = dot(grad_out[out_coeff, :], input[in_coeff, :])
        // = grad_out [m, k] * input.T [k, n] → matrix [m, n] where m=out_coeff, n=in_coeff, k=channels.
        // Compute in f32 per item, then accumulate into f64.
        let matrix_dim = config.matrix_item_stride(); // output_coefficients * input_coefficients

        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && items >= SO3_F32_PARALLEL_ITEMS_MIN {
            grad_matrix.par_chunks_mut(matrix_dim).enumerate().for_each(
                |(item, grad_matrix_item)| {
                    let input_item_start = item * config.input_item_stride();
                    let grad_out_item_start = item * config.output_item_stride();
                    let mut grad_matrix_f32 = vec![0.0_f32; matrix_dim];
                    accelerate_blas_f32::matmul_nt_write(
                        &grad_out[grad_out_item_start
                            ..grad_out_item_start + config.output_item_stride()],
                        config.output_coefficients,
                        config.channels,
                        &input[input_item_start..input_item_start + config.input_item_stride()],
                        config.input_coefficients,
                        &mut grad_matrix_f32,
                    );
                    for (dst, &src) in grad_matrix_item.iter_mut().zip(grad_matrix_f32.iter()) {
                        *dst += src as f64;
                    }
                },
            );
            return Ok(());
        }

        let mut grad_matrix_f32 = vec![0.0_f32; matrix_dim];
        for item in 0..items {
            let matrix_item_start = item * matrix_dim;
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            accelerate_blas_f32::matmul_nt_write(
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.output_coefficients,
                config.channels,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.input_coefficients,
                &mut grad_matrix_f32,
            );
            for (dst, &src) in grad_matrix[matrix_item_start..matrix_item_start + matrix_dim]
                .iter_mut()
                .zip(grad_matrix_f32.iter())
            {
                *dst += src as f64;
            }
        }
    }

    #[cfg(all(feature = "parallel", not(e3nn_core_cblas_backend)))]
    if rayon::current_num_threads() > 1 && items >= 128 {
        let matrix_dim = config.matrix_item_stride();
        grad_matrix
            .par_chunks_mut(matrix_dim)
            .enumerate()
            .for_each(|(item, grad_matrix_item)| {
                let input_item_start = item * config.input_item_stride();
                let grad_out_item_start = item * config.output_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let matrix_row_start = out_coefficient * config.input_coefficients;
                    let grad_out_row_start =
                        grad_out_item_start + out_coefficient * config.channels;
                    for in_coefficient in 0..config.input_coefficients {
                        let input_row_start = input_item_start + in_coefficient * config.channels;
                        let mut dot = 0.0_f64;
                        for channel in 0..config.channels {
                            dot += (grad_out[grad_out_row_start + channel] as f64)
                                * (input[input_row_start + channel] as f64);
                        }
                        grad_matrix_item[matrix_row_start + in_coefficient] += dot;
                    }
                }
            });
        return Ok(());
    }

    #[cfg(not(e3nn_core_cblas_backend))]
    for item in 0..items {
        let matrix_item_start = item * config.matrix_item_stride();
        let input_item_start = item * config.input_item_stride();
        let grad_out_item_start = item * config.output_item_stride();
        for out_coefficient in 0..config.output_coefficients {
            let matrix_row_start = matrix_item_start + out_coefficient * config.input_coefficients;
            let grad_out_row_start = grad_out_item_start + out_coefficient * config.channels;
            for in_coefficient in 0..config.input_coefficients {
                let input_row_start = input_item_start + in_coefficient * config.channels;
                let mut dot = 0.0_f64;
                for channel in 0..config.channels {
                    dot += (grad_out[grad_out_row_start + channel] as f64)
                        * (input[input_row_start + channel] as f64);
                }
                grad_matrix[matrix_row_start + in_coefficient] += dot;
            }
        }
    }
    Ok(())
}

/// f32-input twin of [`fairchem_esen_so3_batched_matmul_vjp_matrix_write`].
pub fn fairchem_esen_so3_batched_matmul_vjp_matrix_write_f32(
    config: EsenSo3BatchedMatmulConfig,
    input: &[f32],
    grad_out: &[f32],
    grad_matrix: &mut [f64],
) -> Result<()> {
    let input_stride = config.input_item_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "fairchem_esen_so3_batched_matmul_vjp_matrix_write_f32 input length {} is not a multiple of item stride {input_stride}",
            input.len()
        )));
    }
    let items = input.len() / input_stride;
    check_len(
        "fairchem_esen_so3_batched_matmul_vjp_matrix_write_f32 grad_out",
        items * config.output_item_stride(),
        grad_out.len(),
    )?;
    check_len(
        "fairchem_esen_so3_batched_matmul_vjp_matrix_write_f32 grad_matrix",
        items * config.matrix_item_stride(),
        grad_matrix.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && items >= SO3_BATCHED_MATMUL_PARALLEL_MIN_ITEMS {
        grad_matrix
            .par_chunks_mut(config.matrix_item_stride())
            .enumerate()
            .for_each(|(item, grad_matrix_item)| {
                let input_item_start = item * config.input_item_stride();
                let grad_out_item_start = item * config.output_item_stride();
                for out_coefficient in 0..config.output_coefficients {
                    let grad_out_row_start =
                        grad_out_item_start + out_coefficient * config.channels;
                    let grad_matrix_row_start = out_coefficient * config.input_coefficients;
                    for in_coefficient in 0..config.input_coefficients {
                        let input_row_start = input_item_start + in_coefficient * config.channels;
                        let mut dot = 0.0_f64;
                        for channel in 0..config.channels {
                            dot += (grad_out[grad_out_row_start + channel] as f64)
                                * (input[input_row_start + channel] as f64);
                        }
                        grad_matrix_item[grad_matrix_row_start + in_coefficient] = dot;
                    }
                }
            });
        return Ok(());
    }

    #[cfg(all(feature = "accelerate-blas", target_os = "macos"))]
    {
        let matrix_dim = config.matrix_item_stride();
        let mut grad_matrix_f32 = vec![0.0_f32; matrix_dim];
        for item in 0..items {
            let matrix_item_start = item * matrix_dim;
            let input_item_start = item * config.input_item_stride();
            let grad_out_item_start = item * config.output_item_stride();
            accelerate_blas_f32::matmul_nt_write(
                &grad_out[grad_out_item_start..grad_out_item_start + config.output_item_stride()],
                config.output_coefficients,
                config.channels,
                &input[input_item_start..input_item_start + config.input_item_stride()],
                config.input_coefficients,
                &mut grad_matrix_f32,
            );
            for (dst, &src) in grad_matrix[matrix_item_start..matrix_item_start + matrix_dim]
                .iter_mut()
                .zip(grad_matrix_f32.iter())
            {
                *dst = src as f64;
            }
        }
    }

    #[cfg(not(all(feature = "accelerate-blas", target_os = "macos")))]
    for item in 0..items {
        let matrix_item_start = item * config.matrix_item_stride();
        let input_item_start = item * config.input_item_stride();
        let grad_out_item_start = item * config.output_item_stride();
        for out_coefficient in 0..config.output_coefficients {
            let matrix_row_start = matrix_item_start + out_coefficient * config.input_coefficients;
            let grad_out_row_start = grad_out_item_start + out_coefficient * config.channels;
            for in_coefficient in 0..config.input_coefficients {
                let input_row_start = input_item_start + in_coefficient * config.channels;
                let mut dot = 0.0_f64;
                for channel in 0..config.channels {
                    dot += (grad_out[grad_out_row_start + channel] as f64)
                        * (input[input_row_start + channel] as f64);
                }
                grad_matrix[matrix_row_start + in_coefficient] = dot;
            }
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
#[allow(clippy::too_many_arguments, dead_code, unsafe_code)]
mod accelerate_blas_f32 {
    use std::os::raw::{c_float, c_int};

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

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
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
    }

    pub fn dense_forward_transposed_weight(
        input: &[f32],
        rows: usize,
        in_dim: usize,
        weight_out_by_in: &[f32],
        out_dim: usize,
        out: &mut [f32],
    ) {
        unsafe {
            // Invariant: all slices are validated by the caller as row-major
            // [rows, in_dim], [out_dim, in_dim], and [rows, out_dim].
            cblas_sgemm(
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

    pub fn dense_vjp_input_add(
        grad_out: &[f32],
        rows: usize,
        out_dim: usize,
        weight_out_by_in: &[f32],
        in_dim: usize,
        grad_input: &mut [f32],
    ) {
        unsafe {
            // Invariant: computes grad_out [rows, out_dim] * weight
            // [out_dim, in_dim] and accumulates into grad_input.
            cblas_sgemm(
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

    pub fn matmul_nn(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
        // Shape guard for the raw cblas_sgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_nn a is [m, k]");
        debug_assert_eq!(b.len(), k * n, "matmul_nn b is [k, n]");
        debug_assert_eq!(out.len(), m * n, "matmul_nn out is [m, n]");
        unsafe {
            // Invariant: computes a [m, k] * b [k, n] into out [m, n] (overwrite).
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_tn_add(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
        // Shape guard for the raw cblas_sgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_tn_add a is [m, k] (a.T is [k, m])");
        debug_assert_eq!(b.len(), m * n, "matmul_tn_add b is [m, n]");
        debug_assert_eq!(out.len(), k * n, "matmul_tn_add out is [k, n]");
        unsafe {
            // Invariant: accumulates a.T [k, m] * b [m, n] into out [k, n].
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                k as c_int,
                n as c_int,
                m as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                1.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_tn_write(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
        // Shape guard for the raw cblas_sgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(
            a.len(),
            m * k,
            "matmul_tn_write a is [m, k] (a.T is [k, m])"
        );
        debug_assert_eq!(b.len(), m * n, "matmul_tn_write b is [m, n]");
        debug_assert_eq!(out.len(), k * n, "matmul_tn_write out is [k, n]");
        unsafe {
            // Invariant: computes a.T [k, m] * b [m, n] into fresh out [k, n].
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                k as c_int,
                n as c_int,
                m as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }

    pub fn matmul_nt_write(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
        // Shape guard for the raw cblas_sgemm FFI boundary (zero-cost in release).
        debug_assert_eq!(a.len(), m * k, "matmul_nt_write a is [m, k]");
        debug_assert_eq!(
            b.len(),
            n * k,
            "matmul_nt_write b is [n, k] (b.T is [k, n])"
        );
        debug_assert_eq!(out.len(), m * n, "matmul_nt_write out is [m, n]");
        unsafe {
            // Invariant: computes a [m, k] * b.T [k, n] into fresh out [m, n].
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                k as c_int,
                0.0,
                out.as_mut_ptr(),
                n as c_int,
            );
        }
    }
}

fn check_so3_batched_matmul_shapes_f32(
    op: &'static str,
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f32],
    input: &[f32],
    out: &[f32],
) -> Result<usize> {
    let input_stride = config.input_item_stride();
    if !input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} input length {} is not a multiple of item stride {input_stride}",
            input.len()
        )));
    }
    let items = input.len() / input_stride;
    check_len(op, items * config.output_item_stride(), out.len())?;
    check_len(
        "fairchem_esen_so3_batched_matmul_f32 matrix",
        items * config.matrix_item_stride(),
        matrix.len(),
    )?;
    Ok(items)
}

fn check_so3_batched_matmul_vjp_shapes_f32(
    op: &'static str,
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f32],
    grad_out: &[f32],
    grad_input: &[f32],
) -> Result<usize> {
    let input_stride = config.input_item_stride();
    if !grad_input.len().is_multiple_of(input_stride) {
        return Err(invalid_so3(format!(
            "{op} grad_input length {} is not a multiple of item stride {input_stride}",
            grad_input.len()
        )));
    }
    let items = grad_input.len() / input_stride;
    check_len(op, items * config.output_item_stride(), grad_out.len())?;
    check_len(
        "fairchem_esen_so3_batched_matmul_f32 matrix",
        items * config.matrix_item_stride(),
        matrix.len(),
    )?;
    Ok(items)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests_f32 {
    use super::*;

    fn make_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.11 + 0.2).sin()).collect()
    }

    #[test]
    fn batched_matmul_forward_f32_vs_f64() {
        // items=4, input_coefficients=9, output_coefficients=9, channels=32
        // (using 32 channels to keep test fast, production uses 128)
        let config = EsenSo3BatchedMatmulConfig::new(9, 9, 32).unwrap();
        let items = 4usize;
        let n_matrix = items * config.matrix_item_stride();
        let n_input = items * config.input_item_stride();
        let n_out = items * config.output_item_stride();

        let matrix_f32 = make_vals(n_matrix);
        let input_f32 = make_vals(n_input);
        let matrix_f64: Vec<f64> = matrix_f32.iter().map(|&x| x as f64).collect();
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0_f32; n_out];
        let mut out_f64 = vec![0.0_f64; n_out];

        fairchem_esen_so3_batched_matmul_forward_f32(config, &matrix_f32, &input_f32, &mut out_f32)
            .unwrap();
        fairchem_esen_so3_batched_matmul_forward(config, &matrix_f64, &input_f64, &mut out_f64)
            .unwrap();

        for (i, (&a, &b)) in out_f32.iter().zip(out_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-6 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 5e-4,
                "index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn batched_matmul_matrix_vjp_f32_vs_f64() {
        assert_batched_matmul_matrix_vjp_f32_vs_f64(4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn batched_matmul_matrix_vjp_f32_vs_f64_parallel_path() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| assert_batched_matmul_matrix_vjp_f32_vs_f64(160));
    }

    #[test]
    fn batched_matmul_input_vjp_write_f32_matches_accumulate() {
        assert_batched_matmul_input_vjp_write_f32_matches_accumulate(4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn batched_matmul_input_vjp_write_f32_matches_accumulate_parallel_path() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| assert_batched_matmul_input_vjp_write_f32_matches_accumulate(160));
    }

    fn assert_batched_matmul_input_vjp_write_f32_matches_accumulate(items: usize) {
        let config = EsenSo3BatchedMatmulConfig::new(9, 6, 32).unwrap();
        let n_matrix = items * config.matrix_item_stride();
        let n_grad_out = items * config.output_item_stride();
        let n_grad_input = items * config.input_item_stride();

        let matrix = make_vals(n_matrix);
        let grad_out = make_vals(n_grad_out);
        let mut expected = vec![0.0_f32; n_grad_input];
        let mut actual = vec![123.0_f32; n_grad_input];

        fairchem_esen_so3_batched_matmul_vjp_input_f32(config, &matrix, &grad_out, &mut expected)
            .unwrap();
        fairchem_esen_so3_batched_matmul_vjp_input_write_f32(
            config,
            &matrix,
            &grad_out,
            &mut actual,
        )
        .unwrap();

        for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "batched_matmul_input_vjp_write_f32 index {i}: write={a} accumulate={b}"
            );
        }
    }

    fn assert_batched_matmul_matrix_vjp_f32_vs_f64(items: usize) {
        let config = EsenSo3BatchedMatmulConfig::new(9, 6, 32).unwrap();
        let n_input = items * config.input_item_stride();
        let n_grad_out = items * config.output_item_stride();
        let n_grad_matrix = items * config.matrix_item_stride();

        let input_f32 = make_vals(n_input);
        let grad_out_f32 = make_vals(n_grad_out);
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
        let grad_out_f64: Vec<f64> = grad_out_f32.iter().map(|&x| x as f64).collect();

        let mut grad_matrix_f32 = vec![0.0_f64; n_grad_matrix];
        let mut grad_matrix_f32_write = vec![123.0_f64; n_grad_matrix];
        let mut grad_matrix_f64 = vec![0.0_f64; n_grad_matrix];

        fairchem_esen_so3_batched_matmul_vjp_matrix_f32(
            config,
            &input_f32,
            &grad_out_f32,
            &mut grad_matrix_f32,
        )
        .unwrap();
        fairchem_esen_so3_batched_matmul_vjp_matrix_write_f32(
            config,
            &input_f32,
            &grad_out_f32,
            &mut grad_matrix_f32_write,
        )
        .unwrap();
        fairchem_esen_so3_batched_matmul_vjp_matrix(
            config,
            &input_f64,
            &grad_out_f64,
            &mut grad_matrix_f64,
        )
        .unwrap();

        for (i, (&a, &b)) in grad_matrix_f32_write
            .iter()
            .zip(grad_matrix_f32.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "batched_matmul_matrix_vjp_write_f32 index {i}: write={a} accumulate={b}"
            );
        }

        for (i, (&a, &b)) in grad_matrix_f32
            .iter()
            .zip(grad_matrix_f64.iter())
            .enumerate()
        {
            let rel_err = if b.abs() > 1e-6 {
                ((a - b) / b).abs()
            } else {
                (a - b).abs()
            };
            assert!(
                rel_err < 2e-3,
                "batched_matmul_matrix_vjp_f32 index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn so3_linear_forward_f32_matches_f64() {
        // nodes=8, lmax=2 (basis_dim=9), in_channels=32, out_channels=32
        // Exercises the BLAS path (on macOS with accelerate-blas) vs the f64 reference.
        let config = EsenSo3LinearConfig::new(2, 32, 32).unwrap();
        let nodes = 8usize;
        let n_input = nodes * config.input_node_stride();
        let n_weight = config.weight_len();
        let n_out = nodes * config.output_node_stride();

        let input_f32 = make_vals(n_input);
        let weight_f32 = make_vals(n_weight);
        let bias_f32 = make_vals(config.out_channels);

        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
        let weight_f64: Vec<f64> = weight_f32.iter().map(|&x| x as f64).collect();
        let bias_f64: Vec<f64> = bias_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0_f32; n_out];
        let mut out_f64 = vec![0.0_f64; n_out];

        fairchem_esen_so3_linear_forward_f32(
            config,
            &input_f32,
            &weight_f32,
            &bias_f32,
            &mut out_f32,
        )
        .unwrap();
        fairchem_esen_so3_linear_forward(config, &input_f64, &weight_f64, &bias_f64, &mut out_f64)
            .unwrap();

        for (i, (&a, &b)) in out_f32.iter().zip(out_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-6 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 5e-4,
                "so3_linear_forward_f32 index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn so3_linear_vjp_f32_matches_f64() {
        // Verify the f32 VJP accumulates correctly vs the f64 reference.
        let config = EsenSo3LinearConfig::new(2, 32, 32).unwrap();
        let nodes = 8usize;
        let n_input = nodes * config.input_node_stride();
        let n_weight = config.weight_len();
        let n_grad_out = nodes * config.output_node_stride();

        let weight_f32 = make_vals(n_weight);
        let grad_out_f32 = make_vals(n_grad_out);
        let weight_f64: Vec<f64> = weight_f32.iter().map(|&x| x as f64).collect();
        let grad_out_f64: Vec<f64> = grad_out_f32.iter().map(|&x| x as f64).collect();

        let mut grad_input_f32 = vec![0.0_f32; n_input];
        let mut grad_input_f64 = vec![0.0_f64; n_input];

        fairchem_esen_so3_linear_vjp_f32(config, &weight_f32, &grad_out_f32, &mut grad_input_f32)
            .unwrap();
        fairchem_esen_so3_linear_vjp(config, &weight_f64, &grad_out_f64, &mut grad_input_f64)
            .unwrap();

        for (i, (&a, &b)) in grad_input_f32.iter().zip(grad_input_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-6 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 2e-3,
                "so3_linear_vjp_f32 index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn so3_linear_vjp_write_f32_matches_zero_accumulate() {
        let config = EsenSo3LinearConfig::new(2, 17, 13).unwrap();
        let nodes = 5usize;
        let n_input = nodes * config.input_node_stride();
        let n_weight = config.weight_len();
        let n_grad_out = nodes * config.output_node_stride();

        let weight = make_vals(n_weight);
        let grad_out = make_vals(n_grad_out);

        let mut expected = vec![0.0_f32; n_input];
        fairchem_esen_so3_linear_vjp_f32(config, &weight, &grad_out, &mut expected).unwrap();

        let mut actual = vec![19.0_f32; n_input];
        fairchem_esen_so3_linear_vjp_write_f32(config, &weight, &grad_out, &mut actual).unwrap();

        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-6,
                "so3_linear_vjp_write_f32 index {i}: actual={a} expected={e}"
            );
        }
    }
}
