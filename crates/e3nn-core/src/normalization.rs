//! Normalization primitives used by fixed-weight equivariant models.

use crate::{CoreError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EsenRmsNormShNormalization {
    /// Sum over spherical components, then average over channels.
    Norm,
    /// Average uniformly over spherical components and channels.
    Component,
    /// Average each degree over its `m` components, then average degrees and channels.
    ComponentDegreeBalanced,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EsenRmsNormShConfig {
    l_max: usize,
    channels: usize,
    eps: f64,
    normalization: EsenRmsNormShNormalization,
    centering: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenDegreeLayerScaleConfig {
    l_max: usize,
    channels: usize,
}

impl EsenDegreeLayerScaleConfig {
    pub fn new(l_max: usize, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(invalid_normalization("channels must be nonzero"));
        }
        l_max
            .checked_add(1)
            .and_then(|degree_count| degree_count.checked_mul(degree_count))
            .ok_or_else(|| invalid_normalization(format!("l_max is too large: {l_max}")))?;
        Ok(Self { l_max, channels })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
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
    pub fn node_stride(&self) -> usize {
        self.basis_dim() * self.channels
    }
}

impl EsenRmsNormShConfig {
    pub fn new(
        l_max: usize,
        channels: usize,
        eps: f64,
        normalization: EsenRmsNormShNormalization,
        centering: bool,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(invalid_normalization("channels must be nonzero"));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(invalid_normalization(format!(
                "eps must be positive and finite, got {eps}"
            )));
        }
        l_max
            .checked_add(1)
            .and_then(|degree_count| degree_count.checked_mul(degree_count))
            .ok_or_else(|| invalid_normalization(format!("l_max is too large: {l_max}")))?;
        Ok(Self {
            l_max,
            channels,
            eps,
            normalization,
            centering,
        })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    #[must_use]
    pub fn normalization(&self) -> EsenRmsNormShNormalization {
        self.normalization
    }

    #[must_use]
    pub fn centering(&self) -> bool {
        self.centering
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
    pub fn node_stride(&self) -> usize {
        self.basis_dim() * self.channels
    }
}

/// FairChem eSEN degree layer scale forward pass for flat
/// `[nodes, (lmax + 1)^2, channels]` inputs.
///
/// `affine_weight` has shape `[lmax + 1, channels]` and is expanded to all
/// `m` components of each degree.
pub fn fairchem_esen_degree_layer_scale_forward(
    config: EsenDegreeLayerScaleConfig,
    input: &[f64],
    affine_weight: &[f64],
    out: &mut [f64],
) -> Result<()> {
    check_degree_layer_scale_shapes(
        "fairchem_esen_degree_layer_scale_forward",
        config,
        input,
        affine_weight,
        out,
    )?;
    for m_index in 0..config.basis_dim() {
        let degree = degree_for_component(m_index);
        for channel in 0..config.channels {
            let flat_offset = m_index * config.channels + channel;
            let weight = affine_weight[degree * config.channels + channel];
            for node_start in (0..input.len()).step_by(config.node_stride()) {
                let flat_index = node_start + flat_offset;
                out[flat_index] = input[flat_index] * weight;
            }
        }
    }
    Ok(())
}

/// Accumulate the input VJP of [`fairchem_esen_degree_layer_scale_forward`] into
/// `grad_input`.
pub fn fairchem_esen_degree_layer_scale_vjp(
    config: EsenDegreeLayerScaleConfig,
    affine_weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_degree_layer_scale_shapes(
        "fairchem_esen_degree_layer_scale_vjp",
        config,
        grad_out,
        affine_weight,
        grad_input,
    )?;
    for m_index in 0..config.basis_dim() {
        let degree = degree_for_component(m_index);
        for channel in 0..config.channels {
            let flat_offset = m_index * config.channels + channel;
            let weight = affine_weight[degree * config.channels + channel];
            for node_start in (0..grad_out.len()).step_by(config.node_stride()) {
                let flat_index = node_start + flat_offset;
                grad_input[flat_index] += grad_out[flat_index] * weight;
            }
        }
    }
    Ok(())
}

/// FairChem eSEN `rms_norm_sh` forward pass for flat
/// `[nodes, (lmax + 1)^2, channels]` inputs.
///
/// `affine_weight`, when present, has shape `[lmax + 1, channels]` and is
/// expanded to all `m` components of each degree. `affine_bias`, when present,
/// has shape `[channels]` and is applied only to the centered scalar channel.
pub fn fairchem_esen_rms_norm_sh_forward(
    config: EsenRmsNormShConfig,
    input: &[f64],
    affine_weight: Option<&[f64]>,
    affine_bias: Option<&[f64]>,
    out: &mut [f64],
) -> Result<()> {
    check_common_shapes(
        "fairchem_esen_rms_norm_sh_forward",
        config,
        input,
        affine_weight,
        affine_bias,
        out,
    )?;
    let stride = config.node_stride();
    let mut centered = vec![0.0; stride];
    for node_start in (0..input.len()).step_by(stride) {
        let input_node = &input[node_start..node_start + stride];
        let out_node = &mut out[node_start..node_start + stride];
        centered.copy_from_slice(input_node);
        apply_centering(config, &mut centered);
        let norm_accumulator = norm_accumulator(config, &centered);
        let scale = (norm_accumulator + config.eps).powf(-0.5);

        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            for channel in 0..config.channels {
                let flat_index = flat_index(config, m_index, channel);
                let weight = affine_weight
                    .map_or(1.0, |weights| weights[degree * config.channels + channel]);
                let mut value = centered[flat_index] * scale * weight;
                if config.centering && m_index == 0 {
                    if let Some(bias) = affine_bias {
                        value += bias[channel];
                    }
                }
                out_node[flat_index] = value;
            }
        }
    }
    Ok(())
}

/// Accumulate the input VJP of [`fairchem_esen_rms_norm_sh_forward`] into
/// `grad_input`.
pub fn fairchem_esen_rms_norm_sh_vjp(
    config: EsenRmsNormShConfig,
    input: &[f64],
    affine_weight: Option<&[f64]>,
    affine_bias: Option<&[f64]>,
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    check_common_shapes(
        "fairchem_esen_rms_norm_sh_vjp",
        config,
        input,
        affine_weight,
        affine_bias,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_rms_norm_sh_vjp grad_input",
        input.len(),
        grad_input.len(),
    )?;

    let stride = config.node_stride();
    let mut centered = vec![0.0; stride];
    let mut grad_centered = vec![0.0; stride];
    for node_start in (0..input.len()).step_by(stride) {
        let input_node = &input[node_start..node_start + stride];
        let grad_out_node = &grad_out[node_start..node_start + stride];
        let grad_input_node = &mut grad_input[node_start..node_start + stride];
        centered.copy_from_slice(input_node);
        apply_centering(config, &mut centered);
        let norm_accumulator = norm_accumulator(config, &centered);
        let scale = (norm_accumulator + config.eps).powf(-0.5);
        let scale_cubed = scale * scale * scale;
        let mut projection = 0.0;

        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            for channel in 0..config.channels {
                let flat_index = flat_index(config, m_index, channel);
                let weight = affine_weight
                    .map_or(1.0, |weights| weights[degree * config.channels + channel]);
                projection += grad_out_node[flat_index] * centered[flat_index] * weight;
            }
        }

        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            let norm_factor = norm_factor(config, degree);
            for channel in 0..config.channels {
                let flat_index = flat_index(config, m_index, channel);
                let weight = affine_weight
                    .map_or(1.0, |weights| weights[degree * config.channels + channel]);
                grad_centered[flat_index] = scale * grad_out_node[flat_index] * weight
                    - scale_cubed * projection * norm_factor * centered[flat_index];
            }
        }

        if config.centering {
            let mut scalar_grad_mean: f64 = grad_centered.iter().take(config.channels).sum();
            scalar_grad_mean /= config.channels as f64;
            for (grad_input_value, grad_centered_value) in grad_input_node
                .iter_mut()
                .zip(grad_centered.iter())
                .take(config.channels)
            {
                *grad_input_value += grad_centered_value - scalar_grad_mean;
            }
            for flat_index in config.channels..stride {
                grad_input_node[flat_index] += grad_centered[flat_index];
            }
        } else {
            for (grad_input_value, grad_centered_value) in
                grad_input_node.iter_mut().zip(grad_centered.iter())
            {
                *grad_input_value += grad_centered_value;
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_rms_norm_sh_forward`].
///
/// Computes the RMS norm in f32. `config.eps` is cast to f32 internally.
pub fn fairchem_esen_rms_norm_sh_forward_f32(
    config: EsenRmsNormShConfig,
    input: &[f32],
    affine_weight: Option<&[f32]>,
    affine_bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    check_common_shapes_f32(
        "fairchem_esen_rms_norm_sh_forward_f32",
        config,
        input,
        affine_weight,
        affine_bias,
        out,
    )?;
    let eps = config.eps as f32;
    let stride = config.node_stride();
    for node_start in (0..input.len()).step_by(stride) {
        let input_node = &input[node_start..node_start + stride];
        let out_node = &mut out[node_start..node_start + stride];
        let mut centered = input_node.to_vec();
        apply_centering_f32(config, &mut centered);
        let norm_acc = norm_accumulator_f32(config, &centered);
        let scale = (norm_acc + eps).powf(-0.5);

        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            for channel in 0..config.channels {
                let fi = flat_index(config, m_index, channel);
                let weight =
                    affine_weight.map_or(1.0_f32, |w| w[degree * config.channels + channel]);
                let mut value = centered[fi] * scale * weight;
                if config.centering && m_index == 0 {
                    if let Some(bias) = affine_bias {
                        value += bias[channel];
                    }
                }
                out_node[fi] = value;
            }
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_rms_norm_sh_vjp`].
///
/// Accumulates the input VJP in f32. `config.eps` is cast to f32 internally.
pub fn fairchem_esen_rms_norm_sh_vjp_f32(
    config: EsenRmsNormShConfig,
    input: &[f32],
    affine_weight: Option<&[f32]>,
    affine_bias: Option<&[f32]>,
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    check_common_shapes_f32(
        "fairchem_esen_rms_norm_sh_vjp_f32",
        config,
        input,
        affine_weight,
        affine_bias,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_rms_norm_sh_vjp_f32 grad_input",
        input.len(),
        grad_input.len(),
    )?;

    let eps = config.eps as f32;
    let stride = config.node_stride();
    for node_start in (0..input.len()).step_by(stride) {
        let input_node = &input[node_start..node_start + stride];
        let grad_out_node = &grad_out[node_start..node_start + stride];
        let grad_input_node = &mut grad_input[node_start..node_start + stride];
        let mut centered = input_node.to_vec();
        apply_centering_f32(config, &mut centered);
        let norm_acc = norm_accumulator_f32(config, &centered);
        let scale = (norm_acc + eps).powf(-0.5);
        let scale_cubed = scale * scale * scale;
        let mut projection = 0.0_f32;

        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            for channel in 0..config.channels {
                let fi = flat_index(config, m_index, channel);
                let weight =
                    affine_weight.map_or(1.0_f32, |w| w[degree * config.channels + channel]);
                projection += grad_out_node[fi] * centered[fi] * weight;
            }
        }

        let mut grad_centered = vec![0.0_f32; stride];
        for m_index in 0..config.basis_dim() {
            let degree = degree_for_component(m_index);
            let nf = norm_factor_f32(config, degree);
            for channel in 0..config.channels {
                let fi = flat_index(config, m_index, channel);
                let weight =
                    affine_weight.map_or(1.0_f32, |w| w[degree * config.channels + channel]);
                grad_centered[fi] = scale * grad_out_node[fi] * weight
                    - scale_cubed * projection * nf * centered[fi];
            }
        }

        if config.centering {
            let mut scalar_grad_mean: f32 = grad_centered.iter().take(config.channels).sum();
            scalar_grad_mean /= config.channels as f32;
            for (giv, gcv) in grad_input_node
                .iter_mut()
                .zip(grad_centered.iter())
                .take(config.channels)
            {
                *giv += gcv - scalar_grad_mean;
            }
            for fi in config.channels..stride {
                grad_input_node[fi] += grad_centered[fi];
            }
        } else {
            for (giv, gcv) in grad_input_node.iter_mut().zip(grad_centered.iter()) {
                *giv += gcv;
            }
        }
    }
    Ok(())
}

fn apply_centering_f32(config: EsenRmsNormShConfig, values: &mut [f32]) {
    if !config.centering {
        return;
    }
    let mut mean: f32 = values.iter().take(config.channels).sum();
    mean /= config.channels as f32;
    for value in values.iter_mut().take(config.channels) {
        *value -= mean;
    }
}

fn norm_accumulator_f32(config: EsenRmsNormShConfig, centered: &[f32]) -> f32 {
    let mut accumulator = 0.0_f32;
    for m_index in 0..config.basis_dim() {
        let degree = degree_for_component(m_index);
        let factor = norm_factor_f32(config, degree);
        for channel in 0..config.channels {
            let value = centered[flat_index(config, m_index, channel)];
            accumulator += factor * value * value;
        }
    }
    accumulator
}

fn norm_factor_f32(config: EsenRmsNormShConfig, degree: usize) -> f32 {
    match config.normalization {
        EsenRmsNormShNormalization::Norm => 1.0 / config.channels as f32,
        EsenRmsNormShNormalization::Component => {
            1.0 / (config.basis_dim() * config.channels) as f32
        }
        EsenRmsNormShNormalization::ComponentDegreeBalanced => {
            let m_count = 2 * degree + 1;
            1.0 / (m_count * config.degree_count() * config.channels) as f32
        }
    }
}

fn check_common_shapes_f32(
    op: &'static str,
    config: EsenRmsNormShConfig,
    input: &[f32],
    affine_weight: Option<&[f32]>,
    affine_bias: Option<&[f32]>,
    output_like: &[f32],
) -> Result<()> {
    check_len(op, input.len(), output_like.len())?;
    let stride = config.node_stride();
    if !input.len().is_multiple_of(stride) {
        return Err(invalid_normalization(format!(
            "{op} input length {} is not a multiple of node stride {stride}",
            input.len()
        )));
    }
    if let Some(weights) = affine_weight {
        check_len(
            "fairchem_esen_rms_norm_sh_f32 affine_weight",
            config.degree_count() * config.channels,
            weights.len(),
        )?;
    }
    if let Some(bias) = affine_bias {
        if !config.centering {
            return Err(invalid_normalization(
                "affine_bias requires centering because FairChem applies it only to l=0",
            ));
        }
        check_len(
            "fairchem_esen_rms_norm_sh_f32 affine_bias",
            config.channels,
            bias.len(),
        )?;
    }
    Ok(())
}

fn apply_centering(config: EsenRmsNormShConfig, values: &mut [f64]) {
    if !config.centering {
        return;
    }
    let mut mean: f64 = values.iter().take(config.channels).sum();
    mean /= config.channels as f64;
    for value in values.iter_mut().take(config.channels) {
        *value -= mean;
    }
}

fn norm_accumulator(config: EsenRmsNormShConfig, centered: &[f64]) -> f64 {
    let mut accumulator = 0.0;
    for m_index in 0..config.basis_dim() {
        let degree = degree_for_component(m_index);
        let factor = norm_factor(config, degree);
        for channel in 0..config.channels {
            let value = centered[flat_index(config, m_index, channel)];
            accumulator += factor * value * value;
        }
    }
    accumulator
}

fn norm_factor(config: EsenRmsNormShConfig, degree: usize) -> f64 {
    match config.normalization {
        EsenRmsNormShNormalization::Norm => 1.0 / config.channels as f64,
        EsenRmsNormShNormalization::Component => {
            1.0 / (config.basis_dim() * config.channels) as f64
        }
        EsenRmsNormShNormalization::ComponentDegreeBalanced => {
            let m_count = 2 * degree + 1;
            1.0 / (m_count * config.degree_count() * config.channels) as f64
        }
    }
}

fn degree_for_component(m_index: usize) -> usize {
    let mut degree = 0;
    while (degree + 1) * (degree + 1) <= m_index {
        degree += 1;
    }
    degree
}

fn flat_index(config: EsenRmsNormShConfig, m_index: usize, channel: usize) -> usize {
    m_index * config.channels + channel
}

fn check_degree_layer_scale_shapes(
    op: &'static str,
    config: EsenDegreeLayerScaleConfig,
    input: &[f64],
    affine_weight: &[f64],
    output_like: &[f64],
) -> Result<()> {
    check_len(op, input.len(), output_like.len())?;
    check_len(
        "fairchem_esen_degree_layer_scale affine_weight",
        config.degree_count() * config.channels,
        affine_weight.len(),
    )?;
    let stride = config.node_stride();
    if !input.len().is_multiple_of(stride) {
        return Err(invalid_normalization(format!(
            "{op} input length {} is not a multiple of node stride {stride}",
            input.len()
        )));
    }
    Ok(())
}

fn check_common_shapes(
    op: &'static str,
    config: EsenRmsNormShConfig,
    input: &[f64],
    affine_weight: Option<&[f64]>,
    affine_bias: Option<&[f64]>,
    output_like: &[f64],
) -> Result<()> {
    check_len(op, input.len(), output_like.len())?;
    let stride = config.node_stride();
    if !input.len().is_multiple_of(stride) {
        return Err(invalid_normalization(format!(
            "{op} input length {} is not a multiple of node stride {stride}",
            input.len()
        )));
    }
    if let Some(weights) = affine_weight {
        check_len(
            "fairchem_esen_rms_norm_sh affine_weight",
            config.degree_count() * config.channels,
            weights.len(),
        )?;
    }
    if let Some(bias) = affine_bias {
        if !config.centering {
            return Err(invalid_normalization(
                "affine_bias requires centering because FairChem applies it only to l=0",
            ));
        }
        check_len(
            "fairchem_esen_rms_norm_sh affine_bias",
            config.channels,
            bias.len(),
        )?;
    }
    Ok(())
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

fn invalid_normalization(message: impl Into<String>) -> CoreError {
    CoreError::InvalidNormalization(message.into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests_f32 {
    use super::*;

    fn make_test_input(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.17 + 0.5).sin()).collect()
    }

    #[test]
    fn rms_norm_sh_forward_f32_vs_f64() {
        // 3 nodes, L=2, C=4 => node_stride = 9*4 = 36, total = 108
        let config =
            EsenRmsNormShConfig::new(2, 4, 1e-5, EsenRmsNormShNormalization::Component, true)
                .unwrap();
        let n = 3 * config.node_stride();
        let input_f32: Vec<f32> = make_test_input(n);
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
        // weights: degree_count * channels = 3 * 4 = 12
        let w_f32: Vec<f32> = make_test_input(config.degree_count() * config.channels);
        let w_f64: Vec<f64> = w_f32.iter().map(|&x| x as f64).collect();
        // bias: channels = 4
        let b_f32: Vec<f32> = make_test_input(config.channels);
        let b_f64: Vec<f64> = b_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0_f32; n];
        let mut out_f64 = vec![0.0_f64; n];
        fairchem_esen_rms_norm_sh_forward_f32(
            config,
            &input_f32,
            Some(&w_f32),
            Some(&b_f32),
            &mut out_f32,
        )
        .unwrap();
        fairchem_esen_rms_norm_sh_forward(
            config,
            &input_f64,
            Some(&w_f64),
            Some(&b_f64),
            &mut out_f64,
        )
        .unwrap();

        for (i, (&a, &b)) in out_f32.iter().zip(out_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-10 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 2e-6,
                "index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn rms_norm_sh_vjp_f32_vs_f64() {
        let config =
            EsenRmsNormShConfig::new(2, 4, 1e-5, EsenRmsNormShNormalization::Component, false)
                .unwrap();
        let n = 2 * config.node_stride();
        let input_f32: Vec<f32> = make_test_input(n);
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
        let grad_out_f32: Vec<f32> = make_test_input(n).iter().map(|&x| x * 0.5).collect();
        let grad_out_f64: Vec<f64> = grad_out_f32.iter().map(|&x| x as f64).collect();

        let mut gi_f32 = vec![0.0_f32; n];
        let mut gi_f64 = vec![0.0_f64; n];
        fairchem_esen_rms_norm_sh_vjp_f32(
            config,
            &input_f32,
            None,
            None,
            &grad_out_f32,
            &mut gi_f32,
        )
        .unwrap();
        fairchem_esen_rms_norm_sh_vjp(config, &input_f64, None, None, &grad_out_f64, &mut gi_f64)
            .unwrap();

        for (i, (&a, &b)) in gi_f32.iter().zip(gi_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-6 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 2e-2,
                "index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }
}
