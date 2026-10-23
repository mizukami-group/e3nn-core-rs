//! Activation primitives used by fixed-weight equivariant models.

use crate::mlp::{silu, silu_derivative};
use crate::so3::{
    fairchem_esen_so3_from_grid_forward, fairchem_esen_so3_from_grid_vjp,
    fairchem_esen_so3_to_grid_forward, fairchem_esen_so3_to_grid_vjp, EsenSo3GridConfig,
};
use crate::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenGateActivationConfig {
    l_max: usize,
    m_max: usize,
    channels: usize,
}

impl EsenGateActivationConfig {
    pub fn new(l_max: usize, m_max: usize, channels: usize) -> Result<Self> {
        if l_max == 0 {
            return Err(invalid_activation(
                "l_max must be positive for gate activation",
            ));
        }
        if m_max > l_max {
            return Err(invalid_activation(format!(
                "m_max must be <= l_max, got {m_max} > {l_max}"
            )));
        }
        if channels == 0 {
            return Err(invalid_activation("channels must be nonzero"));
        }
        reduced_coefficient_count(l_max, m_max)
            .and_then(|count| count.checked_mul(channels))
            .ok_or_else(|| invalid_activation("gate activation dimensions overflow"))?;
        Ok(Self {
            l_max,
            m_max,
            channels,
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
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        (0..=self.l_max)
            .map(|degree| 2 * self.m_max.min(degree) + 1)
            .sum()
    }

    #[must_use]
    pub fn node_stride(&self) -> usize {
        self.coefficient_count() * self.channels
    }

    #[must_use]
    pub fn gate_node_stride(&self) -> usize {
        self.l_max * self.channels
    }
}

/// FairChem eSEN `GateActivation` forward pass.
///
/// `input_tensors` is flat `[nodes, reduced_coefficients, channels]`, where
/// `reduced_coefficients = sum_l (2 * min(l, mmax) + 1)`. `gating_scalars` is
/// flat `[nodes, lmax * channels]` and is interpreted as pre-sigmoid gates.
pub fn fairchem_esen_gate_activation_forward(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_gate_activation_shapes(
        "fairchem_esen_gate_activation_forward",
        config,
        gating_scalars,
        input_tensors,
        out,
    )?;
    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            out[index] = silu(input_tensors[index]);
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = sigmoid(gating_scalars[gate_index]);
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    out[tensor_index + component * config.channels] =
                        input_tensors[tensor_index + component * config.channels] * gate;
                }
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

/// FairChem eSEN `GateActivation` forward pass, also caching the sigmoid gates
/// and scalar SiLU derivatives needed by the VJP.
pub fn fairchem_esen_gate_activation_forward_with_cache(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    out: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
) -> Result<()> {
    let nodes = check_gate_activation_shapes(
        "fairchem_esen_gate_activation_forward_with_cache",
        config,
        gating_scalars,
        input_tensors,
        out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_cache gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_cache scalar_silu_derivatives",
        nodes * config.channels,
        scalar_silu_derivatives.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_forward_with_cache_parallel(
            config,
            gating_scalars,
            input_tensors,
            out,
            gate_sigmoids,
            scalar_silu_derivatives,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_derivative_node_start = node_index * config.channels;
        gate_activation_forward_with_cache_node(
            config,
            &gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
            &input_tensors[tensor_node_start..tensor_node_start + config.node_stride()],
            &mut out[tensor_node_start..tensor_node_start + config.node_stride()],
            &mut gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut scalar_silu_derivatives
                [scalar_derivative_node_start..scalar_derivative_node_start + config.channels],
        );
    }
    Ok(())
}

/// FairChem eSEN `GateActivation` forward pass, caching only sigmoid gates.
///
/// This is useful for fixed-weight inference paths that replay VJPs: the
/// sigmoid gates are relatively expensive to recompute and needed for all
/// non-scalar coefficients, while scalar SiLU derivatives can be recovered from
/// the cached input tensors.
pub fn fairchem_esen_gate_activation_forward_with_sigmoid_cache(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    out: &mut [f64],
    gate_sigmoids: &mut [f64],
) -> Result<()> {
    let nodes = check_gate_activation_shapes(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache",
        config,
        gating_scalars,
        input_tensors,
        out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_forward_with_sigmoid_cache_parallel(
            config,
            gating_scalars,
            input_tensors,
            out,
            gate_sigmoids,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        gate_activation_forward_with_sigmoid_cache_node(
            config,
            &gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
            &input_tensors[tensor_node_start..tensor_node_start + config.node_stride()],
            &mut out[tensor_node_start..tensor_node_start + config.node_stride()],
            &mut gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
        );
    }
    Ok(())
}

/// In-place variant of
/// [`fairchem_esen_gate_activation_forward_with_sigmoid_cache`].
///
/// `input_output_tensors` is overwritten with the activated output. The scalar
/// l=0 inputs are copied to `scalar_inputs` before SiLU is applied so the VJP
/// can later recover scalar SiLU derivatives without retaining the full
/// pre-activation tensor.
pub fn fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_output_tensors: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_inputs: &mut [f64],
) -> Result<()> {
    let node_stride = config.node_stride();
    if !input_output_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place input_output_tensors length {} is not a multiple of node stride {node_stride}",
            input_output_tensors.len()
        )));
    }
    let nodes = input_output_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place gating_scalars",
        nodes * config.gate_node_stride(),
        gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place scalar_inputs",
        nodes * config.channels,
        scalar_inputs.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_parallel(
            config,
            gating_scalars,
            input_output_tensors,
            gate_sigmoids,
            scalar_inputs,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_node_start = node_index * config.channels;
        gate_activation_forward_with_sigmoid_cache_in_place_node(
            config,
            &gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut input_output_tensors[tensor_node_start..tensor_node_start + node_stride],
            &mut gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut scalar_inputs[scalar_node_start..scalar_node_start + config.channels],
        );
    }
    Ok(())
}

/// In-place sigmoid-cache gate activation for coefficient tensors stored in a
/// non-l-major order.
///
/// `coefficient_storage_order[l_major_coefficient]` gives the coefficient row
/// used in `input_output_tensors`. This lets fixed eSEN hot paths keep SO2
/// outputs in m-major order across gate activation instead of scattering back
/// to l-major only to gather again for the next SO2 convolution.
pub fn fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    gating_scalars: &[f64],
    input_output_tensors: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_inputs: &mut [f64],
) -> Result<()> {
    check_coefficient_storage_order(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted coefficient_storage_order",
        config,
        coefficient_storage_order,
    )?;
    let node_stride = config.node_stride();
    if !input_output_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted input_output_tensors length {} is not a multiple of node stride {node_stride}",
            input_output_tensors.len()
        )));
    }
    let nodes = input_output_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted gating_scalars",
        nodes * config.gate_node_stride(),
        gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted scalar_inputs",
        nodes * config.channels,
        scalar_inputs.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted_parallel(
            config,
            coefficient_storage_order,
            gating_scalars,
            input_output_tensors,
            gate_sigmoids,
            scalar_inputs,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_node_start = node_index * config.channels;
        gate_activation_forward_with_sigmoid_cache_in_place_permuted_node(
            config,
            coefficient_storage_order,
            &gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut input_output_tensors[tensor_node_start..tensor_node_start + node_stride],
            &mut gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut scalar_inputs[scalar_node_start..scalar_node_start + config.channels],
        );
    }
    Ok(())
}

#[inline]
fn gate_activation_forward_with_cache_node(
    config: EsenGateActivationConfig,
    gate_node: &[f64],
    input_node: &[f64],
    out_node: &mut [f64],
    gate_sigmoid_node: &mut [f64],
    scalar_derivative_node: &mut [f64],
) {
    let channels = config.channels;
    for channel in 0..channels {
        let (value, derivative) = silu_value_and_derivative(input_node[channel]);
        out_node[channel] = value;
        scalar_derivative_node[channel] = derivative;
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        let gate_start = (degree - 1) * channels;
        let gate_sigmoid_degree = &mut gate_sigmoid_node[gate_start..gate_start + channels];
        for channel in 0..channels {
            gate_sigmoid_degree[channel] = sigmoid(gate_node[gate_start + channel]);
        }
        for component in 0..m_components {
            let component_start = (coefficient + component) * channels;
            let input_component = &input_node[component_start..component_start + channels];
            let out_component = &mut out_node[component_start..component_start + channels];
            for channel in 0..channels {
                out_component[channel] = input_component[channel] * gate_sigmoid_degree[channel];
            }
        }
        coefficient += m_components;
    }
}

fn gate_activation_forward_with_sigmoid_cache_node(
    config: EsenGateActivationConfig,
    gate_node: &[f64],
    input_node: &[f64],
    out_node: &mut [f64],
    gate_sigmoid_node: &mut [f64],
) {
    let channels = config.channels;
    for channel in 0..channels {
        out_node[channel] = silu(input_node[channel]);
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        let gate_start = (degree - 1) * channels;
        let gate_sigmoid_degree = &mut gate_sigmoid_node[gate_start..gate_start + channels];
        for channel in 0..channels {
            gate_sigmoid_degree[channel] = sigmoid(gate_node[gate_start + channel]);
        }
        for component in 0..m_components {
            let component_start = (coefficient + component) * channels;
            let input_component = &input_node[component_start..component_start + channels];
            let out_component = &mut out_node[component_start..component_start + channels];
            for channel in 0..channels {
                out_component[channel] = input_component[channel] * gate_sigmoid_degree[channel];
            }
        }
        coefficient += m_components;
    }
}

fn gate_activation_forward_with_sigmoid_cache_in_place_node(
    config: EsenGateActivationConfig,
    gate_node: &[f64],
    input_output_node: &mut [f64],
    gate_sigmoid_node: &mut [f64],
    scalar_input_node: &mut [f64],
) {
    let channels = config.channels;
    for channel in 0..channels {
        let input = input_output_node[channel];
        scalar_input_node[channel] = input;
        input_output_node[channel] = silu(input);
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        let gate_start = (degree - 1) * channels;
        let gate_sigmoid_degree = &mut gate_sigmoid_node[gate_start..gate_start + channels];
        for channel in 0..channels {
            gate_sigmoid_degree[channel] = sigmoid(gate_node[gate_start + channel]);
        }
        for component in 0..m_components {
            let component_start = (coefficient + component) * channels;
            let input_output_component =
                &mut input_output_node[component_start..component_start + channels];
            for channel in 0..channels {
                input_output_component[channel] *= gate_sigmoid_degree[channel];
            }
        }
        coefficient += m_components;
    }
}

fn gate_activation_forward_with_sigmoid_cache_in_place_permuted_node(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    gate_node: &[f64],
    input_output_node: &mut [f64],
    gate_sigmoid_node: &mut [f64],
    scalar_input_node: &mut [f64],
) {
    let channels = config.channels;
    let scalar_start = coefficient_storage_order[0] * channels;
    for (channel, scalar_input) in scalar_input_node.iter_mut().enumerate().take(channels) {
        let index = scalar_start + channel;
        let input = input_output_node[index];
        *scalar_input = input;
        input_output_node[index] = silu(input);
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        let gate_start = (degree - 1) * channels;
        let gate_sigmoid_degree = &mut gate_sigmoid_node[gate_start..gate_start + channels];
        for channel in 0..channels {
            gate_sigmoid_degree[channel] = sigmoid(gate_node[gate_start + channel]);
        }
        for component in 0..m_components {
            let storage_coefficient = coefficient_storage_order[coefficient + component];
            let component_start = storage_coefficient * channels;
            let input_output_component =
                &mut input_output_node[component_start..component_start + channels];
            for channel in 0..channels {
                input_output_component[channel] *= gate_sigmoid_degree[channel];
            }
        }
        coefficient += m_components;
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_forward_with_cache_parallel(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    out: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_silu_derivatives: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    gating_scalars
        .par_chunks(gate_stride)
        .zip(input_tensors.par_chunks(node_stride))
        .zip(out.par_chunks_mut(node_stride))
        .zip(gate_sigmoids.par_chunks_mut(gate_stride))
        .zip(scalar_silu_derivatives.par_chunks_mut(channels))
        .for_each(
            |((((gate_node, input_node), out_node), gate_sigmoid_node), scalar_derivative_node)| {
                gate_activation_forward_with_cache_node(
                    config,
                    gate_node,
                    input_node,
                    out_node,
                    gate_sigmoid_node,
                    scalar_derivative_node,
                );
            },
        );
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_forward_with_sigmoid_cache_parallel(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    out: &mut [f64],
    gate_sigmoids: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    gating_scalars
        .par_chunks(gate_stride)
        .zip(input_tensors.par_chunks(node_stride))
        .zip(out.par_chunks_mut(node_stride))
        .zip(gate_sigmoids.par_chunks_mut(gate_stride))
        .for_each(|(((gate_node, input_node), out_node), gate_sigmoid_node)| {
            gate_activation_forward_with_sigmoid_cache_node(
                config,
                gate_node,
                input_node,
                out_node,
                gate_sigmoid_node,
            );
        });
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_parallel(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_output_tensors: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_inputs: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    gating_scalars
        .par_chunks(gate_stride)
        .zip(input_output_tensors.par_chunks_mut(node_stride))
        .zip(gate_sigmoids.par_chunks_mut(gate_stride))
        .zip(scalar_inputs.par_chunks_mut(channels))
        .for_each(
            |(((gate_node, input_output_node), gate_sigmoid_node), scalar_input_node)| {
                gate_activation_forward_with_sigmoid_cache_in_place_node(
                    config,
                    gate_node,
                    input_output_node,
                    gate_sigmoid_node,
                    scalar_input_node,
                );
            },
        );
}

/// Accumulate input-tensor and gate-scalar VJPs of
/// [`fairchem_esen_gate_activation_forward`].
pub fn fairchem_esen_gate_activation_vjp(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    grad_out: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) -> Result<()> {
    let nodes = check_gate_activation_shapes(
        "fairchem_esen_gate_activation_vjp",
        config,
        gating_scalars,
        input_tensors,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp grad_gating_scalars",
        gating_scalars.len(),
        grad_gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp grad_input_tensors",
        input_tensors.len(),
        grad_input_tensors.len(),
    )?;

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            grad_input_tensors[index] += grad_out[index] * silu_derivative(input_tensors[index]);
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = sigmoid(gating_scalars[gate_index]);
                let mut grad_gate = 0.0;
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    let component_index = tensor_index + component * config.channels;
                    let grad = grad_out[component_index];
                    grad_input_tensors[component_index] += grad * gate;
                    grad_gate += grad * input_tensors[component_index];
                }
                grad_gating_scalars[gate_index] += grad_gate * gate * (1.0 - gate);
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

/// Accumulate VJPs using cached sigmoid gates and recomputed scalar SiLU
/// derivatives.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache(
    config: EsenGateActivationConfig,
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    grad_out: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) -> Result<()> {
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache input_tensors",
        input_tensors.len(),
        grad_out.len(),
    )?;
    let node_stride = config.node_stride();
    if !input_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_vjp_with_sigmoid_cache input_tensors length {} is not a multiple of node stride {node_stride}",
            input_tensors.len()
        )));
    }
    let nodes = input_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache gate_sigmoids",
        nodes * config.gate_node_stride(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache grad_gating_scalars",
        gate_sigmoids.len(),
        grad_gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache grad_input_tensors",
        input_tensors.len(),
        grad_input_tensors.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_vjp_with_sigmoid_cache_parallel(
            config,
            input_tensors,
            gate_sigmoids,
            grad_out,
            grad_gating_scalars,
            grad_input_tensors,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            grad_input_tensors[index] += grad_out[index] * silu_derivative(input_tensors[index]);
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = gate_sigmoids[gate_index];
                let mut grad_gate = 0.0;
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    let component_index = tensor_index + component * config.channels;
                    let grad = grad_out[component_index];
                    grad_input_tensors[component_index] += grad * gate;
                    grad_gate += grad * input_tensors[component_index];
                }
                grad_gating_scalars[gate_index] += grad_gate * gate * (1.0 - gate);
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

/// Overwrite `grad_out_input_tensors` with the input-tensor VJP of
/// [`fairchem_esen_gate_activation_forward_with_sigmoid_cache`].
///
/// This is equivalent to calling
/// [`fairchem_esen_gate_activation_vjp_with_sigmoid_cache`] with a zeroed
/// `grad_input_tensors`, then replacing `grad_out_input_tensors` by that
/// result. It is intended for fixed-weight inference VJP chains where the
/// activation output cotangent is dead after gate VJP.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place(
    config: EsenGateActivationConfig,
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) -> Result<()> {
    let node_stride = config.node_stride();
    if !input_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place input_tensors length {} is not a multiple of node stride {node_stride}",
            input_tensors.len()
        )));
    }
    let nodes = input_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place grad_out_input_tensors",
        input_tensors.len(),
        grad_out_input_tensors.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place gate_sigmoids",
        nodes * config.gate_node_stride(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place grad_gating_scalars",
        gate_sigmoids.len(),
        grad_gating_scalars.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place_parallel(
            config,
            input_tensors,
            gate_sigmoids,
            grad_out_input_tensors,
            grad_gating_scalars,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            grad_out_input_tensors[index] *= silu_derivative(input_tensors[index]);
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = gate_sigmoids[gate_index];
                let mut grad_gate = 0.0;
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    let component_index = tensor_index + component * config.channels;
                    let grad = grad_out_input_tensors[component_index];
                    grad_out_input_tensors[component_index] = grad * gate;
                    grad_gate += grad * input_tensors[component_index];
                }
                grad_gating_scalars[gate_index] += grad_gate * gate * (1.0 - gate);
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

/// In-place VJP for
/// [`fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place`].
///
/// The full pre-activation tensor is not required. Scalar derivatives are
/// recovered from `scalar_inputs`; non-scalar raw-gate cotangents use
/// `activated * (1 - sigmoid(gate_raw))`, which is algebraically equivalent to
/// `input * sigmoid(gate_raw) * (1 - sigmoid(gate_raw))`.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place(
    config: EsenGateActivationConfig,
    activated_tensors: &[f64],
    scalar_inputs: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) -> Result<()> {
    let node_stride = config.node_stride();
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place grad_out_input_tensors",
        activated_tensors.len(),
        grad_out_input_tensors.len(),
    )?;
    if !activated_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place activated_tensors length {} is not a multiple of node stride {node_stride}",
            activated_tensors.len()
        )));
    }
    let nodes = activated_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place scalar_inputs",
        nodes * config.channels,
        scalar_inputs.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place gate_sigmoids",
        nodes * config.gate_node_stride(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place grad_gating_scalars",
        gate_sigmoids.len(),
        grad_gating_scalars.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_parallel(
            config,
            activated_tensors,
            scalar_inputs,
            gate_sigmoids,
            grad_out_input_tensors,
            grad_gating_scalars,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_node_start = node_index * config.channels;
        gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_node(
            config,
            &activated_tensors[tensor_node_start..tensor_node_start + node_stride],
            &scalar_inputs[scalar_node_start..scalar_node_start + config.channels],
            &gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut grad_out_input_tensors[tensor_node_start..tensor_node_start + node_stride],
            &mut grad_gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
        );
    }
    Ok(())
}

/// In-place activated-tensor VJP for coefficient tensors stored in a
/// non-l-major order.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    activated_tensors: &[f64],
    scalar_inputs: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) -> Result<()> {
    check_coefficient_storage_order(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted coefficient_storage_order",
        config,
        coefficient_storage_order,
    )?;
    let node_stride = config.node_stride();
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted grad_out_input_tensors",
        activated_tensors.len(),
        grad_out_input_tensors.len(),
    )?;
    if !activated_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted activated_tensors length {} is not a multiple of node stride {node_stride}",
            activated_tensors.len()
        )));
    }
    let nodes = activated_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted scalar_inputs",
        nodes * config.channels,
        scalar_inputs.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted gate_sigmoids",
        nodes * config.gate_node_stride(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted grad_gating_scalars",
        gate_sigmoids.len(),
        grad_gating_scalars.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted_parallel(
            config,
            coefficient_storage_order,
            activated_tensors,
            scalar_inputs,
            gate_sigmoids,
            grad_out_input_tensors,
            grad_gating_scalars,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * node_stride;
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_node_start = node_index * config.channels;
        gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted_node(
            config,
            coefficient_storage_order,
            &activated_tensors[tensor_node_start..tensor_node_start + node_stride],
            &scalar_inputs[scalar_node_start..scalar_node_start + config.channels],
            &gate_sigmoids[gate_node_start..gate_node_start + config.gate_node_stride()],
            &mut grad_out_input_tensors[tensor_node_start..tensor_node_start + node_stride],
            &mut grad_gating_scalars[gate_node_start..gate_node_start + config.gate_node_stride()],
        );
    }
    Ok(())
}

/// Accumulate VJPs of [`fairchem_esen_gate_activation_forward_with_cache`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_cache(
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    grad_out: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) -> Result<()> {
    let nodes = check_gate_activation_shapes(
        "fairchem_esen_gate_activation_vjp_with_cache",
        config,
        gating_scalars,
        input_tensors,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache scalar_silu_derivatives",
        nodes * config.channels,
        scalar_silu_derivatives.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache grad_gating_scalars",
        gating_scalars.len(),
        grad_gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache grad_input_tensors",
        input_tensors.len(),
        grad_input_tensors.len(),
    )?;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && nodes >= crate::esen_parallel_min_nodes() {
        fairchem_esen_gate_activation_vjp_with_cache_parallel(
            config,
            input_tensors,
            gate_sigmoids,
            scalar_silu_derivatives,
            grad_out,
            grad_gating_scalars,
            grad_input_tensors,
        );
        return Ok(());
    }

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_derivative_node_start = node_index * config.channels;
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            grad_input_tensors[index] +=
                grad_out[index] * scalar_silu_derivatives[scalar_derivative_node_start + channel];
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = gate_sigmoids[gate_index];
                let mut grad_gate = 0.0;
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    let component_index = tensor_index + component * config.channels;
                    let grad = grad_out[component_index];
                    grad_input_tensors[component_index] += grad * gate;
                    grad_gate += grad * input_tensors[component_index];
                }
                grad_gating_scalars[gate_index] += grad_gate * gate * (1.0 - gate);
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

fn gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_node(
    config: EsenGateActivationConfig,
    activated_node: &[f64],
    scalar_input_node: &[f64],
    gate_sigmoid_node: &[f64],
    grad_out_input_node: &mut [f64],
    grad_gating_node: &mut [f64],
) {
    let channels = config.channels;
    for channel in 0..channels {
        grad_out_input_node[channel] *= silu_derivative(scalar_input_node[channel]);
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        for channel in 0..channels {
            let gate_index = (degree - 1) * channels + channel;
            let gate = gate_sigmoid_node[gate_index];
            let mut grad_gate = 0.0;
            for component in 0..m_components {
                let tensor_index = coefficient * channels + channel;
                let component_index = tensor_index + component * channels;
                let grad = grad_out_input_node[component_index];
                grad_out_input_node[component_index] = grad * gate;
                grad_gate += grad * activated_node[component_index];
            }
            grad_gating_node[gate_index] += grad_gate * (1.0 - gate);
        }
        coefficient += m_components;
    }
}

fn gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted_node(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    activated_node: &[f64],
    scalar_input_node: &[f64],
    gate_sigmoid_node: &[f64],
    grad_out_input_node: &mut [f64],
    grad_gating_node: &mut [f64],
) {
    let channels = config.channels;
    let scalar_start = coefficient_storage_order[0] * channels;
    for (channel, scalar_input) in scalar_input_node.iter().enumerate().take(channels) {
        let index = scalar_start + channel;
        grad_out_input_node[index] *= silu_derivative(*scalar_input);
    }

    let mut coefficient = 1usize;
    for degree in 1..=config.l_max {
        let m_components = 2 * config.m_max.min(degree) + 1;
        for channel in 0..channels {
            let gate_index = (degree - 1) * channels + channel;
            let gate = gate_sigmoid_node[gate_index];
            let mut grad_gate = 0.0;
            for component in 0..m_components {
                let storage_coefficient = coefficient_storage_order[coefficient + component];
                let component_index = storage_coefficient * channels + channel;
                let grad = grad_out_input_node[component_index];
                grad_out_input_node[component_index] = grad * gate;
                grad_gate += grad * activated_node[component_index];
            }
            grad_gating_node[gate_index] += grad_gate * (1.0 - gate);
        }
        coefficient += m_components;
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_vjp_with_cache_parallel(
    config: EsenGateActivationConfig,
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    scalar_silu_derivatives: &[f64],
    grad_out: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    input_tensors
        .par_chunks(node_stride)
        .zip(gate_sigmoids.par_chunks(gate_stride))
        .zip(scalar_silu_derivatives.par_chunks(channels))
        .zip(grad_out.par_chunks(node_stride))
        .zip(grad_gating_scalars.par_chunks_mut(gate_stride))
        .zip(grad_input_tensors.par_chunks_mut(node_stride))
        .for_each(
            |(
                (
                    (((input_node, gate_sigmoid_node), scalar_derivative_node), grad_out_node),
                    grad_gating_node,
                ),
                grad_input_node,
            )| {
                for channel in 0..channels {
                    grad_input_node[channel] +=
                        grad_out_node[channel] * scalar_derivative_node[channel];
                }

                let mut coefficient = 1usize;
                for degree in 1..=config.l_max {
                    let m_components = 2 * config.m_max.min(degree) + 1;
                    for channel in 0..channels {
                        let gate_index = (degree - 1) * channels + channel;
                        let gate = gate_sigmoid_node[gate_index];
                        let mut grad_gate = 0.0;
                        for component in 0..m_components {
                            let tensor_index = coefficient * channels + channel;
                            let component_index = tensor_index + component * channels;
                            let grad = grad_out_node[component_index];
                            grad_input_node[component_index] += grad * gate;
                            grad_gate += grad * input_node[component_index];
                        }
                        grad_gating_node[gate_index] += grad_gate * gate * (1.0 - gate);
                    }
                    coefficient += m_components;
                }
            },
        );
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_parallel(
    config: EsenGateActivationConfig,
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    grad_out: &[f64],
    grad_gating_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    input_tensors
        .par_chunks(node_stride)
        .zip(gate_sigmoids.par_chunks(gate_stride))
        .zip(grad_out.par_chunks(node_stride))
        .zip(grad_gating_scalars.par_chunks_mut(gate_stride))
        .zip(grad_input_tensors.par_chunks_mut(node_stride))
        .for_each(
            |(
                (((input_node, gate_sigmoid_node), grad_out_node), grad_gating_node),
                grad_input_node,
            )| {
                for channel in 0..channels {
                    grad_input_node[channel] +=
                        grad_out_node[channel] * silu_derivative(input_node[channel]);
                }

                let mut coefficient = 1usize;
                for degree in 1..=config.l_max {
                    let m_components = 2 * config.m_max.min(degree) + 1;
                    for channel in 0..channels {
                        let gate_index = (degree - 1) * channels + channel;
                        let gate = gate_sigmoid_node[gate_index];
                        let mut grad_gate = 0.0;
                        for component in 0..m_components {
                            let tensor_index = coefficient * channels + channel;
                            let component_index = tensor_index + component * channels;
                            let grad = grad_out_node[component_index];
                            grad_input_node[component_index] += grad * gate;
                            grad_gate += grad * input_node[component_index];
                        }
                        grad_gating_node[gate_index] += grad_gate * gate * (1.0 - gate);
                    }
                    coefficient += m_components;
                }
            },
        );
}

#[cfg(feature = "parallel")]
fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place_parallel(
    config: EsenGateActivationConfig,
    input_tensors: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    input_tensors
        .par_chunks(node_stride)
        .zip(gate_sigmoids.par_chunks(gate_stride))
        .zip(grad_out_input_tensors.par_chunks_mut(node_stride))
        .zip(grad_gating_scalars.par_chunks_mut(gate_stride))
        .for_each(
            |(((input_node, gate_sigmoid_node), grad_out_input_node), grad_gating_node)| {
                for channel in 0..channels {
                    grad_out_input_node[channel] *= silu_derivative(input_node[channel]);
                }

                let mut coefficient = 1usize;
                for degree in 1..=config.l_max {
                    let m_components = 2 * config.m_max.min(degree) + 1;
                    for channel in 0..channels {
                        let gate_index = (degree - 1) * channels + channel;
                        let gate = gate_sigmoid_node[gate_index];
                        let mut grad_gate = 0.0;
                        for component in 0..m_components {
                            let tensor_index = coefficient * channels + channel;
                            let component_index = tensor_index + component * channels;
                            let grad = grad_out_input_node[component_index];
                            grad_out_input_node[component_index] = grad * gate;
                            grad_gate += grad * input_node[component_index];
                        }
                        grad_gating_node[gate_index] += grad_gate * gate * (1.0 - gate);
                    }
                    coefficient += m_components;
                }
            },
        );
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_parallel(
    config: EsenGateActivationConfig,
    activated_tensors: &[f64],
    scalar_inputs: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    activated_tensors
        .par_chunks(node_stride)
        .zip(scalar_inputs.par_chunks(channels))
        .zip(gate_sigmoids.par_chunks(gate_stride))
        .zip(grad_out_input_tensors.par_chunks_mut(node_stride))
        .zip(grad_gating_scalars.par_chunks_mut(gate_stride))
        .for_each(
            |(
                (((activated_node, scalar_input_node), gate_sigmoid_node), grad_out_input_node),
                grad_gating_node,
            )| {
                gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_node(
                    config,
                    activated_node,
                    scalar_input_node,
                    gate_sigmoid_node,
                    grad_out_input_node,
                    grad_gating_node,
                );
            },
        );
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted_parallel(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    activated_tensors: &[f64],
    scalar_inputs: &[f64],
    gate_sigmoids: &[f64],
    grad_out_input_tensors: &mut [f64],
    grad_gating_scalars: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    activated_tensors
        .par_chunks(node_stride)
        .zip(scalar_inputs.par_chunks(channels))
        .zip(gate_sigmoids.par_chunks(gate_stride))
        .zip(grad_out_input_tensors.par_chunks_mut(node_stride))
        .zip(grad_gating_scalars.par_chunks_mut(gate_stride))
        .for_each(
            |(
                (((activated_node, scalar_input_node), gate_sigmoid_node), grad_out_input_node),
                grad_gating_node,
            )| {
                gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted_node(
                    config,
                    coefficient_storage_order,
                    activated_node,
                    scalar_input_node,
                    gate_sigmoid_node,
                    grad_out_input_node,
                    grad_gating_node,
                );
            },
        );
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted_parallel(
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
    gating_scalars: &[f64],
    input_output_tensors: &mut [f64],
    gate_sigmoids: &mut [f64],
    scalar_inputs: &mut [f64],
) {
    let node_stride = config.node_stride();
    let gate_stride = config.gate_node_stride();
    let channels = config.channels;
    gating_scalars
        .par_chunks(gate_stride)
        .zip(input_output_tensors.par_chunks_mut(node_stride))
        .zip(gate_sigmoids.par_chunks_mut(gate_stride))
        .zip(scalar_inputs.par_chunks_mut(channels))
        .for_each(
            |(((gate_node, input_output_node), gate_sigmoid_node), scalar_input_node)| {
                gate_activation_forward_with_sigmoid_cache_in_place_permuted_node(
                    config,
                    coefficient_storage_order,
                    gate_node,
                    input_output_node,
                    gate_sigmoid_node,
                    scalar_input_node,
                );
            },
        );
}

/// FairChem eSEN `SeparableS2Activation` forward pass.
///
/// `input_scalars` is flat `[nodes, channels]`. `input_tensors` and `out` are
/// flat `[nodes, coefficients, channels]`. The scalar output coefficient is
/// `SiLU(input_scalars)`, while non-scalar tensor coefficients come from
/// `SO3_Grid.from_grid(SiLU(SO3_Grid.to_grid(input_tensors)))`.
pub fn fairchem_esen_separable_s2_activation_forward(
    config: EsenSo3GridConfig,
    input_scalars: &[f64],
    input_tensors: &[f64],
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let nodes = check_separable_s2_activation_shapes(
        "fairchem_esen_separable_s2_activation_forward",
        config,
        input_scalars,
        input_tensors,
        out,
    )?;
    let mut grid = vec![0.0; nodes * config.grid_node_stride()];
    fairchem_esen_so3_to_grid_forward(config, input_tensors, to_grid_mat, &mut grid)?;
    for value in &mut grid {
        *value = silu(*value);
    }
    let mut tensor_out = vec![0.0; input_tensors.len()];
    fairchem_esen_so3_from_grid_forward(config, &grid, from_grid_mat, &mut tensor_out)?;

    for node in 0..nodes {
        let scalar_start = node * config.channels();
        let node_start = node * config.coefficient_node_stride();
        for channel in 0..config.channels() {
            out[node_start + channel] = silu(input_scalars[scalar_start + channel]);
        }
        let vector_start = node_start + config.channels();
        let vector_end = node_start + config.coefficient_node_stride();
        out[vector_start..vector_end].copy_from_slice(&tensor_out[vector_start..vector_end]);
    }
    Ok(())
}

/// Accumulate input-scalar and input-tensor VJPs of
/// [`fairchem_esen_separable_s2_activation_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_separable_s2_activation_vjp(
    config: EsenSo3GridConfig,
    input_scalars: &[f64],
    input_tensors: &[f64],
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    grad_out: &[f64],
    grad_input_scalars: &mut [f64],
    grad_input_tensors: &mut [f64],
) -> Result<()> {
    let nodes = check_separable_s2_activation_shapes(
        "fairchem_esen_separable_s2_activation_vjp",
        config,
        input_scalars,
        input_tensors,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_separable_s2_activation_vjp grad_input_scalars",
        input_scalars.len(),
        grad_input_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_separable_s2_activation_vjp grad_input_tensors",
        input_tensors.len(),
        grad_input_tensors.len(),
    )?;

    let mut grid = vec![0.0; nodes * config.grid_node_stride()];
    fairchem_esen_so3_to_grid_forward(config, input_tensors, to_grid_mat, &mut grid)?;

    let mut grad_tensor_out = vec![0.0; input_tensors.len()];
    for node in 0..nodes {
        let scalar_start = node * config.channels();
        let node_start = node * config.coefficient_node_stride();
        for channel in 0..config.channels() {
            let scalar_value = input_scalars[scalar_start + channel];
            grad_input_scalars[scalar_start + channel] +=
                grad_out[node_start + channel] * silu_derivative(scalar_value);
        }
        let vector_start = node_start + config.channels();
        let vector_end = node_start + config.coefficient_node_stride();
        grad_tensor_out[vector_start..vector_end]
            .copy_from_slice(&grad_out[vector_start..vector_end]);
    }

    let mut grad_grid = vec![0.0; grid.len()];
    fairchem_esen_so3_from_grid_vjp(config, from_grid_mat, &grad_tensor_out, &mut grad_grid)?;
    for (grad, value) in grad_grid.iter_mut().zip(grid.iter()) {
        *grad *= silu_derivative(*value);
    }
    fairchem_esen_so3_to_grid_vjp(config, to_grid_mat, &grad_grid, grad_input_tensors)
}

fn check_gate_activation_shapes(
    op: &'static str,
    config: EsenGateActivationConfig,
    gating_scalars: &[f64],
    input_tensors: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    check_len(op, input_tensors.len(), output_like.len())?;
    let node_stride = config.node_stride();
    if !input_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "{op} input_tensors length {} is not a multiple of node stride {node_stride}",
            input_tensors.len()
        )));
    }
    let nodes = input_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation gating_scalars",
        nodes * config.gate_node_stride(),
        gating_scalars.len(),
    )?;
    Ok(nodes)
}

fn check_separable_s2_activation_shapes(
    op: &'static str,
    config: EsenSo3GridConfig,
    input_scalars: &[f64],
    input_tensors: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    check_len(op, input_tensors.len(), output_like.len())?;
    let tensor_stride = config.coefficient_node_stride();
    if !input_tensors.len().is_multiple_of(tensor_stride) {
        return Err(invalid_activation(format!(
            "{op} input_tensors length {} is not a multiple of node stride {tensor_stride}",
            input_tensors.len()
        )));
    }
    let nodes = input_tensors.len() / tensor_stride;
    check_len(
        "fairchem_esen_separable_s2_activation input_scalars",
        nodes * config.channels(),
        input_scalars.len(),
    )?;
    Ok(nodes)
}

fn reduced_coefficient_count(l_max: usize, m_max: usize) -> Option<usize> {
    let mut count = 0usize;
    for degree in 0..=l_max {
        count = count.checked_add(2 * m_max.min(degree) + 1)?;
    }
    Some(count)
}

fn check_coefficient_storage_order(
    op: &'static str,
    config: EsenGateActivationConfig,
    coefficient_storage_order: &[usize],
) -> Result<()> {
    let coefficient_count = config.coefficient_count();
    check_len(op, coefficient_count, coefficient_storage_order.len())?;
    let mut seen = vec![false; coefficient_count];
    for (source, &target) in coefficient_storage_order.iter().enumerate() {
        if target >= coefficient_count {
            return Err(CoreError::ShapeMismatch {
                op,
                expected: coefficient_count,
                actual: target.saturating_add(1),
            });
        }
        if seen[target] {
            return Err(invalid_activation(format!(
                "{op} must be a permutation, but storage coefficient {target} appears more than once; duplicate source coefficient {source}"
            )));
        }
        seen[target] = true;
    }
    Ok(())
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn silu_value_and_derivative(value: f64) -> (f64, f64) {
    let sig = sigmoid(value);
    (value * sig, sig * (1.0 + value * (1.0 - sig)))
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

fn invalid_activation(message: impl Into<String>) -> CoreError {
    CoreError::InvalidActivation(message.into())
}

/// f32 twin of [`fairchem_esen_gate_activation_forward_with_cache`].
///
/// Computes the gate activation in f32.
pub fn fairchem_esen_gate_activation_forward_with_cache_f32(
    config: EsenGateActivationConfig,
    gating_scalars: &[f32],
    input_tensors: &[f32],
    out: &mut [f32],
    gate_sigmoids: &mut [f32],
    scalar_silu_derivatives: &mut [f32],
) -> Result<()> {
    let nodes = check_gate_activation_shapes_f32(
        "fairchem_esen_gate_activation_forward_with_cache_f32",
        config,
        gating_scalars,
        input_tensors,
        out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_cache_f32 gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_forward_with_cache_f32 scalar_silu_derivatives",
        nodes * config.channels,
        scalar_silu_derivatives.len(),
    )?;

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_derivative_node_start = node_index * config.channels;
        let channels = config.channels;
        for channel in 0..channels {
            let index = tensor_node_start + channel;
            let (value, derivative) = silu_value_and_derivative_f32(input_tensors[index]);
            out[index] = value;
            scalar_silu_derivatives[scalar_derivative_node_start + channel] = derivative;
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            let gate_start = gate_node_start + (degree - 1) * channels;
            for channel in 0..channels {
                gate_sigmoids[gate_start + channel] =
                    sigmoid_f32(gating_scalars[gate_start + channel]);
            }
            for component in 0..m_components {
                let component_start = tensor_node_start + (coefficient + component) * channels;
                for channel in 0..channels {
                    out[component_start + channel] = input_tensors[component_start + channel]
                        * gate_sigmoids[gate_start + channel];
                }
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_gate_activation_vjp_with_cache`].
///
/// Accumulates gate activation VJPs in f32.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_gate_activation_vjp_with_cache_f32(
    config: EsenGateActivationConfig,
    gating_scalars: &[f32],
    input_tensors: &[f32],
    gate_sigmoids: &[f32],
    scalar_silu_derivatives: &[f32],
    grad_out: &[f32],
    grad_gating_scalars: &mut [f32],
    grad_input_tensors: &mut [f32],
) -> Result<()> {
    let nodes = check_gate_activation_shapes_f32(
        "fairchem_esen_gate_activation_vjp_with_cache_f32",
        config,
        gating_scalars,
        input_tensors,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache_f32 gate_sigmoids",
        gating_scalars.len(),
        gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache_f32 scalar_silu_derivatives",
        nodes * config.channels,
        scalar_silu_derivatives.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache_f32 grad_gating_scalars",
        gating_scalars.len(),
        grad_gating_scalars.len(),
    )?;
    check_len(
        "fairchem_esen_gate_activation_vjp_with_cache_f32 grad_input_tensors",
        input_tensors.len(),
        grad_input_tensors.len(),
    )?;

    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let gate_node_start = node_index * config.gate_node_stride();
        let scalar_derivative_node_start = node_index * config.channels;
        for channel in 0..config.channels {
            let index = tensor_node_start + channel;
            grad_input_tensors[index] +=
                grad_out[index] * scalar_silu_derivatives[scalar_derivative_node_start + channel];
        }

        let mut coefficient = 1usize;
        for degree in 1..=config.l_max {
            let m_components = 2 * config.m_max.min(degree) + 1;
            for channel in 0..config.channels {
                let gate_index = gate_node_start + (degree - 1) * config.channels + channel;
                let gate = gate_sigmoids[gate_index];
                let mut grad_gate = 0.0_f32;
                for component in 0..m_components {
                    let tensor_index = tensor_node_start + coefficient * config.channels + channel;
                    let component_index = tensor_index + component * config.channels;
                    let grad = grad_out[component_index];
                    grad_input_tensors[component_index] += grad * gate;
                    grad_gate += grad * input_tensors[component_index];
                }
                grad_gating_scalars[gate_index] += grad_gate * gate * (1.0 - gate);
            }
            coefficient += m_components;
        }
    }
    Ok(())
}

fn check_gate_activation_shapes_f32(
    op: &'static str,
    config: EsenGateActivationConfig,
    gating_scalars: &[f32],
    input_tensors: &[f32],
    output_like: &[f32],
) -> Result<usize> {
    check_len(op, input_tensors.len(), output_like.len())?;
    let node_stride = config.node_stride();
    if !input_tensors.len().is_multiple_of(node_stride) {
        return Err(invalid_activation(format!(
            "{op} input_tensors length {} is not a multiple of node stride {node_stride}",
            input_tensors.len()
        )));
    }
    let nodes = input_tensors.len() / node_stride;
    check_len(
        "fairchem_esen_gate_activation_f32 gating_scalars",
        nodes * config.gate_node_stride(),
        gating_scalars.len(),
    )?;
    Ok(nodes)
}

fn sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn silu_value_and_derivative_f32(value: f32) -> (f32, f32) {
    let sig = sigmoid_f32(value);
    (value * sig, sig * (1.0 + value * (1.0 - sig)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests_f32 {
    use super::*;

    fn make_vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.19 + 0.3).sin()).collect()
    }

    #[test]
    fn gate_activation_forward_with_cache_f32_vs_f64() {
        // l_max=2, m_max=2, channels=4 => coefficient_count = 1+3+5=9, node_stride=36
        // gate_node_stride = l_max * channels = 2 * 4 = 8
        let config = EsenGateActivationConfig::new(2, 2, 4).unwrap();
        let nodes = 3usize;
        let n_tensors = nodes * config.node_stride();
        let n_gates = nodes * config.gate_node_stride();

        let gate_f32 = make_vals(n_gates);
        let input_f32 = make_vals(n_tensors);
        let gate_f64: Vec<f64> = gate_f32.iter().map(|&x| x as f64).collect();
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0_f32; n_tensors];
        let mut gs_f32 = vec![0.0_f32; n_gates];
        let mut sd_f32 = vec![0.0_f32; nodes * config.channels];
        let mut out_f64 = vec![0.0_f64; n_tensors];
        let mut gs_f64 = vec![0.0_f64; n_gates];
        let mut sd_f64 = vec![0.0_f64; nodes * config.channels];

        fairchem_esen_gate_activation_forward_with_cache_f32(
            config,
            &gate_f32,
            &input_f32,
            &mut out_f32,
            &mut gs_f32,
            &mut sd_f32,
        )
        .unwrap();
        fairchem_esen_gate_activation_forward_with_cache(
            config,
            &gate_f64,
            &input_f64,
            &mut out_f64,
            &mut gs_f64,
            &mut sd_f64,
        )
        .unwrap();

        for (i, (&a, &b)) in out_f32.iter().zip(out_f64.iter()).enumerate() {
            let rel_err = if b.abs() > 1e-10 {
                ((a as f64 - b) / b).abs()
            } else {
                (a as f64 - b).abs()
            };
            assert!(
                rel_err < 1e-6,
                "out[{i}]: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }
}
