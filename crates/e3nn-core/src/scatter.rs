//! Scatter primitives used by fixed-weight equivariant models.

use crate::{CoreError, Result};

/// FairChem eSEN edge-to-target-node accumulation:
/// `node[target] += edge_feature / rescale_factor`.
pub fn fairchem_esen_edge_to_node_scaled_scatter_forward(
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    edge_features: &[f64],
    node_features: &mut [f64],
) -> Result<()> {
    check_scatter_shapes(
        "fairchem_esen_edge_to_node_scaled_scatter_forward",
        num_nodes,
        feature_dim,
        edge_targets,
        node_offset,
        rescale_factor,
        edge_features,
        node_features,
    )?;
    let scale = 1.0 / rescale_factor;
    for (edge_index, &target) in edge_targets.iter().enumerate() {
        let node = target - node_offset;
        let edge_start = edge_index * feature_dim;
        let node_start = node * feature_dim;
        for feature in 0..feature_dim {
            node_features[node_start + feature] += edge_features[edge_start + feature] * scale;
        }
    }
    Ok(())
}

/// Accumulate the edge-feature VJP of
/// [`fairchem_esen_edge_to_node_scaled_scatter_forward`] into
/// `grad_edge_features`.
pub fn fairchem_esen_edge_to_node_scaled_scatter_vjp_edges(
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    grad_node_features: &[f64],
    grad_edge_features: &mut [f64],
) -> Result<()> {
    check_scatter_shapes(
        "fairchem_esen_edge_to_node_scaled_scatter_vjp_edges",
        num_nodes,
        feature_dim,
        edge_targets,
        node_offset,
        rescale_factor,
        grad_edge_features,
        grad_node_features,
    )?;
    let scale = 1.0 / rescale_factor;
    for (edge_index, &target) in edge_targets.iter().enumerate() {
        let node = target - node_offset;
        let edge_start = edge_index * feature_dim;
        let node_start = node * feature_dim;
        for feature in 0..feature_dim {
            grad_edge_features[edge_start + feature] +=
                grad_node_features[node_start + feature] * scale;
        }
    }
    Ok(())
}

/// FairChem eSEN Edgewise source/target gather followed by channel concat.
///
/// `node_features` is flat `[nodes, feature_dim]`; `edge_features` is flat
/// `[edges, 2 * feature_dim]` with source features first and target features
/// second for each edge.
pub fn fairchem_esen_source_target_concat_forward(
    feature_dim: usize,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_offset: usize,
    node_features: &[f64],
    edge_features: &mut [f64],
) -> Result<()> {
    let nodes = check_source_target_concat_shapes(
        "fairchem_esen_source_target_concat_forward",
        feature_dim,
        edge_sources,
        edge_targets,
        node_offset,
        node_features,
        edge_features,
    )?;
    for edge_index in 0..edge_sources.len() {
        let source = edge_sources[edge_index] - node_offset;
        let target = edge_targets[edge_index] - node_offset;
        debug_assert!(source < nodes);
        debug_assert!(target < nodes);
        let source_start = source * feature_dim;
        let target_start = target * feature_dim;
        let edge_start = edge_index * 2 * feature_dim;
        edge_features[edge_start..edge_start + feature_dim]
            .copy_from_slice(&node_features[source_start..source_start + feature_dim]);
        edge_features[edge_start + feature_dim..edge_start + 2 * feature_dim]
            .copy_from_slice(&node_features[target_start..target_start + feature_dim]);
    }
    Ok(())
}

/// Accumulate node-feature VJP of
/// [`fairchem_esen_source_target_concat_forward`].
pub fn fairchem_esen_source_target_concat_vjp_nodes(
    feature_dim: usize,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_offset: usize,
    grad_edge_features: &[f64],
    grad_node_features: &mut [f64],
) -> Result<()> {
    let nodes = check_source_target_concat_shapes(
        "fairchem_esen_source_target_concat_vjp_nodes",
        feature_dim,
        edge_sources,
        edge_targets,
        node_offset,
        grad_node_features,
        grad_edge_features,
    )?;
    for edge_index in 0..edge_sources.len() {
        let source = edge_sources[edge_index] - node_offset;
        let target = edge_targets[edge_index] - node_offset;
        debug_assert!(source < nodes);
        debug_assert!(target < nodes);
        let source_start = source * feature_dim;
        let target_start = target * feature_dim;
        let edge_start = edge_index * 2 * feature_dim;
        // Accumulate the source half, then the target half. These are applied
        // as two sequential slice `+=` passes so that a self-loop edge
        // (source == target, hence source_start == target_start) accumulates
        // both contributions instead of overwriting, and so that the borrow
        // checker never sees two overlapping mutable slices. Per element this
        // still performs (existing + source_contrib) + target_contrib in the
        // same order as the previous interleaved loop, so the result is
        // bit-identical.
        for (dst, src) in grad_node_features[source_start..source_start + feature_dim]
            .iter_mut()
            .zip(&grad_edge_features[edge_start..edge_start + feature_dim])
        {
            *dst += *src;
        }
        for (dst, src) in grad_node_features[target_start..target_start + feature_dim]
            .iter_mut()
            .zip(&grad_edge_features[edge_start + feature_dim..edge_start + 2 * feature_dim])
        {
            *dst += *src;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_scatter_shapes(
    op: &'static str,
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    edge_like: &[f64],
    node_like: &[f64],
) -> Result<()> {
    if feature_dim == 0 {
        return Err(invalid_scatter("feature_dim must be nonzero"));
    }
    if !rescale_factor.is_finite() || rescale_factor == 0.0 {
        return Err(invalid_scatter(format!(
            "rescale_factor must be finite and nonzero, got {rescale_factor}"
        )));
    }
    check_len(op, edge_targets.len() * feature_dim, edge_like.len())?;
    check_len(op, num_nodes * feature_dim, node_like.len())?;
    for &target in edge_targets {
        let Some(node) = target.checked_sub(node_offset) else {
            return Err(invalid_scatter(format!(
                "edge target {target} is smaller than node_offset {node_offset}"
            )));
        };
        if node >= num_nodes {
            return Err(invalid_scatter(format!(
                "edge target {target} maps to node {node}, outside num_nodes {num_nodes}"
            )));
        }
    }
    Ok(())
}

fn check_source_target_concat_shapes(
    op: &'static str,
    feature_dim: usize,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_offset: usize,
    node_like: &[f64],
    edge_like: &[f64],
) -> Result<usize> {
    if feature_dim == 0 {
        return Err(invalid_scatter("feature_dim must be nonzero"));
    }
    check_len(
        "fairchem_esen_source_target_concat edge_targets",
        edge_sources.len(),
        edge_targets.len(),
    )?;
    if !node_like.len().is_multiple_of(feature_dim) {
        return Err(invalid_scatter(format!(
            "{op} node feature length {} is not a multiple of feature_dim {feature_dim}",
            node_like.len()
        )));
    }
    let nodes = node_like.len() / feature_dim;
    check_len(op, edge_sources.len() * 2 * feature_dim, edge_like.len())?;
    for &node_index in edge_sources.iter().chain(edge_targets.iter()) {
        let Some(node) = node_index.checked_sub(node_offset) else {
            return Err(invalid_scatter(format!(
                "edge endpoint {node_index} is smaller than node_offset {node_offset}"
            )));
        };
        if node >= nodes {
            return Err(invalid_scatter(format!(
                "edge endpoint {node_index} maps to node {node}, outside num_nodes {nodes}"
            )));
        }
    }
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

fn invalid_scatter(message: impl Into<String>) -> CoreError {
    CoreError::InvalidScatter(message.into())
}

/// f32 twin of [`fairchem_esen_edge_to_node_scaled_scatter_forward`].
pub fn fairchem_esen_edge_to_node_scaled_scatter_forward_f32(
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    edge_features: &[f32],
    node_features: &mut [f32],
) -> Result<()> {
    if feature_dim == 0 {
        return Err(invalid_scatter("feature_dim must be nonzero"));
    }
    if !rescale_factor.is_finite() || rescale_factor == 0.0 {
        return Err(invalid_scatter(format!(
            "rescale_factor must be finite and nonzero, got {rescale_factor}"
        )));
    }
    check_len(
        "fairchem_esen_edge_to_node_scaled_scatter_forward_f32",
        edge_targets.len() * feature_dim,
        edge_features.len(),
    )?;
    check_len(
        "fairchem_esen_edge_to_node_scaled_scatter_forward_f32",
        num_nodes * feature_dim,
        node_features.len(),
    )?;
    for &target in edge_targets {
        let Some(node) = target.checked_sub(node_offset) else {
            return Err(invalid_scatter(format!(
                "edge target {target} is smaller than node_offset {node_offset}"
            )));
        };
        if node >= num_nodes {
            return Err(invalid_scatter(format!(
                "edge target {target} maps to node {node}, outside num_nodes {num_nodes}"
            )));
        }
    }
    let scale = (1.0 / rescale_factor) as f32;
    for (edge_index, &target) in edge_targets.iter().enumerate() {
        let node = target - node_offset;
        let edge_start = edge_index * feature_dim;
        let node_start = node * feature_dim;
        for feature in 0..feature_dim {
            node_features[node_start + feature] += edge_features[edge_start + feature] * scale;
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_edge_to_node_scaled_scatter_vjp_edges`].
pub fn fairchem_esen_edge_to_node_scaled_scatter_vjp_edges_f32(
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    grad_node_features: &[f32],
    grad_edge_features: &mut [f32],
) -> Result<()> {
    if feature_dim == 0 {
        return Err(invalid_scatter("feature_dim must be nonzero"));
    }
    if !rescale_factor.is_finite() || rescale_factor == 0.0 {
        return Err(invalid_scatter(format!(
            "rescale_factor must be finite and nonzero, got {rescale_factor}"
        )));
    }
    check_len(
        "fairchem_esen_edge_to_node_scaled_scatter_vjp_edges_f32",
        edge_targets.len() * feature_dim,
        grad_edge_features.len(),
    )?;
    check_len(
        "fairchem_esen_edge_to_node_scaled_scatter_vjp_edges_f32",
        num_nodes * feature_dim,
        grad_node_features.len(),
    )?;
    for &target in edge_targets {
        let Some(node) = target.checked_sub(node_offset) else {
            return Err(invalid_scatter(format!(
                "edge target {target} is smaller than node_offset {node_offset}"
            )));
        };
        if node >= num_nodes {
            return Err(invalid_scatter(format!(
                "edge target {target} maps to node {node}, outside num_nodes {num_nodes}"
            )));
        }
    }
    let scale = (1.0 / rescale_factor) as f32;
    for (edge_index, &target) in edge_targets.iter().enumerate() {
        let node = target - node_offset;
        let edge_start = edge_index * feature_dim;
        let node_start = node * feature_dim;
        for feature in 0..feature_dim {
            grad_edge_features[edge_start + feature] +=
                grad_node_features[node_start + feature] * scale;
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_esen_source_target_concat_forward`].
pub fn fairchem_esen_source_target_concat_forward_f32(
    feature_dim: usize,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_offset: usize,
    node_features: &[f32],
    edge_features: &mut [f32],
) -> Result<()> {
    if feature_dim == 0 {
        return Err(invalid_scatter("feature_dim must be nonzero"));
    }
    check_len(
        "fairchem_esen_source_target_concat_f32 edge_targets",
        edge_sources.len(),
        edge_targets.len(),
    )?;
    if !node_features.len().is_multiple_of(feature_dim) {
        return Err(invalid_scatter(format!(
            "fairchem_esen_source_target_concat_forward_f32 node feature length {} is not a multiple of feature_dim {feature_dim}",
            node_features.len()
        )));
    }
    let nodes = node_features.len() / feature_dim;
    check_len(
        "fairchem_esen_source_target_concat_forward_f32",
        edge_sources.len() * 2 * feature_dim,
        edge_features.len(),
    )?;
    for &node_index in edge_sources.iter().chain(edge_targets.iter()) {
        let Some(node) = node_index.checked_sub(node_offset) else {
            return Err(invalid_scatter(format!(
                "edge endpoint {node_index} is smaller than node_offset {node_offset}"
            )));
        };
        if node >= nodes {
            return Err(invalid_scatter(format!(
                "edge endpoint {node_index} maps to node {node}, outside num_nodes {nodes}"
            )));
        }
    }
    for edge_index in 0..edge_sources.len() {
        let source = edge_sources[edge_index] - node_offset;
        let target = edge_targets[edge_index] - node_offset;
        let source_start = source * feature_dim;
        let target_start = target * feature_dim;
        let edge_start = edge_index * 2 * feature_dim;
        edge_features[edge_start..edge_start + feature_dim]
            .copy_from_slice(&node_features[source_start..source_start + feature_dim]);
        edge_features[edge_start + feature_dim..edge_start + 2 * feature_dim]
            .copy_from_slice(&node_features[target_start..target_start + feature_dim]);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests_f32 {
    use super::*;

    #[test]
    fn edge_to_node_scaled_scatter_forward_f32_vs_f64() {
        // 4 nodes, 3 features, 5 edges
        let num_nodes = 4usize;
        let feature_dim = 3usize;
        let edge_targets = vec![0usize, 1, 1, 2, 3];
        let rescale_factor = 2.0_f64;
        let edge_features_f32: Vec<f32> = (0..5 * feature_dim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let edge_features_f64: Vec<f64> = edge_features_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0_f32; num_nodes * feature_dim];
        let mut out_f64 = vec![0.0_f64; num_nodes * feature_dim];
        fairchem_esen_edge_to_node_scaled_scatter_forward_f32(
            num_nodes,
            feature_dim,
            &edge_targets,
            0,
            rescale_factor,
            &edge_features_f32,
            &mut out_f32,
        )
        .unwrap();
        fairchem_esen_edge_to_node_scaled_scatter_forward(
            num_nodes,
            feature_dim,
            &edge_targets,
            0,
            rescale_factor,
            &edge_features_f64,
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
                rel_err < 1e-6,
                "index {i}: f32={a} f64={b} rel_err={rel_err}"
            );
        }
    }
}
