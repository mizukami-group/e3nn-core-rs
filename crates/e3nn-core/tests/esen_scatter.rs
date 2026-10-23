#![allow(clippy::unwrap_used)]

use e3nn_core::scatter::{
    fairchem_esen_edge_to_node_scaled_scatter_forward,
    fairchem_esen_edge_to_node_scaled_scatter_vjp_edges,
    fairchem_esen_source_target_concat_forward, fairchem_esen_source_target_concat_vjp_nodes,
};

#[test]
fn edge_to_node_scaled_scatter_accumulates_duplicate_targets_with_offset() {
    let mut node_features = [1.0, 2.0, -1.0, 0.5, 4.0, -3.0];
    let edge_targets = [10, 12, 10];
    let edge_features = [2.0, 4.0, -1.0, 3.0, 6.0, -2.0];

    fairchem_esen_edge_to_node_scaled_scatter_forward(
        3,
        2,
        &edge_targets,
        10,
        2.0,
        &edge_features,
        &mut node_features,
    )
    .unwrap();

    assert_eq!(node_features, [5.0, 3.0, -1.0, 0.5, 3.5, -1.5]);
}

#[test]
fn edge_to_node_scaled_scatter_vjp_matches_finite_difference() {
    let num_nodes = 3;
    let feature_dim = 3;
    let edge_targets = [2, 4, 3, 2];
    let node_offset = 2;
    let rescale_factor = 5.0;
    let mut edge_features = vec![0.0; edge_targets.len() * feature_dim];
    for (index, value) in edge_features.iter_mut().enumerate() {
        *value = 0.3 * index as f64 - 1.2;
    }
    let mut grad_node = vec![0.0; num_nodes * feature_dim];
    for (index, value) in grad_node.iter_mut().enumerate() {
        *value = (0.23 * index as f64).sin();
    }
    let mut grad_edge = vec![0.0; edge_features.len()];

    fairchem_esen_edge_to_node_scaled_scatter_vjp_edges(
        num_nodes,
        feature_dim,
        &edge_targets,
        node_offset,
        rescale_factor,
        &grad_node,
        &mut grad_edge,
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..edge_features.len() {
        let original = edge_features[index];
        edge_features[index] = original + step;
        let plus = scalar_objective(
            num_nodes,
            feature_dim,
            &edge_targets,
            node_offset,
            rescale_factor,
            &edge_features,
            &grad_node,
        );
        edge_features[index] = original - step;
        let minus = scalar_objective(
            num_nodes,
            feature_dim,
            &edge_targets,
            node_offset,
            rescale_factor,
            &edge_features,
            &grad_node,
        );
        edge_features[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_edge[index] - fd).abs() <= 1.0e-10,
            "index={index}, grad={}, fd={fd}",
            grad_edge[index]
        );
    }
}

#[test]
fn edge_to_node_scaled_scatter_rejects_target_before_offset() {
    let mut node_features = [0.0; 2];
    let err = fairchem_esen_edge_to_node_scaled_scatter_forward(
        1,
        2,
        &[4],
        5,
        1.0,
        &[1.0, 2.0],
        &mut node_features,
    );
    assert!(err.is_err());
}

#[test]
fn source_target_concat_gathers_edge_endpoints_with_offset() {
    let edge_sources = [10, 12, 11];
    let edge_targets = [11, 10, 12];
    let node_features = [1.0, -1.0, 2.0, 0.5, -3.0, 4.0];
    let mut edge_features = [0.0; 12];

    fairchem_esen_source_target_concat_forward(
        2,
        &edge_sources,
        &edge_targets,
        10,
        &node_features,
        &mut edge_features,
    )
    .unwrap();

    assert_eq!(
        edge_features,
        [1.0, -1.0, 2.0, 0.5, -3.0, 4.0, 1.0, -1.0, 2.0, 0.5, -3.0, 4.0]
    );
}

#[test]
fn source_target_concat_vjp_accumulates_duplicate_node_endpoints() {
    let edge_sources = [0, 2, 1];
    let edge_targets = [1, 0, 2];
    let grad_edge_features = [
        1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 4.0, -0.75, 1.5, 2.5, -3.0,
    ];
    let mut grad_node_features = [0.0; 6];

    fairchem_esen_source_target_concat_vjp_nodes(
        2,
        &edge_sources,
        &edge_targets,
        0,
        &grad_edge_features,
        &mut grad_node_features,
    )
    .unwrap();

    assert_eq!(grad_node_features, [1.25, 6.0, -1.75, 2.0, 5.5, -5.0]);
}

#[test]
fn source_target_concat_vjp_self_loop_sums_source_and_target_halves() {
    // A single edge whose source and target are the same node (src == dst).
    // The node gradient must accumulate BOTH the source-half and target-half
    // edge contributions rather than have one overwrite the other.
    let edge_sources = [10];
    let edge_targets = [10];
    let grad_edge_features = [1.0, 2.0, 10.0, 20.0];
    let mut grad_node_features = [0.0; 2];

    fairchem_esen_source_target_concat_vjp_nodes(
        2,
        &edge_sources,
        &edge_targets,
        10,
        &grad_edge_features,
        &mut grad_node_features,
    )
    .unwrap();

    // source half [1.0, 2.0] + target half [10.0, 20.0].
    assert_eq!(grad_node_features, [11.0, 22.0]);
}

fn scalar_objective(
    num_nodes: usize,
    feature_dim: usize,
    edge_targets: &[usize],
    node_offset: usize,
    rescale_factor: f64,
    edge_features: &[f64],
    grad_node: &[f64],
) -> f64 {
    let mut node_features = vec![0.0; num_nodes * feature_dim];
    fairchem_esen_edge_to_node_scaled_scatter_forward(
        num_nodes,
        feature_dim,
        edge_targets,
        node_offset,
        rescale_factor,
        edge_features,
        &mut node_features,
    )
    .unwrap();
    node_features
        .iter()
        .zip(grad_node.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
