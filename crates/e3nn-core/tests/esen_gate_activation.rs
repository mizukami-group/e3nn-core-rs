#![allow(clippy::needless_range_loop, clippy::unwrap_used)]

use e3nn_core::activation::{
    fairchem_esen_gate_activation_forward, fairchem_esen_gate_activation_forward_with_cache,
    fairchem_esen_gate_activation_forward_with_sigmoid_cache,
    fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place,
    fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted,
    fairchem_esen_gate_activation_vjp, fairchem_esen_gate_activation_vjp_with_cache,
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache,
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place,
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted,
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place, EsenGateActivationConfig,
};
use e3nn_core::so3::{fairchem_esen_l_to_m_permutation, EsenCoefficientMappingConfig};

#[test]
fn gate_activation_forward_applies_silu_to_scalar_and_sigmoid_gates_by_degree() {
    let config = EsenGateActivationConfig::new(2, 1, 2).unwrap();
    assert_eq!(config.coefficient_count(), 7);
    let input: Vec<_> = (0..config.node_stride())
        .map(|index| index as f64 * 0.2 - 1.1)
        .collect();
    let gates = [-0.5, 0.25, 1.0, -1.5];
    let mut out = vec![0.0; input.len()];

    fairchem_esen_gate_activation_forward(config, &gates, &input, &mut out).unwrap();

    for channel in 0..2 {
        assert!((out[channel] - silu_reference(input[channel])).abs() <= 1.0e-14);
    }
    let degree_by_coefficient = [0, 1, 1, 1, 2, 2, 2];
    for coefficient in 1..config.coefficient_count() {
        let degree = degree_by_coefficient[coefficient];
        for channel in 0..config.channels() {
            let index = coefficient * config.channels() + channel;
            let gate_index = (degree - 1) * config.channels() + channel;
            let expected = input[index] * sigmoid_reference(gates[gate_index]);
            assert!((out[index] - expected).abs() <= 1.0e-14);
        }
    }
}

#[test]
fn gate_activation_vjp_matches_finite_difference_for_inputs_and_gates() {
    let config = EsenGateActivationConfig::new(3, 2, 3).unwrap();
    let mut input: Vec<_> = (0..config.node_stride())
        .map(|index| index as f64 * 0.11 - 1.7)
        .collect();
    let mut gates: Vec<_> = (0..config.gate_node_stride())
        .map(|index| index as f64 * 0.17 - 0.8)
        .collect();
    let grad_out: Vec<_> = (0..config.node_stride())
        .map(|index| (index as f64 * 0.13).sin())
        .collect();
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_gates = vec![0.0; gates.len()];

    fairchem_esen_gate_activation_vjp(
        config,
        &gates,
        &input,
        &grad_out,
        &mut grad_gates,
        &mut grad_input,
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = scalar_objective(config, &gates, &input, &grad_out);
        input[index] = original - step;
        let minus = scalar_objective(config, &gates, &input, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..gates.len() {
        let original = gates[index];
        gates[index] = original + step;
        let plus = scalar_objective(config, &gates, &input, &grad_out);
        gates[index] = original - step;
        let minus = scalar_objective(config, &gates, &input, &grad_out);
        gates[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_gates[index] - fd).abs() <= 5.0e-8,
            "gate index={index}, grad={}, fd={fd}",
            grad_gates[index]
        );
    }
}

#[test]
fn gate_activation_sigmoid_cache_matches_full_cache_vjp() {
    let config = EsenGateActivationConfig::new(3, 2, 4).unwrap();
    let nodes = 3;
    let input: Vec<_> = (0..nodes * config.node_stride())
        .map(|index| (index as f64 * 0.07).cos() - 0.3)
        .collect();
    let gates: Vec<_> = (0..nodes * config.gate_node_stride())
        .map(|index| index as f64 * 0.09 - 1.2)
        .collect();
    let grad_out: Vec<_> = (0..input.len())
        .map(|index| (index as f64 * 0.05).sin())
        .collect();

    let mut full_out = vec![0.0; input.len()];
    let mut full_gate_sigmoids = vec![0.0; gates.len()];
    let mut scalar_derivatives = vec![0.0; nodes * config.channels()];
    fairchem_esen_gate_activation_forward_with_cache(
        config,
        &gates,
        &input,
        &mut full_out,
        &mut full_gate_sigmoids,
        &mut scalar_derivatives,
    )
    .unwrap();

    let mut sigmoid_only_out = vec![0.0; input.len()];
    let mut sigmoid_only_gate_sigmoids = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_forward_with_sigmoid_cache(
        config,
        &gates,
        &input,
        &mut sigmoid_only_out,
        &mut sigmoid_only_gate_sigmoids,
    )
    .unwrap();

    assert_eq!(full_out, sigmoid_only_out);
    assert_eq!(full_gate_sigmoids, sigmoid_only_gate_sigmoids);

    let mut in_place_out = input.clone();
    let mut in_place_gate_sigmoids = vec![0.0; gates.len()];
    let mut scalar_inputs = vec![0.0; nodes * config.channels()];
    fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place(
        config,
        &gates,
        &mut in_place_out,
        &mut in_place_gate_sigmoids,
        &mut scalar_inputs,
    )
    .unwrap();

    assert_eq!(sigmoid_only_out, in_place_out);
    assert_eq!(sigmoid_only_gate_sigmoids, in_place_gate_sigmoids);
    for node_index in 0..nodes {
        let tensor_node_start = node_index * config.node_stride();
        let scalar_node_start = node_index * config.channels();
        assert_eq!(
            &input[tensor_node_start..tensor_node_start + config.channels()],
            &scalar_inputs[scalar_node_start..scalar_node_start + config.channels()]
        );
    }

    let mut full_grad_input = vec![0.0; input.len()];
    let mut full_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_cache(
        config,
        &gates,
        &input,
        &full_gate_sigmoids,
        &scalar_derivatives,
        &grad_out,
        &mut full_grad_gates,
        &mut full_grad_input,
    )
    .unwrap();

    let mut sigmoid_only_grad_input = vec![0.0; input.len()];
    let mut sigmoid_only_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache(
        config,
        &input,
        &sigmoid_only_gate_sigmoids,
        &grad_out,
        &mut sigmoid_only_grad_gates,
        &mut sigmoid_only_grad_input,
    )
    .unwrap();

    assert_eq!(full_grad_input, sigmoid_only_grad_input);
    assert_eq!(full_grad_gates, sigmoid_only_grad_gates);

    let mut in_place_grad_input = grad_out.clone();
    let mut in_place_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_in_place(
        config,
        &input,
        &sigmoid_only_gate_sigmoids,
        &mut in_place_grad_input,
        &mut in_place_grad_gates,
    )
    .unwrap();

    assert_eq!(sigmoid_only_grad_input, in_place_grad_input);
    assert_eq!(sigmoid_only_grad_gates, in_place_grad_gates);

    let mut from_activated_grad_input = grad_out.clone();
    let mut from_activated_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place(
        config,
        &in_place_out,
        &scalar_inputs,
        &in_place_gate_sigmoids,
        &mut from_activated_grad_input,
        &mut from_activated_grad_gates,
    )
    .unwrap();

    assert_close_slice(
        &sigmoid_only_grad_input,
        &from_activated_grad_input,
        2.0e-15,
    );
    assert_close_slice(
        &sigmoid_only_grad_gates,
        &from_activated_grad_gates,
        2.0e-15,
    );
}

#[test]
fn gate_activation_permuted_storage_matches_l_major_in_place_forward_and_vjp() {
    let config = EsenGateActivationConfig::new(3, 2, 4).unwrap();
    let storage_order = l_major_to_m_major_storage_order(config);
    let nodes = 5;
    let input_l_major: Vec<_> = (0..nodes * config.node_stride())
        .map(|index| (index as f64 * 0.041).sin() - 0.2)
        .collect();
    let gates: Vec<_> = (0..nodes * config.gate_node_stride())
        .map(|index| (index as f64 * 0.067).cos() - 0.4)
        .collect();
    let grad_out_l_major: Vec<_> = (0..input_l_major.len())
        .map(|index| (index as f64 * 0.053).sin())
        .collect();

    let mut expected_out = input_l_major.clone();
    let mut expected_gate_sigmoids = vec![0.0; gates.len()];
    let mut expected_scalar_inputs = vec![0.0; nodes * config.channels()];
    fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place(
        config,
        &gates,
        &mut expected_out,
        &mut expected_gate_sigmoids,
        &mut expected_scalar_inputs,
    )
    .unwrap();

    let mut permuted_out = reorder_l_major_to_storage(config, &storage_order, &input_l_major);
    let mut permuted_gate_sigmoids = vec![0.0; gates.len()];
    let mut permuted_scalar_inputs = vec![0.0; nodes * config.channels()];
    fairchem_esen_gate_activation_forward_with_sigmoid_cache_in_place_permuted(
        config,
        &storage_order,
        &gates,
        &mut permuted_out,
        &mut permuted_gate_sigmoids,
        &mut permuted_scalar_inputs,
    )
    .unwrap();
    let actual_out = reorder_storage_to_l_major(config, &storage_order, &permuted_out);
    assert_close_slice(&actual_out, &expected_out, 1.0e-14);
    assert_close_slice(&permuted_gate_sigmoids, &expected_gate_sigmoids, 1.0e-14);
    assert_close_slice(&permuted_scalar_inputs, &expected_scalar_inputs, 1.0e-14);

    let mut expected_grad_input = grad_out_l_major.clone();
    let mut expected_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place(
        config,
        &expected_out,
        &expected_scalar_inputs,
        &expected_gate_sigmoids,
        &mut expected_grad_input,
        &mut expected_grad_gates,
    )
    .unwrap();

    let mut permuted_grad_input =
        reorder_l_major_to_storage(config, &storage_order, &grad_out_l_major);
    let mut permuted_grad_gates = vec![0.0; gates.len()];
    fairchem_esen_gate_activation_vjp_with_sigmoid_cache_from_activated_in_place_permuted(
        config,
        &storage_order,
        &permuted_out,
        &permuted_scalar_inputs,
        &permuted_gate_sigmoids,
        &mut permuted_grad_input,
        &mut permuted_grad_gates,
    )
    .unwrap();
    let actual_grad_input =
        reorder_storage_to_l_major(config, &storage_order, &permuted_grad_input);
    assert_close_slice(&actual_grad_input, &expected_grad_input, 1.0e-14);
    assert_close_slice(&permuted_grad_gates, &expected_grad_gates, 1.0e-14);
}

#[test]
fn gate_activation_rejects_invalid_mmax() {
    assert!(EsenGateActivationConfig::new(1, 2, 4).is_err());
}

fn l_major_to_m_major_storage_order(config: EsenGateActivationConfig) -> Vec<usize> {
    let mapping = EsenCoefficientMappingConfig::new(config.l_max(), config.m_max()).unwrap();
    let m_to_l = fairchem_esen_l_to_m_permutation(mapping);
    let mut l_to_m = vec![0usize; m_to_l.len()];
    for (m_major, l_major) in m_to_l.iter().copied().enumerate() {
        l_to_m[l_major] = m_major;
    }
    l_to_m
}

fn reorder_l_major_to_storage(
    config: EsenGateActivationConfig,
    storage_order: &[usize],
    input: &[f64],
) -> Vec<f64> {
    let nodes = input.len() / config.node_stride();
    let mut out = vec![0.0; input.len()];
    for node in 0..nodes {
        let node_start = node * config.node_stride();
        for (l_major, &storage) in storage_order.iter().enumerate() {
            let source_start = node_start + l_major * config.channels();
            let target_start = node_start + storage * config.channels();
            out[target_start..target_start + config.channels()]
                .copy_from_slice(&input[source_start..source_start + config.channels()]);
        }
    }
    out
}

fn reorder_storage_to_l_major(
    config: EsenGateActivationConfig,
    storage_order: &[usize],
    input: &[f64],
) -> Vec<f64> {
    let nodes = input.len() / config.node_stride();
    let mut out = vec![0.0; input.len()];
    for node in 0..nodes {
        let node_start = node * config.node_stride();
        for (l_major, &storage) in storage_order.iter().enumerate() {
            let source_start = node_start + storage * config.channels();
            let target_start = node_start + l_major * config.channels();
            out[target_start..target_start + config.channels()]
                .copy_from_slice(&input[source_start..source_start + config.channels()]);
        }
    }
    out
}

fn scalar_objective(
    config: EsenGateActivationConfig,
    gates: &[f64],
    input: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; input.len()];
    fairchem_esen_gate_activation_forward(config, gates, input, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn silu_reference(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn sigmoid_reference(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let error = (actual_value - expected_value).abs();
        assert!(
            error <= tolerance,
            "index={index}, actual={actual_value}, expected={expected_value}, error={error}, tolerance={tolerance}"
        );
    }
}
