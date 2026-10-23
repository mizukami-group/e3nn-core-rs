#![allow(clippy::unwrap_used)]

use e3nn_core::activation::{
    fairchem_esen_separable_s2_activation_forward, fairchem_esen_separable_s2_activation_vjp,
};
use e3nn_core::mlp::silu;
use e3nn_core::so3::EsenSo3GridConfig;

#[test]
fn separable_s2_activation_forward_uses_scalar_silu_and_grid_tensor_branch() {
    let config = EsenSo3GridConfig::new(3, 2, 1, 2).unwrap();
    let input_scalars = [0.5, -1.0];
    let input_tensors = [1.0, -0.5, 2.0, 0.25, -1.5, 3.0];
    let to_grid = [1.0, 0.5, -0.25, -1.0, 0.75, 2.0];
    let from_grid = [0.25, -1.0, 0.5, 1.5, 0.0, -0.75];
    let mut out = [0.0; 6];

    fairchem_esen_separable_s2_activation_forward(
        config,
        &input_scalars,
        &input_tensors,
        &to_grid,
        &from_grid,
        &mut out,
    )
    .unwrap();

    assert_eq!(out[0], silu(input_scalars[0]));
    assert_eq!(out[1], silu(input_scalars[1]));
    let tensor_out = independent_s2_tensor_branch(config, &input_tensors, &to_grid, &from_grid);
    assert_eq!(out[2..], tensor_out[2..]);
}

#[test]
fn separable_s2_activation_vjp_matches_finite_difference() {
    let config = EsenSo3GridConfig::new(4, 2, 2, 2).unwrap();
    let mut input_scalars = vec![0.0; config.channels()];
    for (index, value) in input_scalars.iter_mut().enumerate() {
        *value = 0.17 * index as f64 - 0.4;
    }
    let mut input_tensors = vec![0.0; config.coefficient_node_stride()];
    for (index, value) in input_tensors.iter_mut().enumerate() {
        *value = 0.07 * index as f64 - 0.8;
    }
    let mut to_grid = vec![0.0; config.matrix_len()];
    for (index, value) in to_grid.iter_mut().enumerate() {
        *value = (0.11 * index as f64).sin();
    }
    let mut from_grid = vec![0.0; config.matrix_len()];
    for (index, value) in from_grid.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }
    let mut grad_out = vec![0.0; config.coefficient_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.19 * index as f64).sin();
    }
    let mut grad_scalars = vec![0.0; input_scalars.len()];
    let mut grad_tensors = vec![0.0; input_tensors.len()];

    fairchem_esen_separable_s2_activation_vjp(
        config,
        &input_scalars,
        &input_tensors,
        &to_grid,
        &from_grid,
        &grad_out,
        &mut grad_scalars,
        &mut grad_tensors,
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..input_scalars.len() {
        let original = input_scalars[index];
        input_scalars[index] = original + step;
        let plus = separable_s2_scalar_objective(
            config,
            &input_scalars,
            &input_tensors,
            &to_grid,
            &from_grid,
            &grad_out,
        );
        input_scalars[index] = original - step;
        let minus = separable_s2_scalar_objective(
            config,
            &input_scalars,
            &input_tensors,
            &to_grid,
            &from_grid,
            &grad_out,
        );
        input_scalars[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_scalars[index] - fd).abs() <= 5.0e-9,
            "scalar index={index}, grad={}, fd={fd}",
            grad_scalars[index]
        );
    }

    for index in 0..input_tensors.len() {
        let original = input_tensors[index];
        input_tensors[index] = original + step;
        let plus = separable_s2_scalar_objective(
            config,
            &input_scalars,
            &input_tensors,
            &to_grid,
            &from_grid,
            &grad_out,
        );
        input_tensors[index] = original - step;
        let minus = separable_s2_scalar_objective(
            config,
            &input_scalars,
            &input_tensors,
            &to_grid,
            &from_grid,
            &grad_out,
        );
        input_tensors[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_tensors[index] - fd).abs() <= 5.0e-8,
            "tensor index={index}, grad={}, fd={fd}",
            grad_tensors[index]
        );
    }
}

fn independent_s2_tensor_branch(
    config: EsenSo3GridConfig,
    input_tensors: &[f64],
    to_grid: &[f64],
    from_grid: &[f64],
) -> Vec<f64> {
    let mut grid = vec![0.0; config.grid_node_stride()];
    for grid_index in 0..config.grid_points() {
        for channel in 0..config.channels() {
            let mut value = 0.0;
            for coefficient in 0..config.coefficient_count() {
                value += to_grid[grid_index * config.coefficient_count() + coefficient]
                    * input_tensors[coefficient * config.channels() + channel];
            }
            grid[grid_index * config.channels() + channel] = silu(value);
        }
    }
    let mut out = vec![0.0; config.coefficient_node_stride()];
    for coefficient in 0..config.coefficient_count() {
        for channel in 0..config.channels() {
            let mut value = 0.0;
            for grid_index in 0..config.grid_points() {
                value += from_grid[grid_index * config.coefficient_count() + coefficient]
                    * grid[grid_index * config.channels() + channel];
            }
            out[coefficient * config.channels() + channel] = value;
        }
    }
    out
}

fn separable_s2_scalar_objective(
    config: EsenSo3GridConfig,
    input_scalars: &[f64],
    input_tensors: &[f64],
    to_grid: &[f64],
    from_grid: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_separable_s2_activation_forward(
        config,
        input_scalars,
        input_tensors,
        to_grid,
        from_grid,
        &mut out,
    )
    .unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
