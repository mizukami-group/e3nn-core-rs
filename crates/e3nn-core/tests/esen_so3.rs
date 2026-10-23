#![allow(clippy::unwrap_used)]

use e3nn_core::so3::{
    fairchem_esen_l_to_m_forward, fairchem_esen_l_to_m_permutation, fairchem_esen_m_to_l_forward,
    fairchem_esen_so3_batched_matmul_forward, fairchem_esen_so3_batched_matmul_vjp_input,
    fairchem_esen_so3_batched_matmul_vjp_input_write,
    fairchem_esen_so3_batched_matmul_vjp_input_write_f32,
    fairchem_esen_so3_batched_matmul_vjp_matrix, fairchem_esen_so3_batched_matmul_vjp_matrix_write,
    fairchem_esen_so3_from_grid_forward, fairchem_esen_so3_from_grid_vjp,
    fairchem_esen_so3_linear_forward, fairchem_esen_so3_linear_vjp,
    fairchem_esen_so3_to_grid_forward, fairchem_esen_so3_to_grid_vjp, EsenCoefficientMappingConfig,
    EsenSo3BatchedMatmulConfig, EsenSo3GridConfig, EsenSo3LinearConfig,
};

#[test]
fn coefficient_mapping_matches_fairchem_l_to_m_order_for_reduced_mmax() {
    let config = EsenCoefficientMappingConfig::new(2, 1).unwrap();
    assert_eq!(config.coefficient_count(), 7);
    assert_eq!(config.m_size(0), Some(3));
    assert_eq!(config.m_size(1), Some(2));
    assert_eq!(config.m_size(2), None);
    assert_eq!(
        fairchem_esen_l_to_m_permutation(config),
        vec![0, 2, 5, 3, 6, 1, 4]
    );
}

#[test]
fn coefficient_mapping_roundtrips_l_major_and_m_major_tensors() {
    let config = EsenCoefficientMappingConfig::new(2, 1).unwrap();
    let channels = 2;
    let input = [
        0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, 60.0, 61.0,
    ];
    let mut m_major = [0.0; 14];
    let mut roundtrip = [0.0; 14];

    fairchem_esen_l_to_m_forward(config, channels, &input, &mut m_major).unwrap();
    assert_eq!(
        m_major,
        [0.0, 1.0, 20.0, 21.0, 50.0, 51.0, 30.0, 31.0, 60.0, 61.0, 10.0, 11.0, 40.0, 41.0]
    );

    fairchem_esen_m_to_l_forward(config, channels, &m_major, &mut roundtrip).unwrap();
    assert_eq!(roundtrip, input);
}

#[test]
fn so3_linear_forward_expands_degree_weights_and_scalar_bias() {
    let config = EsenSo3LinearConfig::new(1, 2, 3).unwrap();
    let input = [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.5];
    let weight = [
        0.5, -1.0, 1.5, 2.0, -0.25, 0.75, 2.0, 0.25, -1.0, 0.5, 0.75, -1.5,
    ];
    let bias = [0.1, -0.2, 0.3];
    let mut out = [0.0; 12];

    fairchem_esen_so3_linear_forward(config, &input, &weight, &bias, &mut out).unwrap();

    let degree_by_component = [0, 1, 1, 1];
    for m_index in 0..config.basis_dim() {
        for out_channel in 0..config.out_channels() {
            let degree = degree_by_component[m_index];
            let mut expected = if m_index == 0 { bias[out_channel] } else { 0.0 };
            for in_channel in 0..config.in_channels() {
                let weight_index = (degree * config.out_channels() + out_channel)
                    * config.in_channels()
                    + in_channel;
                expected +=
                    weight[weight_index] * input[m_index * config.in_channels() + in_channel];
            }
            assert_eq!(out[m_index * config.out_channels() + out_channel], expected);
        }
    }
}

#[test]
fn so3_linear_vjp_matches_finite_difference() {
    let config = EsenSo3LinearConfig::new(2, 3, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.13 * index as f64 - 1.0;
    }
    let mut weight = vec![0.0; config.weight_len()];
    for (index, value) in weight.iter_mut().enumerate() {
        *value = (0.17 * index as f64).sin();
    }
    let bias = [0.4, -0.6];
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.11 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; config.input_node_stride()];

    fairchem_esen_so3_linear_vjp(config, &weight, &grad_out, &mut grad_input).unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = scalar_objective(config, &input, &weight, &bias, &grad_out);
        input[index] = original - step;
        let minus = scalar_objective(config, &input, &weight, &bias, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-9,
            "index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

#[test]
fn so3_grid_projection_matches_hand_computed_values() {
    let config = EsenSo3GridConfig::new(3, 2, 2, 2).unwrap();
    let input = [1.0, -1.0, 2.0, 0.5, -3.0, 4.0];
    let to_grid = [
        1.0, 0.0, 0.5, -1.0, 2.0, 0.0, 0.0, -0.5, 1.5, 0.25, 0.75, -2.0,
    ];
    let mut grid = [0.0; 8];

    fairchem_esen_so3_to_grid_forward(config, &input, &to_grid, &mut grid).unwrap();

    for grid_index in 0..config.grid_points() {
        for channel in 0..config.channels() {
            let mut expected = 0.0;
            for coefficient in 0..config.coefficient_count() {
                expected += to_grid[grid_index * config.coefficient_count() + coefficient]
                    * input[coefficient * config.channels() + channel];
            }
            assert_eq!(grid[grid_index * config.channels() + channel], expected);
        }
    }
}

#[test]
fn so3_grid_to_coefficients_matches_hand_computed_values() {
    let config = EsenSo3GridConfig::new(2, 2, 2, 2).unwrap();
    let grid = [1.0, -1.0, 2.0, 0.5, -3.0, 4.0, 0.25, -2.0];
    let from_grid = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 1.0, 0.0];
    let mut out = [0.0; 4];

    fairchem_esen_so3_from_grid_forward(config, &grid, &from_grid, &mut out).unwrap();

    for coefficient in 0..config.coefficient_count() {
        for channel in 0..config.channels() {
            let mut expected = 0.0;
            for grid_index in 0..config.grid_points() {
                expected += from_grid[grid_index * config.coefficient_count() + coefficient]
                    * grid[grid_index * config.channels() + channel];
            }
            assert_eq!(out[coefficient * config.channels() + channel], expected);
        }
    }
}

#[test]
fn so3_to_grid_vjp_matches_finite_difference() {
    let config = EsenSo3GridConfig::new(4, 3, 2, 3).unwrap();
    let mut input = vec![0.0; config.coefficient_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.07 * index as f64 - 0.8;
    }
    let mut to_grid = vec![0.0; config.matrix_len()];
    for (index, value) in to_grid.iter_mut().enumerate() {
        *value = (0.11 * index as f64).sin();
    }
    let mut grad_grid = vec![0.0; config.grid_node_stride()];
    for (index, value) in grad_grid.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; input.len()];

    fairchem_esen_so3_to_grid_vjp(config, &to_grid, &grad_grid, &mut grad_input).unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = so3_to_grid_scalar_objective(config, &input, &to_grid, &grad_grid);
        input[index] = original - step;
        let minus = so3_to_grid_scalar_objective(config, &input, &to_grid, &grad_grid);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-9,
            "to-grid input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

#[test]
fn so3_from_grid_vjp_matches_finite_difference() {
    let config = EsenSo3GridConfig::new(4, 3, 2, 3).unwrap();
    let mut grid = vec![0.0; config.grid_node_stride()];
    for (index, value) in grid.iter_mut().enumerate() {
        *value = 0.05 * index as f64 - 0.4;
    }
    let mut from_grid = vec![0.0; config.matrix_len()];
    for (index, value) in from_grid.iter_mut().enumerate() {
        *value = (0.09 * index as f64).cos();
    }
    let mut grad_out = vec![0.0; config.coefficient_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.17 * index as f64).sin();
    }
    let mut grad_grid = vec![0.0; grid.len()];

    fairchem_esen_so3_from_grid_vjp(config, &from_grid, &grad_out, &mut grad_grid).unwrap();

    let step = 1.0e-6;
    for index in 0..grid.len() {
        let original = grid[index];
        grid[index] = original + step;
        let plus = so3_from_grid_scalar_objective(config, &grid, &from_grid, &grad_out);
        grid[index] = original - step;
        let minus = so3_from_grid_scalar_objective(config, &grid, &from_grid, &grad_out);
        grid[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_grid[index] - fd).abs() <= 5.0e-9,
            "from-grid input index={index}, grad={}, fd={fd}",
            grad_grid[index]
        );
    }
}

#[test]
fn so3_batched_matmul_matches_hand_computed_values() {
    let config = EsenSo3BatchedMatmulConfig::new(2, 3, 2).unwrap();
    let matrix = [
        1.0, 0.5, -1.0, 2.0, 0.25, -0.75, 0.0, 1.5, 2.0, -0.5, 1.25, 0.75,
    ];
    let input = [2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.0, -0.5];
    let mut out = [0.0; 12];

    fairchem_esen_so3_batched_matmul_forward(config, &matrix, &input, &mut out).unwrap();

    for item in 0..2 {
        for out_coefficient in 0..config.output_coefficients() {
            for channel in 0..config.channels() {
                let mut expected = 0.0;
                for in_coefficient in 0..config.input_coefficients() {
                    expected += matrix[item * config.matrix_item_stride()
                        + out_coefficient * config.input_coefficients()
                        + in_coefficient]
                        * input[item * config.input_item_stride()
                            + in_coefficient * config.channels()
                            + channel];
                }
                assert_eq!(
                    out[item * config.output_item_stride()
                        + out_coefficient * config.channels()
                        + channel],
                    expected
                );
            }
        }
    }
}

#[test]
fn so3_batched_matmul_input_vjp_matches_finite_difference() {
    let config = EsenSo3BatchedMatmulConfig::new(3, 2, 2).unwrap();
    let mut matrix = vec![0.0; 2 * config.matrix_item_stride()];
    for (index, value) in matrix.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin();
    }
    let mut input = vec![0.0; 2 * config.input_item_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.11 * index as f64 - 0.9;
    }
    let mut grad_out = vec![0.0; 2 * config.output_item_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; input.len()];

    fairchem_esen_so3_batched_matmul_vjp_input(config, &matrix, &grad_out, &mut grad_input)
        .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = so3_batched_matmul_scalar_objective(config, &matrix, &input, &grad_out);
        input[index] = original - step;
        let minus = so3_batched_matmul_scalar_objective(config, &matrix, &input, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-9,
            "batched matmul input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

#[test]
fn so3_batched_matmul_matrix_vjp_matches_finite_difference() {
    let config = EsenSo3BatchedMatmulConfig::new(3, 2, 2).unwrap();
    let mut matrix = vec![0.0; 2 * config.matrix_item_stride()];
    for (index, value) in matrix.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin();
    }
    let mut input = vec![0.0; 2 * config.input_item_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.11 * index as f64 - 0.9;
    }
    let mut grad_out = vec![0.0; 2 * config.output_item_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }
    let mut grad_matrix = vec![0.0; matrix.len()];

    fairchem_esen_so3_batched_matmul_vjp_matrix(config, &input, &grad_out, &mut grad_matrix)
        .unwrap();

    let step = 1.0e-6;
    for index in 0..matrix.len() {
        let original = matrix[index];
        matrix[index] = original + step;
        let plus = so3_batched_matmul_scalar_objective(config, &matrix, &input, &grad_out);
        matrix[index] = original - step;
        let minus = so3_batched_matmul_scalar_objective(config, &matrix, &input, &grad_out);
        matrix[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_matrix[index] - fd).abs() <= 5.0e-9,
            "batched matmul matrix index={index}, grad={}, fd={fd}",
            grad_matrix[index]
        );
    }
}

#[test]
fn so3_batched_matmul_write_vjps_overwrite_existing_buffers() {
    assert_so3_batched_matmul_write_vjps_overwrite_existing_buffers(3);
}

#[cfg(feature = "parallel")]
#[test]
fn so3_batched_matmul_write_vjps_overwrite_existing_buffers_parallel_path() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| assert_so3_batched_matmul_write_vjps_overwrite_existing_buffers(512));
}

fn assert_so3_batched_matmul_write_vjps_overwrite_existing_buffers(items: usize) {
    let config = EsenSo3BatchedMatmulConfig::new(3, 2, 2).unwrap();
    let mut matrix = vec![0.0; items * config.matrix_item_stride()];
    for (index, value) in matrix.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin();
    }
    let mut input = vec![0.0; items * config.input_item_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.11 * index as f64 - 0.9;
    }
    let mut grad_out = vec![0.0; items * config.output_item_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }

    let mut expected_grad_input = vec![0.0; input.len()];
    fairchem_esen_so3_batched_matmul_vjp_input(
        config,
        &matrix,
        &grad_out,
        &mut expected_grad_input,
    )
    .unwrap();
    let mut actual_grad_input = vec![123.0; input.len()];
    fairchem_esen_so3_batched_matmul_vjp_input_write(
        config,
        &matrix,
        &grad_out,
        &mut actual_grad_input,
    )
    .unwrap();
    assert_eq!(actual_grad_input, expected_grad_input);

    let mut expected_grad_matrix = vec![0.0; matrix.len()];
    fairchem_esen_so3_batched_matmul_vjp_matrix(
        config,
        &input,
        &grad_out,
        &mut expected_grad_matrix,
    )
    .unwrap();
    let mut actual_grad_matrix = vec![-456.0; matrix.len()];
    fairchem_esen_so3_batched_matmul_vjp_matrix_write(
        config,
        &input,
        &grad_out,
        &mut actual_grad_matrix,
    )
    .unwrap();
    assert_eq!(actual_grad_matrix, expected_grad_matrix);
}

#[test]
fn so3_batched_matmul_f32_input_write_vjp_overwrites_existing_buffer() {
    assert_so3_batched_matmul_f32_input_write_vjp_overwrites_existing_buffer(3);
}

#[cfg(feature = "parallel")]
#[test]
fn so3_batched_matmul_f32_input_write_vjp_overwrites_existing_buffer_parallel_path() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| assert_so3_batched_matmul_f32_input_write_vjp_overwrites_existing_buffer(512));
}

fn assert_so3_batched_matmul_f32_input_write_vjp_overwrites_existing_buffer(items: usize) {
    let config = EsenSo3BatchedMatmulConfig::new(3, 2, 4).unwrap();
    let mut matrix = vec![0.0f32; items * config.matrix_item_stride()];
    for (index, value) in matrix.iter_mut().enumerate() {
        *value = (0.07 * index as f32).sin();
    }
    let mut grad_out = vec![0.0f32; items * config.output_item_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.13 * index as f32).cos();
    }

    let mut expected = vec![0.0f32; items * config.input_item_stride()];
    e3nn_core::so3::fairchem_esen_so3_batched_matmul_vjp_input_f32(
        config,
        &matrix,
        &grad_out,
        &mut expected,
    )
    .unwrap();

    let mut actual = vec![123.0f32; items * config.input_item_stride()];
    fairchem_esen_so3_batched_matmul_vjp_input_write_f32(config, &matrix, &grad_out, &mut actual)
        .unwrap();

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1.0e-6,
            "f32 write VJP mismatch at {index}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

fn scalar_objective(
    config: EsenSo3LinearConfig,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so3_linear_forward(config, input, weight, bias, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn so3_to_grid_scalar_objective(
    config: EsenSo3GridConfig,
    input: &[f64],
    to_grid: &[f64],
    grad_grid: &[f64],
) -> f64 {
    let mut grid = vec![0.0; grad_grid.len()];
    fairchem_esen_so3_to_grid_forward(config, input, to_grid, &mut grid).unwrap();
    grid.iter()
        .zip(grad_grid.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn so3_from_grid_scalar_objective(
    config: EsenSo3GridConfig,
    grid: &[f64],
    from_grid: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so3_from_grid_forward(config, grid, from_grid, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn so3_batched_matmul_scalar_objective(
    config: EsenSo3BatchedMatmulConfig,
    matrix: &[f64],
    input: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so3_batched_matmul_forward(config, matrix, input, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
