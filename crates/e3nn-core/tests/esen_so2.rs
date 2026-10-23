#![allow(clippy::too_many_arguments, clippy::unwrap_used)]

#[cfg(feature = "parallel")]
use e3nn_core::so2::fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink;
use e3nn_core::so2::{
    fairchem_esen_so2_convolution_forward,
    fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled,
    fairchem_esen_so2_convolution_forward_prepared_scaled,
    fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source,
    fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source_m_major_out,
    fairchem_esen_so2_convolution_forward_prepared_with_radial_strided_block_source,
    fairchem_esen_so2_convolution_vjp, fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out,
    fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out_with_radial_block_source_and_sink,
    fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink,
    fairchem_esen_so2_linear_block_forward, fairchem_esen_so2_linear_block_vjp,
    fairchem_esen_so2_linear_block_with_radial_forward,
    fairchem_esen_so2_linear_block_with_radial_vjp, fairchem_esen_so2_linear_forward,
    fairchem_esen_so2_linear_vjp, fairchem_esen_so2_m_conv_block_forward,
    fairchem_esen_so2_m_conv_block_vjp, fairchem_esen_so2_m_conv_block_with_radial_forward,
    fairchem_esen_so2_m_conv_block_with_radial_vjp, fairchem_esen_so2_radial_modulation_forward,
    fairchem_esen_so2_radial_modulation_vjp, EsenSo2ConvolutionConfig,
    EsenSo2ConvolutionPreparedPlan, EsenSo2LinearBlockConfig, EsenSo2LinearConfig,
    EsenSo2MConvBlockConfig, EsenSo2RadialModulationConfig,
};
use e3nn_core::so3::{fairchem_esen_l_to_m_forward, EsenCoefficientMappingConfig};

#[test]
fn so2_public_m_size_rejects_orders_above_lmax() {
    let conv = EsenSo2ConvolutionConfig::new(3, 2, 4, 5, 1).unwrap();
    assert_eq!(conv.m_size(0), Some(4));
    assert_eq!(conv.m_size(2), Some(2));
    assert_eq!(conv.m_size(4), None);

    let linear = EsenSo2LinearConfig::new(3, 2, 4, 5).unwrap();
    assert_eq!(linear.m_size(0), Some(4));
    assert_eq!(linear.m_size(2), Some(2));
    assert_eq!(linear.m_size(4), None);
}

#[test]
fn so2_radial_modulation_reuses_radial_slice_for_real_and_imag_copies() {
    let config = EsenSo2RadialModulationConfig::new(3, 2).unwrap();
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 4.0];
    let radial = [0.25, -0.5, 2.0];
    let mut out = [0.0; 6];

    fairchem_esen_so2_radial_modulation_forward(config, &input, &radial, &mut out).unwrap();

    assert_eq!(out, [0.25, 1.0, 1.0, 0.75, 0.75, 8.0]);
}

#[test]
fn so2_radial_modulation_vjp_matches_finite_difference() {
    let config = EsenSo2RadialModulationConfig::new(4, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.17 * index as f64 - 0.9;
    }
    let mut radial = vec![0.0; config.feature_dim()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = 0.23 * index as f64 + 0.4;
    }
    let mut grad_out = vec![0.0; config.input_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.19 * index as f64).sin();
    }
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_radial = vec![0.0; radial.len()];

    fairchem_esen_so2_radial_modulation_vjp(
        config,
        &input,
        &radial,
        &grad_out,
        &mut grad_input,
        &mut grad_radial,
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = radial_modulation_scalar_objective(config, &input, &radial, &grad_out);
        input[index] = original - step;
        let minus = radial_modulation_scalar_objective(config, &input, &radial, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 1.0e-10,
            "input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..radial.len() {
        let original = radial[index];
        radial[index] = original + step;
        let plus = radial_modulation_scalar_objective(config, &input, &radial, &grad_out);
        radial[index] = original - step;
        let minus = radial_modulation_scalar_objective(config, &input, &radial, &grad_out);
        radial[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_radial[index] - fd).abs() <= 1.0e-10,
            "radial index={index}, grad={}, fd={fd}",
            grad_radial[index]
        );
    }
}

#[test]
fn so2_linear_block_with_radial_vjp_matches_finite_difference() {
    let config = EsenSo2LinearBlockConfig::new(2, 2, 2, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.13 * index as f64 - 0.7;
    }
    let mut radial = vec![0.0; config.input_copy_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = 0.19 * index as f64 + 0.5;
    }
    let mut weight = vec![0.0; config.weight_len()];
    for (index, value) in weight.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.11 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_radial = vec![0.0; radial.len()];

    fairchem_esen_so2_linear_block_with_radial_vjp(
        config,
        &input,
        &radial,
        &weight,
        &grad_out,
        &mut grad_input,
        &mut grad_radial,
    )
    .unwrap();

    assert_linear_with_radial_fd(
        config,
        &mut input,
        &mut radial,
        &weight,
        &grad_out,
        &grad_input,
        &grad_radial,
    );
}

#[test]
fn so2_m_conv_block_with_radial_vjp_matches_finite_difference() {
    let config = EsenSo2MConvBlockConfig::new(2, 2, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.17 * index as f64 - 1.2;
    }
    let mut radial = vec![0.0; config.input_copy_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = 0.23 * index as f64 + 0.4;
    }
    let mut weight = vec![0.0; config.weight_len()];
    for (index, value) in weight.iter_mut().enumerate() {
        *value = (0.09 * index as f64).sin();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.15 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_radial = vec![0.0; radial.len()];

    fairchem_esen_so2_m_conv_block_with_radial_vjp(
        config,
        &input,
        &radial,
        &weight,
        &grad_out,
        &mut grad_input,
        &mut grad_radial,
    )
    .unwrap();

    assert_m_conv_with_radial_fd(
        config,
        &mut input,
        &mut radial,
        &weight,
        &grad_out,
        &grad_input,
        &grad_radial,
    );
}

#[test]
fn so2_linear_assembly_maps_l_major_to_m_blocks_and_back() {
    let config = EsenSo2LinearConfig::new(1, 1, 1, 1).unwrap();
    let input = [1.0, 2.0, 3.0, 4.0];
    let radial = [10.0, 20.0, 30.0];
    let m0_weight = [1.0, 0.0, 0.0, 1.0];
    let m0_bias = [0.5, -0.5];
    let m1_weight = [2.0];
    let mut out = [0.0; 4];

    fairchem_esen_so2_linear_forward(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &m0_bias,
        &[&m1_weight],
        &mut out,
    )
    .unwrap();

    assert_eq!(out, [10.5, 120.0, 59.5, 240.0]);
}

#[test]
fn so2_linear_assembly_vjp_matches_finite_difference() {
    let config = EsenSo2LinearConfig::new(2, 1, 2, 1).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.11 * index as f64 - 0.6;
    }
    let mut radial = vec![0.0; config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = 0.17 * index as f64 + 0.25;
    }
    let mut m0_weight = vec![0.0; config.m_weight_len(0)];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin();
    }
    let m0_bias = vec![0.0; config.m0_bias_len()];
    let mut m1_weight = vec![0.0; config.m_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.13 * index as f64).cos();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.19 * index as f64).sin();
    }
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_radial = vec![0.0; radial.len()];

    fairchem_esen_so2_linear_vjp(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &[&m1_weight],
        &grad_out,
        &mut grad_input,
        Some(&mut grad_radial),
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = so2_linear_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
        );
        input[index] = original - step;
        let minus = so2_linear_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
        );
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "SO2 linear input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..radial.len() {
        let original = radial[index];
        radial[index] = original + step;
        let plus = so2_linear_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
        );
        radial[index] = original - step;
        let minus = so2_linear_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
        );
        radial[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_radial[index] - fd).abs() <= 5.0e-8,
            "SO2 linear radial index={index}, grad={}, fd={fd}",
            grad_radial[index]
        );
    }
}

#[test]
fn so2_convolution_assembly_matches_linear_case_for_diagonal_m_conv() {
    let config = EsenSo2ConvolutionConfig::new(1, 1, 1, 1, 0).unwrap();
    let input = [1.0, 2.0, 3.0, 4.0];
    let radial = [10.0, 20.0, 30.0];
    let m0_weight = [1.0, 0.0, 0.0, 1.0];
    let m0_bias = [0.5, -0.5];
    let m1_weight = [2.0, 0.0];
    let mut out = [0.0; 4];

    fairchem_esen_so2_convolution_forward(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &m0_bias,
        &[&m1_weight],
        &mut out,
        None,
    )
    .unwrap();

    assert_eq!(out, [10.5, 120.0, 59.5, 240.0]);
}

#[test]
fn so2_convolution_assembly_extracts_extra_m0_features() {
    let config = EsenSo2ConvolutionConfig::new(0, 0, 1, 1, 2).unwrap();
    let input = [3.0];
    let radial = [2.0];
    let m0_weight = [1.0, 10.0, 100.0];
    let m0_bias = [0.5, 1.5, 2.5];
    let mut out = [0.0; 1];
    let mut extra = [0.0; 2];

    fairchem_esen_so2_convolution_forward(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &m0_bias,
        &[],
        &mut out,
        Some(&mut extra),
    )
    .unwrap();

    assert_eq!(extra, [6.5, 61.5]);
    assert_eq!(out, [602.5]);
}

#[test]
fn so2_convolution_prepared_scaled_matches_forward_then_row_scale() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 0).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 3;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.031 * index as f64).sin() - 0.4;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.043 * index as f64).cos();
    }
    let mut m0_bias = vec![0.0; config.m0_bias_len()];
    for (index, value) in m0_bias.iter_mut().enumerate() {
        *value = 0.02 * index as f64 - 0.1;
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.037 * index as f64).sin();
    }
    let weights = [&m1_weight[..]];
    let scales = [0.25, -1.5, 2.0];
    let mut expected = vec![0.0; nodes * config.output_node_stride()];
    let mut actual = vec![0.0; expected.len()];

    fairchem_esen_so2_convolution_forward(
        config,
        &input,
        None,
        &m0_weight,
        &m0_bias,
        &weights,
        &mut expected,
        None,
    )
    .unwrap();
    for (node, scale) in scales.iter().copied().enumerate() {
        let start = node * config.output_node_stride();
        for value in &mut expected[start..start + config.output_node_stride()] {
            *value *= scale;
        }
    }
    fairchem_esen_so2_convolution_forward_prepared_scaled(
        &plan,
        &input,
        None,
        &m0_weight,
        &m0_bias,
        &weights,
        None,
        &scales,
        &mut actual,
        None,
    )
    .unwrap();

    assert_slices_close("prepared scaled", &actual, &expected, 1.0e-12);
}

#[test]
fn so2_convolution_radial_block_source_matches_full_radial_forward() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 1).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 512;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.017 * index as f64).sin() - 0.2 * (0.023 * index as f64).cos();
    }
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = (0.019 * index as f64).cos() + 0.3;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.029 * index as f64).sin();
    }
    let mut m0_bias = vec![0.0; config.m0_bias_len()];
    for (index, value) in m0_bias.iter_mut().enumerate() {
        *value = 0.013 * index as f64 - 0.05;
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.031 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let mut expected = vec![0.0; nodes * config.output_node_stride()];
    let mut actual = vec![0.0; expected.len()];
    let mut actual_strided = vec![0.0; expected.len()];
    let mut expected_extra = vec![0.0; nodes * config.extra_m0_output_channels()];
    let mut actual_extra = vec![0.0; expected_extra.len()];
    let mut actual_strided_extra = vec![0.0; expected_extra.len()];

    fairchem_esen_so2_convolution_forward(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &m0_bias,
        &weights,
        &mut expected,
        Some(&mut expected_extra),
    )
    .unwrap();
    fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source(
        &plan,
        &input,
        &m0_weight,
        &m0_bias,
        &weights,
        None,
        &mut actual,
        Some(&mut actual_extra),
        |radial_offset, block_len, block| {
            for node in 0..nodes {
                let source_start = node * config.radial_node_stride() + radial_offset;
                let target_start = node * block_len;
                block[target_start..target_start + block_len]
                    .copy_from_slice(&radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
    )
    .unwrap();

    assert_slices_close("radial block source output", &actual, &expected, 1.0e-12);
    assert_slices_close(
        "radial block source extra",
        &actual_extra,
        &expected_extra,
        1.0e-12,
    );

    fairchem_esen_so2_convolution_forward_prepared_with_radial_strided_block_source(
        &plan,
        &input,
        &m0_weight,
        &m0_bias,
        &weights,
        None,
        &mut actual_strided,
        Some(&mut actual_strided_extra),
        |radial_offset, block_len, row_stride, feature_offset, block| {
            for node in 0..nodes {
                let source_start = node * config.radial_node_stride() + radial_offset;
                let target_start = node * row_stride + feature_offset;
                block[target_start..target_start + block_len]
                    .copy_from_slice(&radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
    )
    .unwrap();
    assert_slices_close(
        "radial strided source output",
        &actual_strided,
        &expected,
        1.0e-12,
    );
    assert_slices_close(
        "radial strided source extra",
        &actual_strided_extra,
        &expected_extra,
        1.0e-12,
    );
}

#[test]
fn so2_convolution_radial_block_source_m_major_out_matches_l_to_m_forward() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 1).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 512;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.017 * index as f64).sin() - 0.2 * (0.023 * index as f64).cos();
    }
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = (0.019 * index as f64).cos() + 0.3;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.029 * index as f64).sin();
    }
    let mut m0_bias = vec![0.0; config.m0_bias_len()];
    for (index, value) in m0_bias.iter_mut().enumerate() {
        *value = 0.013 * index as f64 - 0.05;
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.031 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let mut expected_l_major = vec![0.0; nodes * config.output_node_stride()];
    let mut expected_m_major = vec![0.0; expected_l_major.len()];
    let mut actual_m_major = vec![0.0; expected_l_major.len()];
    let mut expected_extra = vec![0.0; nodes * config.extra_m0_output_channels()];
    let mut actual_extra = vec![0.0; expected_extra.len()];

    fairchem_esen_so2_convolution_forward(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &m0_bias,
        &weights,
        &mut expected_l_major,
        Some(&mut expected_extra),
    )
    .unwrap();
    let mapping = EsenCoefficientMappingConfig::new(config.l_max(), config.m_max()).unwrap();
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels(),
        &expected_l_major,
        &mut expected_m_major,
    )
    .unwrap();

    fairchem_esen_so2_convolution_forward_prepared_with_radial_block_source_m_major_out(
        &plan,
        &input,
        &m0_weight,
        &m0_bias,
        &weights,
        None,
        &mut actual_m_major,
        Some(&mut actual_extra),
        |radial_offset, block_len, block| {
            for node in 0..nodes {
                let source_start = node * config.radial_node_stride() + radial_offset;
                let target_start = node * block_len;
                block[target_start..target_start + block_len]
                    .copy_from_slice(&radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
    )
    .unwrap();

    assert_slices_close(
        "radial block source m-major output",
        &actual_m_major,
        &expected_m_major,
        1.0e-12,
    );
    assert_slices_close(
        "radial block source m-major extra",
        &actual_extra,
        &expected_extra,
        1.0e-12,
    );
}

#[test]
fn so2_convolution_m_major_input_scaled_matches_l_major_prepared_scaled() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 0).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 512;
    let mut input_l_major = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input_l_major.iter_mut().enumerate() {
        *value = (0.021 * index as f64).sin() + 0.1 * (0.034 * index as f64).cos();
    }
    let mut input_m_major = vec![0.0; input_l_major.len()];
    let input_mapping = EsenCoefficientMappingConfig::new(config.l_max(), config.m_max()).unwrap();
    fairchem_esen_l_to_m_forward(
        input_mapping,
        config.in_channels(),
        &input_l_major,
        &mut input_m_major,
    )
    .unwrap();
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.025 * index as f64).sin();
    }
    let mut m0_bias = vec![0.0; config.m0_bias_len()];
    for (index, value) in m0_bias.iter_mut().enumerate() {
        *value = 0.011 * index as f64 - 0.04;
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.027 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let scales: Vec<_> = (0..nodes)
        .map(|node| 0.5 + (0.013 * node as f64).sin())
        .collect();
    let mut expected = vec![0.0; nodes * config.output_node_stride()];
    let mut actual = vec![0.0; expected.len()];

    fairchem_esen_so2_convolution_forward_prepared_scaled(
        &plan,
        &input_l_major,
        None,
        &m0_weight,
        &m0_bias,
        &weights,
        None,
        &scales,
        &mut expected,
        None,
    )
    .unwrap();
    fairchem_esen_so2_convolution_forward_prepared_m_major_input_scaled(
        &plan,
        &input_m_major,
        None,
        &m0_weight,
        &m0_bias,
        &weights,
        &scales,
        &mut actual,
        None,
    )
    .unwrap();

    assert_slices_close("m-major input scaled", &actual, &expected, 1.0e-12);
}

#[test]
fn so2_convolution_m_major_grad_out_vjp_matches_l_major_vjp() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 0).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 512;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.017 * index as f64).sin() + 0.2 * (0.011 * index as f64).cos();
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.023 * index as f64).sin();
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.029 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let mut grad_out_l_major = vec![0.0; nodes * config.output_node_stride()];
    for (index, value) in grad_out_l_major.iter_mut().enumerate() {
        *value = (0.031 * index as f64).sin();
    }
    let mut grad_out_m_major = vec![0.0; grad_out_l_major.len()];
    let mapping = EsenCoefficientMappingConfig::new(config.l_max(), config.m_max()).unwrap();
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels(),
        &grad_out_l_major,
        &mut grad_out_m_major,
    )
    .unwrap();

    let mut expected_grad_input = vec![0.0; input.len()];
    fairchem_esen_so2_convolution_vjp(
        config,
        &input,
        None,
        &m0_weight,
        &weights,
        &grad_out_l_major,
        None,
        &mut expected_grad_input,
        None,
    )
    .unwrap();

    let mut actual_grad_input = vec![0.0; input.len()];
    fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out(
        &plan,
        &input,
        None,
        &m0_weight,
        &weights,
        None,
        &grad_out_m_major,
        None,
        &mut actual_grad_input,
        None,
    )
    .unwrap();

    assert_slices_close(
        "m-major grad-out VJP grad input",
        &actual_grad_input,
        &expected_grad_input,
        1.0e-10,
    );
}

#[test]
fn so2_convolution_m_major_grad_out_radial_block_source_vjp_matches_l_major_source_sink() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 2, 1).unwrap();
    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let nodes = 512;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.017 * index as f64).sin() + 0.2 * (0.011 * index as f64).cos();
    }
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = (0.019 * index as f64).cos() + 0.3;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.023 * index as f64).sin();
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.029 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let mut grad_out_l_major = vec![0.0; nodes * config.output_node_stride()];
    for (index, value) in grad_out_l_major.iter_mut().enumerate() {
        *value = (0.031 * index as f64).sin();
    }
    let mut grad_out_m_major = vec![0.0; grad_out_l_major.len()];
    let mapping = EsenCoefficientMappingConfig::new(config.l_max(), config.m_max()).unwrap();
    fairchem_esen_l_to_m_forward(
        mapping,
        config.out_channels(),
        &grad_out_l_major,
        &mut grad_out_m_major,
    )
    .unwrap();
    let mut grad_extra = vec![0.0; nodes * config.extra_m0_output_channels()];
    for (index, value) in grad_extra.iter_mut().enumerate() {
        *value = (0.037 * index as f64).cos();
    }

    let mut expected_grad_input = vec![0.0; input.len()];
    let mut expected_grad_radial = vec![0.0; radial.len()];
    fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink(
        &plan,
        &input,
        &m0_weight,
        &weights,
        None,
        &grad_out_l_major,
        Some(&grad_extra),
        &mut expected_grad_input,
        |radial_offset, block_len, block| {
            for node in 0..nodes {
                let source_start = node * config.radial_node_stride() + radial_offset;
                let target_start = node * block_len;
                block[target_start..target_start + block_len]
                    .copy_from_slice(&radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
        |radial_offset, block_len, block_grad_radial| {
            for node in 0..nodes {
                let source_start = node * block_len;
                let target_start = node * config.radial_node_stride() + radial_offset;
                expected_grad_radial[target_start..target_start + block_len]
                    .copy_from_slice(&block_grad_radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
    )
    .unwrap();

    let mut actual_grad_input = vec![0.0; input.len()];
    let mut actual_grad_radial = vec![0.0; radial.len()];
    fairchem_esen_so2_convolution_vjp_prepared_m_major_grad_out_with_radial_block_source_and_sink(
        &plan,
        &input,
        &m0_weight,
        &weights,
        None,
        &grad_out_m_major,
        Some(&grad_extra),
        &mut actual_grad_input,
        |radial_offset, block_len, block| {
            for node in 0..nodes {
                let source_start = node * config.radial_node_stride() + radial_offset;
                let target_start = node * block_len;
                block[target_start..target_start + block_len]
                    .copy_from_slice(&radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
        |radial_offset, block_len, block_grad_radial| {
            for node in 0..nodes {
                let source_start = node * block_len;
                let target_start = node * config.radial_node_stride() + radial_offset;
                actual_grad_radial[target_start..target_start + block_len]
                    .copy_from_slice(&block_grad_radial[source_start..source_start + block_len]);
            }
            Ok(())
        },
    )
    .unwrap();

    assert_slices_close(
        "m-major grad-out source/sink grad input",
        &actual_grad_input,
        &expected_grad_input,
        1.0e-10,
    );
    assert_slices_close(
        "m-major grad-out source/sink grad radial",
        &actual_grad_radial,
        &expected_grad_radial,
        1.0e-10,
    );
}

#[test]
fn so2_convolution_assembly_vjp_matches_finite_difference() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 1, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.09 * index as f64 - 0.4;
    }
    let mut radial = vec![0.0; config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = 0.15 * index as f64 + 0.35;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.05 * index as f64).sin();
    }
    let m0_bias = vec![0.0; config.m0_bias_len()];
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.11 * index as f64).cos();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.17 * index as f64).sin();
    }
    let mut grad_extra = vec![0.0; config.extra_m0_output_channels()];
    for (index, value) in grad_extra.iter_mut().enumerate() {
        *value = (0.23 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_radial = vec![0.0; radial.len()];

    fairchem_esen_so2_convolution_vjp(
        config,
        &input,
        Some(&radial),
        &m0_weight,
        &[&m1_weight],
        &grad_out,
        Some(&grad_extra),
        &mut grad_input,
        Some(&mut grad_radial),
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = so2_convolution_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
            Some(&grad_extra),
        );
        input[index] = original - step;
        let minus = so2_convolution_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
            Some(&grad_extra),
        );
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "SO2 convolution input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..radial.len() {
        let original = radial[index];
        radial[index] = original + step;
        let plus = so2_convolution_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
            Some(&grad_extra),
        );
        radial[index] = original - step;
        let minus = so2_convolution_scalar_objective(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &grad_out,
            Some(&grad_extra),
        );
        radial[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_radial[index] - fd).abs() <= 5.0e-8,
            "SO2 convolution radial index={index}, grad={}, fd={fd}",
            grad_radial[index]
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn so2_convolution_parallel_order_matches_serial() {
    let config = EsenSo2ConvolutionConfig::new(2, 1, 2, 1, 2).unwrap();
    let nodes = 512;
    let mut input = vec![0.0; nodes * config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.013 * index as f64).sin() - 0.2;
    }
    let mut radial = vec![0.0; nodes * config.radial_node_stride()];
    for (index, value) in radial.iter_mut().enumerate() {
        *value = (0.017 * index as f64).cos() + 0.8;
    }
    let mut m0_weight = vec![0.0; config.m0_weight_len()];
    for (index, value) in m0_weight.iter_mut().enumerate() {
        *value = (0.019 * index as f64).sin();
    }
    let mut m0_bias = vec![0.0; config.m0_bias_len()];
    for (index, value) in m0_bias.iter_mut().enumerate() {
        *value = 0.01 * index as f64 - 0.03;
    }
    let mut m1_weight = vec![0.0; config.m_conv_weight_len(1)];
    for (index, value) in m1_weight.iter_mut().enumerate() {
        *value = (0.023 * index as f64).cos();
    }
    let weights = [&m1_weight[..]];
    let mut serial_out = vec![0.0; nodes * config.output_node_stride()];
    let mut parallel_out = vec![0.0; serial_out.len()];
    let mut serial_extra = vec![0.0; nodes * config.extra_m0_output_channels()];
    let mut parallel_extra = vec![0.0; serial_extra.len()];

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    serial_pool.install(|| {
        fairchem_esen_so2_convolution_forward(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &weights,
            &mut serial_out,
            Some(&mut serial_extra),
        )
        .unwrap();
    });
    parallel_pool.install(|| {
        fairchem_esen_so2_convolution_forward(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &weights,
            &mut parallel_out,
            Some(&mut parallel_extra),
        )
        .unwrap();
    });
    assert_max_abs_close("parallel forward out", &parallel_out, &serial_out, 1.0e-12);
    assert_max_abs_close(
        "parallel forward extra",
        &parallel_extra,
        &serial_extra,
        1.0e-12,
    );

    let mut grad_out = vec![0.0; serial_out.len()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.029 * index as f64).sin();
    }
    let mut grad_extra = vec![0.0; serial_extra.len()];
    for (index, value) in grad_extra.iter_mut().enumerate() {
        *value = (0.031 * index as f64).cos();
    }
    let mut serial_grad_input = vec![0.0; input.len()];
    let mut parallel_grad_input = vec![0.0; input.len()];
    let mut serial_grad_radial = vec![0.0; radial.len()];
    let mut parallel_grad_radial = vec![0.0; radial.len()];
    serial_pool.install(|| {
        fairchem_esen_so2_convolution_vjp(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &weights,
            &grad_out,
            Some(&grad_extra),
            &mut serial_grad_input,
            Some(&mut serial_grad_radial),
        )
        .unwrap();
    });
    parallel_pool.install(|| {
        fairchem_esen_so2_convolution_vjp(
            config,
            &input,
            Some(&radial),
            &m0_weight,
            &weights,
            &grad_out,
            Some(&grad_extra),
            &mut parallel_grad_input,
            Some(&mut parallel_grad_radial),
        )
        .unwrap();
    });
    assert_max_abs_close(
        "parallel vjp input",
        &parallel_grad_input,
        &serial_grad_input,
        1.0e-12,
    );
    assert_max_abs_close(
        "parallel vjp radial",
        &parallel_grad_radial,
        &serial_grad_radial,
        1.0e-12,
    );

    let plan = EsenSo2ConvolutionPreparedPlan::new(config).unwrap();
    let mut sink_grad_input = vec![0.0; input.len()];
    let mut sink_grad_radial = vec![0.0; radial.len()];
    parallel_pool.install(|| {
        fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_sink(
            &plan,
            &input,
            &radial,
            &m0_weight,
            &weights,
            None,
            &grad_out,
            Some(&grad_extra),
            &mut sink_grad_input,
            |radial_offset, block_len, block_grad_radial| {
                for node in 0..nodes {
                    let source_start = node * block_len;
                    let target_start = node * config.radial_node_stride() + radial_offset;
                    sink_grad_radial[target_start..target_start + block_len].copy_from_slice(
                        &block_grad_radial[source_start..source_start + block_len],
                    );
                }
                Ok(())
            },
        )
        .unwrap();
    });
    assert_max_abs_close(
        "radial sink vjp input",
        &sink_grad_input,
        &parallel_grad_input,
        1.0e-12,
    );
    assert_max_abs_close(
        "radial sink vjp radial",
        &sink_grad_radial,
        &parallel_grad_radial,
        1.0e-12,
    );

    let mut source_sink_grad_input = vec![0.0; input.len()];
    let mut source_sink_grad_radial = vec![0.0; radial.len()];
    parallel_pool.install(|| {
        fairchem_esen_so2_convolution_vjp_prepared_with_radial_block_source_and_sink(
            &plan,
            &input,
            &m0_weight,
            &weights,
            None,
            &grad_out,
            Some(&grad_extra),
            &mut source_sink_grad_input,
            |radial_offset, block_len, block| {
                for node in 0..nodes {
                    let source_start = node * config.radial_node_stride() + radial_offset;
                    let target_start = node * block_len;
                    block[target_start..target_start + block_len]
                        .copy_from_slice(&radial[source_start..source_start + block_len]);
                }
                Ok(())
            },
            |radial_offset, block_len, block_grad_radial| {
                for node in 0..nodes {
                    let source_start = node * block_len;
                    let target_start = node * config.radial_node_stride() + radial_offset;
                    source_sink_grad_radial[target_start..target_start + block_len]
                        .copy_from_slice(
                            &block_grad_radial[source_start..source_start + block_len],
                        );
                }
                Ok(())
            },
        )
        .unwrap();
    });
    assert_max_abs_close(
        "radial source/sink vjp input",
        &source_sink_grad_input,
        &parallel_grad_input,
        1.0e-12,
    );
    assert_max_abs_close(
        "radial source/sink vjp radial",
        &source_sink_grad_radial,
        &parallel_grad_radial,
        1.0e-12,
    );
}

#[test]
fn so2_m_conv_block_mixes_real_and_imag_outputs() {
    let config = EsenSo2MConvBlockConfig::new(1, 2, 1).unwrap();
    let input = [1.0, -2.0, 0.5, 3.0];
    let weight = [0.25, -1.5, 2.0, 0.75];
    let mut out = [0.0; 2];

    fairchem_esen_so2_m_conv_block_forward(config, &input, &weight, &mut out).unwrap();

    let real_first = weight[0] * input[0] + weight[1] * input[1];
    let real_second = weight[2] * input[0] + weight[3] * input[1];
    let imag_first = weight[0] * input[2] + weight[1] * input[3];
    let imag_second = weight[2] * input[2] + weight[3] * input[3];
    assert_eq!(out[0], real_first - imag_second);
    assert_eq!(out[1], imag_first + real_second);
}

#[test]
fn so2_m_conv_block_vjp_matches_finite_difference() {
    let config = EsenSo2MConvBlockConfig::new(2, 2, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.21 * index as f64 - 0.8;
    }
    let mut weight = vec![0.0; config.weight_len()];
    for (index, value) in weight.iter_mut().enumerate() {
        *value = (0.09 * index as f64).sin();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.15 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; config.input_node_stride()];

    fairchem_esen_so2_m_conv_block_vjp(config, &weight, &grad_out, &mut grad_input).unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = m_conv_scalar_objective(config, &input, &weight, &grad_out);
        input[index] = original - step;
        let minus = m_conv_scalar_objective(config, &input, &weight, &grad_out);
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
fn so2_linear_m0_block_applies_dense_bias() {
    let config = EsenSo2LinearBlockConfig::new(2, 2, 2, 1).unwrap();
    let input = [1.0, -2.0, 0.5, 3.0];
    let weight = [
        0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, -2.0, 1.25, -0.5, 0.75, 1.0, -1.5, 0.5, 2.5, -0.25,
    ];
    let bias = [0.1, -0.2, 0.3, -0.4];
    let mut out = [0.0; 4];

    fairchem_esen_so2_linear_block_forward(config, &input, &weight, Some(&bias), &mut out).unwrap();

    for out_index in 0..config.output_copy_stride() {
        let mut expected = bias[out_index];
        let row_start = out_index * config.input_copy_stride();
        for in_index in 0..config.input_copy_stride() {
            expected += weight[row_start + in_index] * input[in_index];
        }
        assert_eq!(out[out_index], expected);
    }
}

#[test]
fn so2_linear_m_positive_block_reuses_dense_for_real_and_imag() {
    let config = EsenSo2LinearBlockConfig::new(2, 2, 1, 2).unwrap();
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
    let weight = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, -2.0];
    let mut out = [0.0; 4];

    fairchem_esen_so2_linear_block_forward(config, &input, &weight, None, &mut out).unwrap();

    for copy_index in 0..2 {
        let input_start = copy_index * config.input_copy_stride();
        let out_start = copy_index * config.output_copy_stride();
        for out_index in 0..config.output_copy_stride() {
            let mut expected = 0.0;
            let row_start = out_index * config.input_copy_stride();
            for in_index in 0..config.input_copy_stride() {
                expected += weight[row_start + in_index] * input[input_start + in_index];
            }
            assert_eq!(out[out_start + out_index], expected);
        }
    }
}

#[test]
fn so2_linear_block_vjp_matches_finite_difference() {
    let config = EsenSo2LinearBlockConfig::new(3, 2, 2, 2).unwrap();
    let mut input = vec![0.0; config.input_node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.2 * index as f64 - 1.1;
    }
    let mut weight = vec![0.0; config.weight_len()];
    for (index, value) in weight.iter_mut().enumerate() {
        *value = (0.13 * index as f64).sin();
    }
    let mut grad_out = vec![0.0; config.output_node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.17 * index as f64).cos();
    }
    let mut grad_input = vec![0.0; config.input_node_stride()];

    fairchem_esen_so2_linear_block_vjp(config, &weight, &grad_out, &mut grad_input).unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = scalar_objective(config, &input, &weight, &grad_out);
        input[index] = original - step;
        let minus = scalar_objective(config, &input, &weight, &grad_out);
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
fn so2_linear_block_rejects_bias_for_real_imag_copies() {
    let config = EsenSo2LinearBlockConfig::new(1, 1, 1, 2).unwrap();
    let input = [1.0, 2.0];
    let weight = [0.5];
    let bias = [0.25];
    let mut out = [0.0; 2];

    assert!(
        fairchem_esen_so2_linear_block_forward(config, &input, &weight, Some(&bias), &mut out)
            .is_err()
    );
}

fn scalar_objective(
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so2_linear_block_forward(config, input, weight, None, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn m_conv_scalar_objective(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so2_m_conv_block_forward(config, input, weight, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn radial_modulation_scalar_objective(
    config: EsenSo2RadialModulationConfig,
    input: &[f64],
    radial: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; input.len()];
    fairchem_esen_so2_radial_modulation_forward(config, input, radial, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn so2_linear_scalar_objective(
    config: EsenSo2LinearConfig,
    input: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so2_linear_forward(
        config, input, radial, m0_weight, m0_bias, m_weights, &mut out,
    )
    .unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn so2_convolution_scalar_objective(
    config: EsenSo2ConvolutionConfig,
    input: &[f64],
    radial: Option<&[f64]>,
    m0_weight: &[f64],
    m0_bias: &[f64],
    m_weights: &[&[f64]],
    grad_out: &[f64],
    grad_extra: Option<&[f64]>,
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    let mut extra = vec![0.0; grad_extra.map_or(0, |values| values.len())];
    fairchem_esen_so2_convolution_forward(
        config,
        input,
        radial,
        m0_weight,
        m0_bias,
        m_weights,
        &mut out,
        if extra.is_empty() {
            None
        } else {
            Some(&mut extra)
        },
    )
    .unwrap();
    let output_score: f64 = out
        .iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum();
    let extra_score: f64 = extra
        .iter()
        .zip(grad_extra.unwrap_or(&[]).iter())
        .map(|(value, grad)| value * grad)
        .sum();
    output_score + extra_score
}

fn assert_linear_with_radial_fd(
    config: EsenSo2LinearBlockConfig,
    input: &mut [f64],
    radial: &mut [f64],
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &[f64],
    grad_radial: &[f64],
) {
    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = linear_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        input[index] = original - step;
        let minus = linear_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "linear input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..radial.len() {
        let original = radial[index];
        radial[index] = original + step;
        let plus = linear_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        radial[index] = original - step;
        let minus = linear_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        radial[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_radial[index] - fd).abs() <= 5.0e-8,
            "linear radial index={index}, grad={}, fd={fd}",
            grad_radial[index]
        );
    }
}

fn assert_m_conv_with_radial_fd(
    config: EsenSo2MConvBlockConfig,
    input: &mut [f64],
    radial: &mut [f64],
    weight: &[f64],
    grad_out: &[f64],
    grad_input: &[f64],
    grad_radial: &[f64],
) {
    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = m_conv_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        input[index] = original - step;
        let minus = m_conv_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "m-conv input index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
    for index in 0..radial.len() {
        let original = radial[index];
        radial[index] = original + step;
        let plus = m_conv_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        radial[index] = original - step;
        let minus = m_conv_with_radial_scalar_objective(config, input, radial, weight, grad_out);
        radial[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_radial[index] - fd).abs() <= 5.0e-8,
            "m-conv radial index={index}, grad={}, fd={fd}",
            grad_radial[index]
        );
    }
}

#[cfg(feature = "parallel")]
fn assert_max_abs_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_slices_close(label, actual, expected, tolerance);
}

fn assert_slices_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        max_abs <= tolerance,
        "{label} max_abs={max_abs} exceeds tolerance={tolerance}"
    );
}

fn linear_with_radial_scalar_objective(
    config: EsenSo2LinearBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so2_linear_block_with_radial_forward(
        config, input, radial, weight, None, &mut out,
    )
    .unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn m_conv_with_radial_scalar_objective(
    config: EsenSo2MConvBlockConfig,
    input: &[f64],
    radial: &[f64],
    weight: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_so2_m_conv_block_with_radial_forward(config, input, radial, weight, &mut out)
        .unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
