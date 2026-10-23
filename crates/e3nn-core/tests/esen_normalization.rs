#![allow(clippy::unwrap_used)]

use e3nn_core::normalization::{
    fairchem_esen_degree_layer_scale_forward, fairchem_esen_degree_layer_scale_vjp,
    fairchem_esen_rms_norm_sh_forward, fairchem_esen_rms_norm_sh_vjp, EsenDegreeLayerScaleConfig,
    EsenRmsNormShConfig, EsenRmsNormShNormalization,
};

#[test]
fn rms_norm_sh_forward_matches_hand_computed_balanced_affine_case() {
    let config = EsenRmsNormShConfig::new(
        1,
        2,
        1.0e-5,
        EsenRmsNormShNormalization::ComponentDegreeBalanced,
        true,
    )
    .unwrap();
    let input = [2.0, 4.0, 1.0, -1.0, 3.0, 5.0, -2.0, 6.0];
    let weight = [1.5, 0.5, 2.0, -1.0];
    let bias = [0.25, -0.75];
    let mut out = [0.0; 8];

    fairchem_esen_rms_norm_sh_forward(config, &input, Some(&weight), Some(&bias), &mut out)
        .unwrap();

    let centered = [-1.0, 1.0, 1.0, -1.0, 3.0, 5.0, -2.0, 6.0];
    let q = (centered[0] * centered[0] + centered[1] * centered[1]) / 4.0
        + (centered[2] * centered[2]
            + centered[3] * centered[3]
            + centered[4] * centered[4]
            + centered[5] * centered[5]
            + centered[6] * centered[6]
            + centered[7] * centered[7])
            / 12.0;
    let scale = (q + 1.0e-5_f64).powf(-0.5);
    let expected = [
        centered[0] * scale * weight[0] + bias[0],
        centered[1] * scale * weight[1] + bias[1],
        centered[2] * scale * weight[2],
        centered[3] * scale * weight[3],
        centered[4] * scale * weight[2],
        centered[5] * scale * weight[3],
        centered[6] * scale * weight[2],
        centered[7] * scale * weight[3],
    ];

    for (actual, expected) in out.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= 1.0e-14);
    }
}

#[test]
fn rms_norm_sh_vjp_matches_finite_difference() {
    let config = EsenRmsNormShConfig::new(
        2,
        3,
        1.0e-5,
        EsenRmsNormShNormalization::ComponentDegreeBalanced,
        true,
    )
    .unwrap();
    let mut input = vec![0.0; config.node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = 0.17 * index as f64 - 1.3;
    }
    let weight = [1.1, 0.7, -0.5, 0.9, -1.2, 1.4, 0.6, 1.5, -0.8];
    let bias = [0.2, -0.3, 0.4];
    let mut grad_out = vec![0.0; config.node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (index as f64 * 0.31).sin();
    }
    let mut grad_input = vec![0.0; config.node_stride()];

    fairchem_esen_rms_norm_sh_vjp(
        config,
        &input,
        Some(&weight),
        Some(&bias),
        &grad_out,
        &mut grad_input,
    )
    .unwrap();

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
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

#[test]
fn rms_norm_sh_multi_node_matches_single_node_buffer_reuse() {
    let config = EsenRmsNormShConfig::new(
        2,
        3,
        1.0e-5,
        EsenRmsNormShNormalization::ComponentDegreeBalanced,
        true,
    )
    .unwrap();
    let stride = config.node_stride();
    let weight = [1.1, 0.7, -0.5, 0.9, -1.2, 1.4, 0.6, 1.5, -0.8];
    let bias = [0.2, -0.3, 0.4];

    // Two nodes with deliberately different value patterns so that a stale
    // reused scratch buffer would corrupt the second node's result.
    let mut input = vec![0.0; 2 * stride];
    for (index, value) in input.iter_mut().enumerate() {
        let node = index / stride;
        let within = (index % stride) as f64;
        *value = if node == 0 {
            0.17 * within - 1.3
        } else {
            -0.23 * within + 2.6
        };
    }
    let mut grad_out = vec![0.0; 2 * stride];
    for (index, value) in grad_out.iter_mut().enumerate() {
        let node = index / stride;
        let within = index % stride;
        *value = if node == 0 {
            (within as f64 * 0.31).sin()
        } else {
            (within as f64 * 0.41 + 0.5).cos()
        };
    }

    let mut multi_out = vec![0.0; 2 * stride];
    fairchem_esen_rms_norm_sh_forward(config, &input, Some(&weight), Some(&bias), &mut multi_out)
        .unwrap();
    let mut multi_grad_input = vec![0.0; 2 * stride];
    fairchem_esen_rms_norm_sh_vjp(
        config,
        &input,
        Some(&weight),
        Some(&bias),
        &grad_out,
        &mut multi_grad_input,
    )
    .unwrap();

    for node in 0..2 {
        let lo = node * stride;
        let hi = lo + stride;
        let mut single_out = vec![0.0; stride];
        fairchem_esen_rms_norm_sh_forward(
            config,
            &input[lo..hi],
            Some(&weight),
            Some(&bias),
            &mut single_out,
        )
        .unwrap();
        let mut single_grad_input = vec![0.0; stride];
        fairchem_esen_rms_norm_sh_vjp(
            config,
            &input[lo..hi],
            Some(&weight),
            Some(&bias),
            &grad_out[lo..hi],
            &mut single_grad_input,
        )
        .unwrap();

        for (multi, single) in multi_out[lo..hi].iter().zip(single_out.iter()) {
            assert_eq!(multi.to_bits(), single.to_bits());
        }
        for (multi, single) in multi_grad_input[lo..hi]
            .iter()
            .zip(single_grad_input.iter())
        {
            assert_eq!(multi.to_bits(), single.to_bits());
        }
    }
}

#[test]
fn rms_norm_sh_rejects_bias_without_centering() {
    let config =
        EsenRmsNormShConfig::new(1, 2, 1.0e-5, EsenRmsNormShNormalization::Component, false)
            .unwrap();
    let input = [1.0; 8];
    let bias = [0.0; 2];
    let mut out = [0.0; 8];

    assert!(
        fairchem_esen_rms_norm_sh_forward(config, &input, None, Some(&bias), &mut out).is_err()
    );
}

#[test]
fn degree_layer_scale_expands_weights_by_degree() {
    let config = EsenDegreeLayerScaleConfig::new(2, 2).unwrap();
    let input = [
        1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 5.0, 6.0, 7.0, 8.0, -3.0, -4.0, 9.0, 10.0, 11.0, 12.0,
        -5.0, -6.0,
    ];
    let weight = [0.5, -1.0, 2.0, 3.0, -0.25, 4.0];
    let mut out = [0.0; 18];

    fairchem_esen_degree_layer_scale_forward(config, &input, &weight, &mut out).unwrap();

    let degree_by_component = [0, 1, 1, 1, 2, 2, 2, 2, 2];
    for m_index in 0..config.basis_dim() {
        for channel in 0..config.channels() {
            let flat_index = m_index * config.channels() + channel;
            let expected = input[flat_index]
                * weight[degree_by_component[m_index] * config.channels() + channel];
            assert_eq!(out[flat_index], expected);
        }
    }
}

#[test]
fn degree_layer_scale_vjp_matches_finite_difference() {
    let config = EsenDegreeLayerScaleConfig::new(2, 2).unwrap();
    let mut input = vec![0.0; config.node_stride()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f64 * 0.2 - 0.7;
    }
    let weight = [0.5, -1.0, 2.0, 3.0, -0.25, 4.0];
    let mut grad_out = vec![0.0; config.node_stride()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (index as f64 * 0.19).cos();
    }
    let mut grad_input = vec![0.0; config.node_stride()];

    fairchem_esen_degree_layer_scale_vjp(config, &weight, &grad_out, &mut grad_input).unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = degree_layer_scale_objective(config, &input, &weight, &grad_out);
        input[index] = original - step;
        let minus = degree_layer_scale_objective(config, &input, &weight, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-9,
            "index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

fn scalar_objective(
    config: EsenRmsNormShConfig,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; input.len()];
    fairchem_esen_rms_norm_sh_forward(config, input, Some(weight), Some(bias), &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn degree_layer_scale_objective(
    config: EsenDegreeLayerScaleConfig,
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; input.len()];
    fairchem_esen_degree_layer_scale_forward(config, input, weight, &mut out).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
