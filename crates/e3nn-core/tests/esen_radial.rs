#![allow(clippy::expect_used, clippy::needless_range_loop, clippy::unwrap_used)]

use e3nn_core::radial::{
    fairchem_esen_edge_envelope_modulation_forward,
    fairchem_esen_edge_envelope_modulation_forward_in_place,
    fairchem_esen_edge_envelope_modulation_vjp,
    fairchem_esen_edge_envelope_modulation_vjp_from_modulated_unchecked,
    fairchem_esen_enveloped_bessel_forward, fairchem_esen_enveloped_bessel_vjp,
    fairchem_esen_gaussian_smearing_forward, fairchem_esen_gaussian_smearing_vjp,
    fairchem_esen_polynomial_envelope_forward,
};

fn central_difference<F>(x: f64, step: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + step) - f(x - step)) / (2.0 * step)
}

#[test]
fn fairchem_esen_polynomial_envelope_matches_expected_values() {
    let at_zero = fairchem_esen_polynomial_envelope_forward(0.0, 5).unwrap();
    assert_eq!(at_zero.value, 1.0);
    assert_eq!(at_zero.derivative, 0.0);

    let at_cutoff = fairchem_esen_polynomial_envelope_forward(1.0, 5).unwrap();
    assert_eq!(at_cutoff.value, 0.0);
    assert_eq!(at_cutoff.derivative, 0.0);

    let middle = fairchem_esen_polynomial_envelope_forward(0.5, 5).unwrap();
    let finite_difference = central_difference(0.5, 1.0e-6, |x| {
        fairchem_esen_polynomial_envelope_forward(x, 5)
            .unwrap()
            .value
    });
    assert!((middle.derivative - finite_difference).abs() < 1.0e-8);
}

#[test]
fn fairchem_esen_gaussian_smearing_vjp_matches_finite_difference() {
    let distance = 1.7;
    let cutoff = 5.0;
    let basis_width_scalar = 2.0;
    let mut values = vec![0.0; 8];
    fairchem_esen_gaussian_smearing_forward(distance, cutoff, basis_width_scalar, &mut values)
        .unwrap();
    let bar_values = [0.3, -0.2, 0.7, 1.1, -0.4, 0.6, -0.8, 0.5];
    let analytic = fairchem_esen_gaussian_smearing_vjp(
        distance,
        cutoff,
        basis_width_scalar,
        &values,
        &bar_values,
    )
    .unwrap();

    let finite_difference = central_difference(distance, 1.0e-6, |d| {
        let mut perturbed = vec![0.0; values.len()];
        fairchem_esen_gaussian_smearing_forward(d, cutoff, basis_width_scalar, &mut perturbed)
            .unwrap();
        perturbed
            .iter()
            .zip(bar_values.iter())
            .map(|(value, bar)| value * bar)
            .sum::<f64>()
    });
    assert!((analytic - finite_difference).abs() < 1.0e-8);
}

#[test]
fn fairchem_esen_enveloped_bessel_vjp_matches_finite_difference() {
    let distance = 2.1;
    let cutoff = 5.0;
    let mut values = vec![0.0; 6];
    fairchem_esen_enveloped_bessel_forward(distance, cutoff, &mut values).unwrap();
    let bar_values = [0.2, -0.1, 0.5, -0.4, 0.8, -0.3];
    let analytic =
        fairchem_esen_enveloped_bessel_vjp(distance, cutoff, &values, &bar_values).unwrap();

    let finite_difference = central_difference(distance, 1.0e-6, |d| {
        let mut perturbed = vec![0.0; values.len()];
        fairchem_esen_enveloped_bessel_forward(d, cutoff, &mut perturbed).unwrap();
        perturbed
            .iter()
            .zip(bar_values.iter())
            .map(|(value, bar)| value * bar)
            .sum::<f64>()
    });
    assert!((analytic - finite_difference).abs() < 1.0e-8);
}

#[test]
fn fairchem_esen_enveloped_bessel_rejects_zero_distance() {
    let mut values = vec![0.0; 4];
    let error = fairchem_esen_enveloped_bessel_forward(0.0, 5.0, &mut values)
        .expect_err("zero distance would match upstream non-finite sin(x)/x behavior");
    assert!(
        error.to_string().contains("distance"),
        "unexpected error: {error}"
    );
}

#[test]
fn fairchem_esen_edge_envelope_modulation_multiplies_each_edge_feature() {
    let feature_dim = 3;
    let distances = [0.0, 2.5, 5.0];
    let cutoff = 5.0;
    let features = [1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 7.0, -8.0, 9.0];
    let mut out = [0.0; 9];

    fairchem_esen_edge_envelope_modulation_forward(
        feature_dim,
        &distances,
        cutoff,
        5,
        &features,
        &mut out,
    )
    .unwrap();

    for edge in 0..distances.len() {
        let envelope = fairchem_esen_polynomial_envelope_forward(distances[edge] / cutoff, 5)
            .unwrap()
            .value;
        for feature in 0..feature_dim {
            let index = edge * feature_dim + feature;
            assert_eq!(out[index], features[index] * envelope);
        }
    }
}

#[test]
fn fairchem_esen_edge_envelope_modulation_in_place_matches_out_of_place() {
    let feature_dim = 4;
    let distances = [0.0, 1.2, 4.9, 5.0];
    let cutoff = 5.0;
    let features = [
        1.0, -2.0, 0.5, 0.25, 3.0, 4.0, -1.0, 0.75, 7.0, -8.0, 9.0, -0.5, 1.5, -2.5, 3.5, -4.5,
    ];
    let mut expected = vec![0.0; features.len()];
    fairchem_esen_edge_envelope_modulation_forward(
        feature_dim,
        &distances,
        cutoff,
        5,
        &features,
        &mut expected,
    )
    .unwrap();

    let mut in_place = features.to_vec();
    fairchem_esen_edge_envelope_modulation_forward_in_place(
        feature_dim,
        &distances,
        cutoff,
        5,
        &mut in_place,
    )
    .unwrap();

    assert_eq!(in_place, expected);
}

#[test]
fn fairchem_esen_edge_envelope_modulation_vjp_matches_finite_difference() {
    let feature_dim = 3;
    let mut distances = vec![1.2, 3.4];
    let cutoff = 5.0;
    let mut features = vec![0.3, -0.7, 1.1, 2.0, -1.5, 0.4];
    let grad_out = [0.5, -0.2, 0.9, -1.1, 0.8, 0.25];
    let mut grad_features = vec![0.0; features.len()];
    let mut grad_distances = vec![0.0; distances.len()];

    fairchem_esen_edge_envelope_modulation_vjp(
        feature_dim,
        &distances,
        cutoff,
        5,
        &features,
        &grad_out,
        &mut grad_features,
        &mut grad_distances,
    )
    .unwrap();

    let step = 1.0e-6;
    for index in 0..features.len() {
        let original = features[index];
        features[index] = original + step;
        let plus =
            edge_envelope_scalar_objective(feature_dim, &distances, cutoff, &features, &grad_out);
        features[index] = original - step;
        let minus =
            edge_envelope_scalar_objective(feature_dim, &distances, cutoff, &features, &grad_out);
        features[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_features[index] - fd).abs() <= 1.0e-9,
            "feature index={index}, grad={}, fd={fd}",
            grad_features[index]
        );
    }
    for index in 0..distances.len() {
        let original = distances[index];
        distances[index] = original + step;
        let plus =
            edge_envelope_scalar_objective(feature_dim, &distances, cutoff, &features, &grad_out);
        distances[index] = original - step;
        let minus =
            edge_envelope_scalar_objective(feature_dim, &distances, cutoff, &features, &grad_out);
        distances[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_distances[index] - fd).abs() <= 1.0e-8,
            "distance index={index}, grad={}, fd={fd}",
            grad_distances[index]
        );
    }
}

#[test]
fn fairchem_esen_edge_envelope_modulation_vjp_from_modulated_matches_original() {
    let feature_dim = 3;
    let distances = vec![0.2, 1.2, 3.4, 5.0];
    let cutoff = 5.0;
    let features = vec![
        0.3, -0.7, 1.1, 2.0, -1.5, 0.4, -0.2, 0.8, -1.3, 3.0, -2.0, 1.0,
    ];
    let grad_out = [
        0.5, -0.2, 0.9, -1.1, 0.8, 0.25, 0.4, -0.6, 0.7, 1.2, -0.3, 0.1,
    ];
    let mut modulated = vec![0.0; features.len()];
    fairchem_esen_edge_envelope_modulation_forward(
        feature_dim,
        &distances,
        cutoff,
        5,
        &features,
        &mut modulated,
    )
    .unwrap();

    let mut expected_grad_features = vec![0.0; features.len()];
    let mut expected_grad_distances = vec![0.0; distances.len()];
    fairchem_esen_edge_envelope_modulation_vjp(
        feature_dim,
        &distances,
        cutoff,
        5,
        &features,
        &grad_out,
        &mut expected_grad_features,
        &mut expected_grad_distances,
    )
    .unwrap();

    let mut actual_grad_features = vec![0.0; features.len()];
    let mut actual_grad_distances = vec![0.0; distances.len()];
    fairchem_esen_edge_envelope_modulation_vjp_from_modulated_unchecked(
        feature_dim,
        &distances,
        cutoff,
        5,
        &modulated,
        &grad_out,
        &mut actual_grad_features,
        &mut actual_grad_distances,
    );

    for (index, (actual, expected)) in actual_grad_features
        .iter()
        .zip(expected_grad_features.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "feature index={index}, actual={actual}, expected={expected}"
        );
    }
    for (index, (actual, expected)) in actual_grad_distances
        .iter()
        .zip(expected_grad_distances.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1.0e-9,
            "distance index={index}, actual={actual}, expected={expected}"
        );
    }
}

fn edge_envelope_scalar_objective(
    feature_dim: usize,
    distances: &[f64],
    cutoff: f64,
    features: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; grad_out.len()];
    fairchem_esen_edge_envelope_modulation_forward(
        feature_dim,
        distances,
        cutoff,
        5,
        features,
        &mut out,
    )
    .unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}
