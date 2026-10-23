use e3nn_core::mlp::{silu, silu_derivative};

#[test]
fn silu_matches_reference_formula() {
    for value in [-80.0_f64, -12.5, -2.0, -0.25, 0.0, 0.5, 3.0, 16.0, 80.0] {
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        let expected = value * sigmoid;
        assert!((silu(value) - expected).abs() <= 1.0e-14, "value={value}");
    }
}

#[test]
fn silu_derivative_matches_finite_difference() {
    let step = 1.0e-6;
    for value in [-12.5_f64, -2.0, -0.25, 0.0, 0.5, 3.0, 16.0] {
        let fd = (silu(value + step) - silu(value - step)) / (2.0 * step);
        assert!(
            (silu_derivative(value) - fd).abs() <= 5.0e-9,
            "value={value}, derivative={}, fd={fd}",
            silu_derivative(value)
        );
    }
}

#[test]
fn silu_derivative_has_expected_origin_value() {
    assert!((silu(0.0) - 0.0).abs() <= f64::EPSILON);
    assert!((silu_derivative(0.0) - 0.5).abs() <= f64::EPSILON);
}
