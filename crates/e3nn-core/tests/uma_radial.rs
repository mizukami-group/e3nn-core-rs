use e3nn_core::radial::{
    fairchem_esen_gaussian_smearing_forward, fairchem_esen_gaussian_smearing_vjp,
    fairchem_esen_polynomial_envelope_forward,
};

#[test]
fn uma_distance_basis_derivatives_match_finite_difference() -> e3nn_core::Result<()> {
    const CUTOFF: f64 = 6.0;
    const BASIS_WIDTH_SCALAR: f64 = 2.0;
    const BASIS: usize = 32;
    let cotangents = (0..BASIS)
        .map(|i| ((i as f64 + 1.0) * 0.173).sin())
        .collect::<Vec<_>>();

    for sample in 1..100 {
        let r = CUTOFF * sample as f64 / 100.0;
        let mut values = vec![0.0; BASIS];
        fairchem_esen_gaussian_smearing_forward(r, CUTOFF, BASIS_WIDTH_SCALAR, &mut values)?;
        let analytic = fairchem_esen_gaussian_smearing_vjp(
            r,
            CUTOFF,
            BASIS_WIDTH_SCALAR,
            &values,
            &cotangents,
        )?;

        let step = 1.0e-6;
        let mut plus = vec![0.0; BASIS];
        let mut minus = vec![0.0; BASIS];
        fairchem_esen_gaussian_smearing_forward(r + step, CUTOFF, BASIS_WIDTH_SCALAR, &mut plus)?;
        fairchem_esen_gaussian_smearing_forward(r - step, CUTOFF, BASIS_WIDTH_SCALAR, &mut minus)?;
        let plus_dot = plus
            .iter()
            .zip(cotangents.iter())
            .map(|(value, bar)| value * bar)
            .sum::<f64>();
        let minus_dot = minus
            .iter()
            .zip(cotangents.iter())
            .map(|(value, bar)| value * bar)
            .sum::<f64>();
        let fd = (plus_dot - minus_dot) / (2.0 * step);
        assert!(
            (analytic - fd).abs() <= 2.0e-8,
            "r={r}: analytic={analytic:.15e} fd={fd:.15e}"
        );
    }
    Ok(())
}

#[test]
fn uma_polynomial_envelope_derivative_is_smooth_inside_cutoff() -> e3nn_core::Result<()> {
    const EXPONENT: usize = 5;
    let step = 1.0e-6;
    let mut previous_derivative = None;
    for sample in 1..999 {
        let x = sample as f64 / 1000.0;
        let envelope = fairchem_esen_polynomial_envelope_forward(x, EXPONENT)?;
        let plus = fairchem_esen_polynomial_envelope_forward(x + step, EXPONENT)?.value;
        let minus = fairchem_esen_polynomial_envelope_forward(x - step, EXPONENT)?.value;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (envelope.derivative - fd).abs() <= 2.0e-8,
            "x={x}: derivative={:.15e} fd={fd:.15e}",
            envelope.derivative
        );
        if let Some(previous) = previous_derivative {
            let jump: f64 = envelope.derivative - previous;
            assert!(
                jump.abs() < 0.04,
                "unexpected envelope derivative jump at x={x}: {jump:.15e}"
            );
        }
        previous_derivative = Some(envelope.derivative);
    }
    Ok(())
}
