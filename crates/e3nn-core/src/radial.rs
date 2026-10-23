//! Distance, cutoff, and radial-basis primitives for the CPU/f64 force path.

use crate::geometry::{dot, Vec3};
use crate::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct DistanceOutput {
    pub distance: f64,
    pub unit: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct DistanceJvpOutput {
    pub distance: f64,
    pub unit: Vec3,
    pub tangent: f64,
}

/// Forward pass for `d = ||r||`.
///
/// The `unit` vector uses the force-only VJP near-zero policy from
/// `docs/FORCE_ONLY_VJP_SPEC.md`: divide by `max(d, eps)`.
pub fn distance_forward(r: Vec3, eps: f64) -> Result<DistanceOutput> {
    validate_vec3("distance_forward r", r)?;
    validate_positive_finite("distance_forward eps", eps)?;
    let distance = dot(r, r).sqrt();
    let denom = distance.max(eps);
    Ok(DistanceOutput {
        distance,
        unit: [r[0] / denom, r[1] / denom, r[2] / denom],
    })
}

/// VJP for `d = ||r||`, returning the contribution to `bar_r`.
pub fn distance_vjp(r: Vec3, bar_distance: f64, eps: f64) -> Result<Vec3> {
    let mut bar_r = [0.0; 3];
    distance_vjp_accumulate(r, bar_distance, eps, &mut bar_r)?;
    Ok(bar_r)
}

/// Accumulating VJP for `d = ||r||`.
pub fn distance_vjp_accumulate(
    r: Vec3,
    bar_distance: f64,
    eps: f64,
    bar_r: &mut Vec3,
) -> Result<()> {
    validate_finite("distance_vjp_accumulate bar_distance", bar_distance)?;
    validate_vec3("distance_vjp_accumulate bar_r", *bar_r)?;
    let out = distance_forward(r, eps)?;
    bar_r[0] += bar_distance * out.unit[0];
    bar_r[1] += bar_distance * out.unit[1];
    bar_r[2] += bar_distance * out.unit[2];
    Ok(())
}

pub fn distance_forward_jvp(r: Vec3, r_dot: Vec3, eps: f64) -> Result<DistanceJvpOutput> {
    validate_vec3("distance_forward_jvp r", r)?;
    validate_vec3("distance_forward_jvp r_dot", r_dot)?;
    validate_positive_finite("distance_forward_jvp eps", eps)?;
    let out = distance_forward(r, eps)?;
    Ok(DistanceJvpOutput {
        distance: out.distance,
        unit: out.unit,
        tangent: dot(out.unit, r_dot),
    })
}

pub fn distance_vjp_linearize(
    r: Vec3,
    r_dot: Vec3,
    bar_distance: f64,
    bar_distance_dot: f64,
    eps: f64,
) -> Result<Vec3> {
    let mut bar_r_dot = [0.0; 3];
    distance_vjp_linearize_accumulate(
        r,
        r_dot,
        bar_distance,
        bar_distance_dot,
        eps,
        &mut bar_r_dot,
    )?;
    Ok(bar_r_dot)
}

pub fn distance_vjp_linearize_accumulate(
    r: Vec3,
    r_dot: Vec3,
    bar_distance: f64,
    bar_distance_dot: f64,
    eps: f64,
    bar_r_dot: &mut Vec3,
) -> Result<()> {
    validate_vec3("distance_vjp_linearize_accumulate r", r)?;
    validate_vec3("distance_vjp_linearize_accumulate r_dot", r_dot)?;
    validate_finite(
        "distance_vjp_linearize_accumulate bar_distance",
        bar_distance,
    )?;
    validate_finite(
        "distance_vjp_linearize_accumulate bar_distance_dot",
        bar_distance_dot,
    )?;
    validate_positive_finite("distance_vjp_linearize_accumulate eps", eps)?;
    validate_vec3("distance_vjp_linearize_accumulate bar_r_dot", *bar_r_dot)?;
    let out = distance_forward(r, eps)?;
    let unit_dot = if out.distance > eps {
        let unit_dot_r_dot = dot(out.unit, r_dot);
        [
            (r_dot[0] - out.unit[0] * unit_dot_r_dot) / out.distance,
            (r_dot[1] - out.unit[1] * unit_dot_r_dot) / out.distance,
            (r_dot[2] - out.unit[2] * unit_dot_r_dot) / out.distance,
        ]
    } else {
        [r_dot[0] / eps, r_dot[1] / eps, r_dot[2] / eps]
    };
    for axis in 0..3 {
        bar_r_dot[axis] += bar_distance_dot * out.unit[axis] + bar_distance * unit_dot[axis];
    }
    Ok(())
}

/// Cosine cutoff value and derivative wrt distance.
///
/// `c(d) = 0.5 * (cos(pi * d / radius) + 1)` for `0 <= d < radius`, else 0.
/// At and beyond the cutoff radius both value and derivative are pinned to zero.
#[derive(Clone, Copy, Debug)]
pub struct CutoffOutput {
    pub value: f64,
    pub derivative: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct RadialScalarOutput {
    pub value: f64,
    pub derivative: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct RadialScalarDeriv2 {
    pub value: f64,
    pub derivative: f64,
    pub second_derivative: f64,
}

pub const MACE_AGNESI_A: f64 = 1.0805;
pub const MACE_AGNESI_Q: f64 = 0.9183;
pub const MACE_AGNESI_P: f64 = 4.5791;
pub const MACE_ZBL_A_EXP: f64 = 0.300;
pub const MACE_ZBL_A_PREFACTOR: f64 = 0.4543;
pub const MACE_ZBL_COEFFICIENTS: [f64; 4] = [0.1818, 0.5099, 0.2802, 0.02817];
const MACE_ZBL_EXPONENTS: [f64; 4] = [3.2, 0.9423, 0.4028, 0.2016];
const MACE_ZBL_ENERGY_FACTOR: f64 = 14.3996;
const BOHR_RADIUS_ANGSTROM: f64 = 0.529;
const MACE_COVALENT_RADII: [f64; 119] = [
    0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07,
    1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22,
    1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45,
    1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,
    1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36,
    1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80,
    1.69, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
];

pub fn cosine_cutoff_forward(distance: f64, radius: f64) -> Result<CutoffOutput> {
    validate_distance("cosine_cutoff_forward distance", distance)?;
    validate_positive_finite("cosine_cutoff_forward radius", radius)?;
    if distance >= radius {
        return Ok(CutoffOutput {
            value: 0.0,
            derivative: 0.0,
        });
    }

    let x = std::f64::consts::PI * distance / radius;
    Ok(CutoffOutput {
        value: 0.5 * (x.cos() + 1.0),
        derivative: -0.5 * std::f64::consts::PI / radius * x.sin(),
    })
}

pub fn cosine_cutoff_vjp(distance: f64, radius: f64, bar_cutoff: f64) -> Result<f64> {
    validate_finite("cosine_cutoff_vjp bar_cutoff", bar_cutoff)?;
    Ok(bar_cutoff * cosine_cutoff_forward(distance, radius)?.derivative)
}

pub fn mace_covalent_radius(atomic_number: usize) -> Result<f64> {
    mace_covalent_radius_from_parts(atomic_number, &MACE_COVALENT_RADII)
}

pub fn mace_covalent_radius_from_parts(
    atomic_number: usize,
    covalent_radii: &[f64],
) -> Result<f64> {
    let Some(radius) = covalent_radii.get(atomic_number).copied() else {
        return Err(CoreError::InvalidRadial(format!(
            "MACE covalent radius is unavailable for atomic number {atomic_number}"
        )));
    };
    if radius.is_finite() && radius > 0.0 {
        Ok(radius)
    } else {
        Err(CoreError::InvalidRadial(format!(
            "MACE covalent radius for atomic number {atomic_number} must be finite and positive, got {radius}"
        )))
    }
}

pub fn mace_polynomial_cutoff_forward(
    distance: f64,
    radius: f64,
    p: usize,
) -> Result<CutoffOutput> {
    validate_distance("mace_polynomial_cutoff_forward distance", distance)?;
    validate_positive_finite("mace_polynomial_cutoff_forward radius", radius)?;
    if p == 0 {
        return Err(CoreError::InvalidRadial(
            "mace_polynomial_cutoff_forward p must be positive".to_owned(),
        ));
    }
    if distance >= radius {
        return Ok(CutoffOutput {
            value: 0.0,
            derivative: 0.0,
        });
    }

    let p_i = p as i32;
    let p = p as f64;
    let x = distance / radius;
    let coeff_a = (p + 1.0) * (p + 2.0) / 2.0;
    let coeff_b = p * (p + 2.0);
    let coeff_c = p * (p + 1.0) / 2.0;
    let x_p_minus_1 = if p_i == 1 { 1.0 } else { x.powi(p_i - 1) };
    let x_p = x_p_minus_1 * x;
    let x_p_plus_1 = x_p * x;
    let x_p_plus_2 = x_p_plus_1 * x;
    let value = 1.0 - coeff_a * x_p + coeff_b * x_p_plus_1 - coeff_c * x_p_plus_2;
    let derivative = (-coeff_a * p * x_p_minus_1 + coeff_b * (p + 1.0) * x_p
        - coeff_c * (p + 2.0) * x_p_plus_1)
        / radius;
    Ok(CutoffOutput { value, derivative })
}

pub fn mace_polynomial_cutoff_forward_deriv2(
    distance: f64,
    radius: f64,
    p: usize,
) -> Result<RadialScalarDeriv2> {
    validate_distance("mace_polynomial_cutoff_forward_deriv2 distance", distance)?;
    validate_positive_finite("mace_polynomial_cutoff_forward_deriv2 radius", radius)?;
    if p == 0 {
        return Err(CoreError::InvalidRadial(
            "mace_polynomial_cutoff_forward_deriv2 p must be positive".to_owned(),
        ));
    }
    if distance >= radius {
        return Ok(RadialScalarDeriv2 {
            value: 0.0,
            derivative: 0.0,
            second_derivative: 0.0,
        });
    }

    let p_i = p as i32;
    let p_f = p as f64;
    let x = distance / radius;
    let coeff_a = (p_f + 1.0) * (p_f + 2.0) / 2.0;
    let coeff_b = p_f * (p_f + 2.0);
    let coeff_c = p_f * (p_f + 1.0) / 2.0;
    let x_p_minus_2 = if p_i <= 1 { 0.0 } else { x.powi(p_i - 2) };
    let x_p_minus_1 = if p_i == 1 { 1.0 } else { x.powi(p_i - 1) };
    let x_p = x_p_minus_1 * x;
    let x_p_plus_1 = x_p * x;
    let x_p_plus_2 = x_p_plus_1 * x;
    let value = 1.0 - coeff_a * x_p + coeff_b * x_p_plus_1 - coeff_c * x_p_plus_2;
    let derivative = (-coeff_a * p_f * x_p_minus_1 + coeff_b * (p_f + 1.0) * x_p
        - coeff_c * (p_f + 2.0) * x_p_plus_1)
        / radius;
    let second_derivative = (-coeff_a * p_f * (p_f - 1.0) * x_p_minus_2
        + coeff_b * (p_f + 1.0) * p_f * x_p_minus_1
        - coeff_c * (p_f + 2.0) * (p_f + 1.0) * x_p)
        / (radius * radius);
    Ok(RadialScalarDeriv2 {
        value,
        derivative,
        second_derivative,
    })
}

/// FairChem eSEN polynomial envelope over scaled distance.
///
/// This matches the eSEN `PolynomialEnvelope(exponent=5)` family:
///
/// `1 + a x^p + b x^(p+1) + c x^(p+2)` for `x < 1`, else zero, where
/// `a = -(p + 1)(p + 2)/2`, `b = p(p + 2)`, and `c = -p(p + 1)/2`.
///
/// The derivative is with respect to the scaled distance `x`, not Angstrom.
pub fn fairchem_esen_polynomial_envelope_forward(
    scaled_distance: f64,
    exponent: usize,
) -> Result<CutoffOutput> {
    validate_distance(
        "fairchem_esen_polynomial_envelope_forward scaled_distance",
        scaled_distance,
    )?;
    if exponent == 0 {
        return Err(CoreError::InvalidRadial(
            "fairchem_esen_polynomial_envelope_forward exponent must be positive".to_owned(),
        ));
    }
    if scaled_distance >= 1.0 {
        return Ok(CutoffOutput {
            value: 0.0,
            derivative: 0.0,
        });
    }

    Ok(fairchem_esen_polynomial_envelope_forward_unchecked(
        scaled_distance,
        exponent,
    ))
}

/// FairChem eSEN polynomial envelope without input validation.
///
/// Callers must ensure `scaled_distance` is finite and non-negative and
/// `exponent > 0`. This is intended for model-internal hot paths after graph
/// preparation has already validated distances and constants.
pub fn fairchem_esen_polynomial_envelope_forward_unchecked(
    scaled_distance: f64,
    exponent: usize,
) -> CutoffOutput {
    if scaled_distance >= 1.0 {
        return CutoffOutput {
            value: 0.0,
            derivative: 0.0,
        };
    }
    let p_i = exponent as i32;
    let p = exponent as f64;
    let a = -((p + 1.0) * (p + 2.0) / 2.0);
    let b = p * (p + 2.0);
    let c = -(p * (p + 1.0) / 2.0);
    let x = scaled_distance;
    let x_p_minus_1 = if p_i == 1 { 1.0 } else { x.powi(p_i - 1) };
    let x_p = x_p_minus_1 * x;
    let x_p_plus_1 = x_p * x;
    let x_p_plus_2 = x_p_plus_1 * x;
    let value = 1.0 + a * x_p + b * x_p_plus_1 + c * x_p_plus_2;
    let derivative = a * p * x_p_minus_1 + b * (p + 1.0) * x_p + c * (p + 2.0) * x_p_plus_1;
    CutoffOutput { value, derivative }
}

/// FairChem eSEN Edgewise envelope modulation:
/// `out[edge, feature] = edge_features[edge, feature] * envelope(distance/cutoff)`.
pub fn fairchem_esen_edge_envelope_modulation_forward(
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    edge_features: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let edges = check_edge_envelope_modulation_shapes(
        "fairchem_esen_edge_envelope_modulation_forward",
        feature_dim,
        edge_distances,
        cutoff,
        exponent,
        edge_features,
        out,
    )?;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && edges >= crate::esen_parallel_min_nodes() {
        edge_distances
            .par_iter()
            .zip(edge_features.par_chunks(feature_dim))
            .zip(out.par_chunks_mut(feature_dim))
            .for_each(|((distance, edge_feature), out_edge)| {
                let envelope = fairchem_esen_polynomial_envelope_forward_unchecked(
                    *distance / cutoff,
                    exponent,
                )
                .value;
                for feature in 0..feature_dim {
                    out_edge[feature] = edge_feature[feature] * envelope;
                }
            });
        return Ok(());
    }
    for (edge_index, &distance) in edge_distances.iter().enumerate().take(edges) {
        let envelope =
            fairchem_esen_polynomial_envelope_forward_unchecked(distance / cutoff, exponent).value;
        let start = edge_index * feature_dim;
        for feature in 0..feature_dim {
            out[start + feature] = edge_features[start + feature] * envelope;
        }
    }
    Ok(())
}

/// In-place FairChem eSEN edgewise envelope modulation.
///
/// This is equivalent to [`fairchem_esen_edge_envelope_modulation_forward`] with
/// `edge_features` and `out` aliased, but avoids materializing a second
/// edge-feature buffer in model-internal fixed-architecture paths.
pub fn fairchem_esen_edge_envelope_modulation_forward_in_place(
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    edge_features: &mut [f64],
) -> Result<()> {
    let edges = check_edge_envelope_modulation_shapes(
        "fairchem_esen_edge_envelope_modulation_forward_in_place",
        feature_dim,
        edge_distances,
        cutoff,
        exponent,
        edge_features,
        edge_features,
    )?;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && edges >= crate::esen_parallel_min_nodes() {
        edge_distances
            .par_iter()
            .zip(edge_features.par_chunks_mut(feature_dim))
            .for_each(|(distance, edge_feature)| {
                let envelope = fairchem_esen_polynomial_envelope_forward_unchecked(
                    *distance / cutoff,
                    exponent,
                )
                .value;
                for feature in edge_feature {
                    *feature *= envelope;
                }
            });
        return Ok(());
    }
    for (edge_index, &distance) in edge_distances.iter().enumerate().take(edges) {
        let envelope =
            fairchem_esen_polynomial_envelope_forward_unchecked(distance / cutoff, exponent).value;
        let start = edge_index * feature_dim;
        for feature in &mut edge_features[start..start + feature_dim] {
            *feature *= envelope;
        }
    }
    Ok(())
}

/// Accumulate edge-feature and edge-distance VJPs of
/// [`fairchem_esen_edge_envelope_modulation_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_edge_envelope_modulation_vjp(
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    edge_features: &[f64],
    grad_out: &[f64],
    grad_edge_features: &mut [f64],
    grad_edge_distances: &mut [f64],
) -> Result<()> {
    check_edge_envelope_modulation_shapes(
        "fairchem_esen_edge_envelope_modulation_vjp",
        feature_dim,
        edge_distances,
        cutoff,
        exponent,
        edge_features,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_edge_envelope_modulation_vjp grad_edge_features",
        edge_features.len(),
        grad_edge_features.len(),
    )?;
    check_len(
        "fairchem_esen_edge_envelope_modulation_vjp grad_edge_distances",
        edge_distances.len(),
        grad_edge_distances.len(),
    )?;
    fairchem_esen_edge_envelope_modulation_vjp_unchecked(
        feature_dim,
        edge_distances,
        cutoff,
        exponent,
        edge_features,
        grad_out,
        grad_edge_features,
        grad_edge_distances,
    );
    Ok(())
}

/// Accumulate edge envelope VJPs without shape or finite-value validation.
///
/// This is intended for model-internal hot paths after graph preparation and
/// tensor shape validation have already established the invariants checked by
/// [`fairchem_esen_edge_envelope_modulation_vjp`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_edge_envelope_modulation_vjp_unchecked(
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    edge_features: &[f64],
    grad_out: &[f64],
    grad_edge_features: &mut [f64],
    grad_edge_distances: &mut [f64],
) {
    let edges = edge_distances.len();
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && edges >= crate::esen_parallel_min_nodes() {
        edge_distances
            .par_iter()
            .zip(edge_features.par_chunks(feature_dim))
            .zip(grad_out.par_chunks(feature_dim))
            .zip(grad_edge_features.par_chunks_mut(feature_dim))
            .zip(grad_edge_distances.par_iter_mut())
            .for_each(
                |(
                    (((distance, edge_feature), grad_out_edge), grad_edge_feature),
                    grad_distance,
                )| {
                    let envelope = fairchem_esen_polynomial_envelope_forward_unchecked(
                        *distance / cutoff,
                        exponent,
                    );
                    let mut grad_envelope = 0.0;
                    for feature in 0..feature_dim {
                        let grad = grad_out_edge[feature];
                        grad_edge_feature[feature] += grad * envelope.value;
                        grad_envelope += grad * edge_feature[feature];
                    }
                    *grad_distance += grad_envelope * envelope.derivative / cutoff;
                },
            );
        return;
    }
    for (edge_index, &distance) in edge_distances.iter().enumerate().take(edges) {
        let envelope =
            fairchem_esen_polynomial_envelope_forward_unchecked(distance / cutoff, exponent);
        let start = edge_index * feature_dim;
        let mut grad_envelope = 0.0;
        for feature in 0..feature_dim {
            let index = start + feature;
            let grad = grad_out[index];
            grad_edge_features[index] += grad * envelope.value;
            grad_envelope += grad * edge_features[index];
        }
        grad_edge_distances[edge_index] += grad_envelope * envelope.derivative / cutoff;
    }
}

/// Accumulate edge envelope VJPs when only the modulated edge features are
/// retained from forward.
///
/// Given `modulated_edge_features = edge_features * envelope(distance/cutoff)`,
/// this produces the same cotangents as
/// [`fairchem_esen_edge_envelope_modulation_vjp_unchecked`]. At and beyond the
/// cutoff, the FairChem polynomial has both value and derivative equal to zero,
/// so the distance cotangent is zero and the original edge feature is not
/// needed.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_edge_envelope_modulation_vjp_from_modulated_unchecked(
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    modulated_edge_features: &[f64],
    grad_out: &[f64],
    grad_edge_features: &mut [f64],
    grad_edge_distances: &mut [f64],
) {
    let edges = edge_distances.len();
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && edges >= crate::esen_parallel_min_nodes() {
        edge_distances
            .par_iter()
            .zip(modulated_edge_features.par_chunks(feature_dim))
            .zip(grad_out.par_chunks(feature_dim))
            .zip(grad_edge_features.par_chunks_mut(feature_dim))
            .zip(grad_edge_distances.par_iter_mut())
            .for_each(
                |(
                    (((distance, modulated_edge_feature), grad_out_edge), grad_edge_feature),
                    grad_distance,
                )| {
                    let envelope = fairchem_esen_polynomial_envelope_forward_unchecked(
                        *distance / cutoff,
                        exponent,
                    );
                    let mut grad_envelope = 0.0;
                    if envelope.value != 0.0 {
                        let inv_envelope = envelope.value.recip();
                        for feature in 0..feature_dim {
                            let grad = grad_out_edge[feature];
                            grad_edge_feature[feature] += grad * envelope.value;
                            grad_envelope += grad * modulated_edge_feature[feature] * inv_envelope;
                        }
                    }
                    *grad_distance += grad_envelope * envelope.derivative / cutoff;
                },
            );
        return;
    }
    for (edge_index, &distance) in edge_distances.iter().enumerate().take(edges) {
        let envelope =
            fairchem_esen_polynomial_envelope_forward_unchecked(distance / cutoff, exponent);
        let start = edge_index * feature_dim;
        let mut grad_envelope = 0.0;
        if envelope.value != 0.0 {
            let inv_envelope = envelope.value.recip();
            for feature in 0..feature_dim {
                let index = start + feature;
                let grad = grad_out[index];
                grad_edge_features[index] += grad * envelope.value;
                grad_envelope += grad * modulated_edge_features[index] * inv_envelope;
            }
        }
        grad_edge_distances[edge_index] += grad_envelope * envelope.derivative / cutoff;
    }
}

/// FairChem eSEN `GaussianSmearing(0, cutoff, N, basis_width_scalar)`.
pub fn fairchem_esen_gaussian_smearing_forward(
    distance: f64,
    cutoff: f64,
    basis_width_scalar: f64,
    out: &mut [f64],
) -> Result<()> {
    validate_distance("fairchem_esen_gaussian_smearing_forward distance", distance)?;
    validate_positive_finite("fairchem_esen_gaussian_smearing_forward cutoff", cutoff)?;
    validate_positive_finite(
        "fairchem_esen_gaussian_smearing_forward basis_width_scalar",
        basis_width_scalar,
    )?;
    validate_nonempty("fairchem_esen_gaussian_smearing_forward out", out)?;
    if out.len() < 2 {
        return Err(CoreError::InvalidRadial(
            "fairchem_esen_gaussian_smearing_forward requires at least two basis functions"
                .to_owned(),
        ));
    }

    let step = cutoff / (out.len() - 1) as f64;
    let coeff = -0.5 / (basis_width_scalar * step).powi(2);
    for (index, value) in out.iter_mut().enumerate() {
        let offset = step * index as f64;
        let delta = distance - offset;
        *value = (coeff * delta * delta).exp();
    }
    Ok(())
}

pub fn fairchem_esen_gaussian_smearing_vjp(
    distance: f64,
    cutoff: f64,
    basis_width_scalar: f64,
    values: &[f64],
    bar_values: &[f64],
) -> Result<f64> {
    validate_distance("fairchem_esen_gaussian_smearing_vjp distance", distance)?;
    validate_positive_finite("fairchem_esen_gaussian_smearing_vjp cutoff", cutoff)?;
    validate_positive_finite(
        "fairchem_esen_gaussian_smearing_vjp basis_width_scalar",
        basis_width_scalar,
    )?;
    if values.len() != bar_values.len() {
        return Err(CoreError::ShapeMismatch {
            op: "fairchem_esen_gaussian_smearing_vjp cotangents",
            expected: values.len(),
            actual: bar_values.len(),
        });
    }
    validate_nonempty("fairchem_esen_gaussian_smearing_vjp values", values)?;
    if values.len() < 2 {
        return Err(CoreError::InvalidRadial(
            "fairchem_esen_gaussian_smearing_vjp requires at least two basis functions".to_owned(),
        ));
    }
    validate_finite_slice("fairchem_esen_gaussian_smearing_vjp values", values)?;
    validate_finite_slice("fairchem_esen_gaussian_smearing_vjp bar_values", bar_values)?;

    let step = cutoff / (values.len() - 1) as f64;
    let coeff = -0.5 / (basis_width_scalar * step).powi(2);
    let mut bar_distance = 0.0;
    for (index, (value, bar_value)) in values.iter().zip(bar_values).enumerate() {
        let offset = step * index as f64;
        let derivative = value * 2.0 * coeff * (distance - offset);
        bar_distance += bar_value * derivative;
    }
    Ok(bar_distance)
}

/// FairChem eSEN `EnvelopedBesselBasis`.
///
/// The frequency values intentionally use f32-rounded multiples of pi before
/// widening to f64, matching FairChem's `torch.tensor(np.pi * arange(...,
/// dtype=np.float32))` initialization followed by `.double()`.
pub fn fairchem_esen_enveloped_bessel_forward(
    distance: f64,
    cutoff: f64,
    out: &mut [f64],
) -> Result<()> {
    validate_positive_finite("fairchem_esen_enveloped_bessel_forward distance", distance)?;
    validate_positive_finite("fairchem_esen_enveloped_bessel_forward cutoff", cutoff)?;
    validate_nonempty("fairchem_esen_enveloped_bessel_forward out", out)?;

    let scaled = distance / cutoff;
    let envelope = fairchem_esen_polynomial_envelope_forward(scaled, 5)?.value;
    let norm = (2.0 / cutoff.powi(3)).sqrt();
    for (index, value) in out.iter_mut().enumerate() {
        let frequency = (std::f32::consts::PI * (index + 1) as f32) as f64;
        *value = envelope * norm * (frequency * scaled).sin() / scaled;
    }
    Ok(())
}

pub fn fairchem_esen_enveloped_bessel_vjp(
    distance: f64,
    cutoff: f64,
    values: &[f64],
    bar_values: &[f64],
) -> Result<f64> {
    validate_positive_finite("fairchem_esen_enveloped_bessel_vjp distance", distance)?;
    validate_positive_finite("fairchem_esen_enveloped_bessel_vjp cutoff", cutoff)?;
    if values.len() != bar_values.len() {
        return Err(CoreError::ShapeMismatch {
            op: "fairchem_esen_enveloped_bessel_vjp cotangents",
            expected: values.len(),
            actual: bar_values.len(),
        });
    }
    validate_nonempty("fairchem_esen_enveloped_bessel_vjp values", values)?;
    validate_finite_slice("fairchem_esen_enveloped_bessel_vjp values", values)?;
    validate_finite_slice("fairchem_esen_enveloped_bessel_vjp bar_values", bar_values)?;

    let scaled = distance / cutoff;
    let envelope = fairchem_esen_polynomial_envelope_forward(scaled, 5)?;
    let norm = (2.0 / cutoff.powi(3)).sqrt();
    let mut bar_distance = 0.0;
    for (index, bar_value) in bar_values.iter().enumerate() {
        let frequency = (std::f32::consts::PI * (index + 1) as f32) as f64;
        let argument = frequency * scaled;
        let rbf_derivative_scaled =
            norm * (frequency * argument.cos() * scaled - argument.sin()) / (scaled * scaled);
        let rbf_value = norm * argument.sin() / scaled;
        let derivative_distance =
            (envelope.derivative * rbf_value + envelope.value * rbf_derivative_scaled) / cutoff;
        bar_distance += bar_value * derivative_distance;
    }
    Ok(bar_distance)
}

pub fn mace_bessel_basis_forward(distance: f64, radius: f64, out: &mut [f64]) -> Result<()> {
    validate_positive_finite("mace_bessel_basis_forward distance", distance)?;
    validate_positive_finite("mace_bessel_basis_forward radius", radius)?;
    validate_nonempty("mace_bessel_basis_forward out", out)?;

    let prefactor = (2.0 / radius).sqrt();
    let theta = std::f64::consts::PI * distance / radius;
    let (sin_theta, cos_theta) = theta.sin_cos();
    let mut sin_n = sin_theta;
    let mut cos_n = cos_theta;
    for (index, value) in out.iter_mut().enumerate() {
        if index > 0 {
            let next_sin = sin_n * cos_theta + cos_n * sin_theta;
            let next_cos = cos_n * cos_theta - sin_n * sin_theta;
            sin_n = next_sin;
            cos_n = next_cos;
        }
        *value = prefactor * sin_n / distance;
    }
    Ok(())
}

pub fn mace_bessel_basis_vjp(
    distance: f64,
    radius: f64,
    basis_values: &[f64],
    bar_basis: &[f64],
) -> Result<f64> {
    validate_positive_finite("mace_bessel_basis_vjp distance", distance)?;
    validate_positive_finite("mace_bessel_basis_vjp radius", radius)?;
    if basis_values.len() != bar_basis.len() {
        return Err(CoreError::ShapeMismatch {
            op: "mace_bessel_basis_vjp cotangents",
            expected: basis_values.len(),
            actual: bar_basis.len(),
        });
    }
    validate_nonempty("mace_bessel_basis_vjp basis_values", basis_values)?;
    validate_finite_slice("mace_bessel_basis_vjp basis_values", basis_values)?;
    validate_finite_slice("mace_bessel_basis_vjp bar_basis", bar_basis)?;

    let prefactor = (2.0 / radius).sqrt();
    let theta = std::f64::consts::PI * distance / radius;
    let (sin_theta, cos_theta) = theta.sin_cos();
    let mut sin_n = sin_theta;
    let mut cos_n = cos_theta;
    let mut bar_distance = 0.0;
    for (index, bar_value) in bar_basis.iter().enumerate() {
        if index > 0 {
            let next_sin = sin_n * cos_theta + cos_n * sin_theta;
            let next_cos = cos_n * cos_theta - sin_n * sin_theta;
            sin_n = next_sin;
            cos_n = next_cos;
        }
        let weight = std::f64::consts::PI * (index + 1) as f64 / radius;
        let derivative = prefactor * (weight * cos_n * distance - sin_n) / (distance * distance);
        bar_distance += bar_value * derivative;
    }
    Ok(bar_distance)
}

pub fn mace_bessel_basis_forward_deriv2(
    distance: f64,
    radius: f64,
    out: &mut [RadialScalarDeriv2],
) -> Result<()> {
    validate_positive_finite("mace_bessel_basis_forward_deriv2 distance", distance)?;
    validate_positive_finite("mace_bessel_basis_forward_deriv2 radius", radius)?;
    if out.is_empty() {
        return Err(CoreError::InvalidRadial(
            "mace_bessel_basis_forward_deriv2 out must not be empty".to_owned(),
        ));
    }

    let prefactor = (2.0 / radius).sqrt();
    for (index, value) in out.iter_mut().enumerate() {
        let weight = std::f64::consts::PI * (index + 1) as f64 / radius;
        let theta = weight * distance;
        let sin = theta.sin();
        let cos = theta.cos();
        let distance2 = distance * distance;
        let distance3 = distance2 * distance;
        *value = RadialScalarDeriv2 {
            value: prefactor * sin / distance,
            derivative: prefactor * (weight * cos * distance - sin) / distance2,
            second_derivative: prefactor
                * (-(weight * weight) * sin / distance - 2.0 * weight * cos / distance2
                    + 2.0 * sin / distance3),
        };
    }
    Ok(())
}

pub fn mace_bessel_cutoff_basis_forward(
    distance: f64,
    radius: f64,
    cutoff_p: usize,
    out: &mut [f64],
) -> Result<CutoffOutput> {
    let cutoff = mace_polynomial_cutoff_forward(distance, radius, cutoff_p)?;
    mace_bessel_basis_forward(distance, radius, out)?;
    for value in out {
        *value *= cutoff.value;
    }
    Ok(cutoff)
}

pub fn mace_bessel_cutoff_basis_vjp(
    distance: f64,
    radius: f64,
    cutoff_p: usize,
    basis_values: &[f64],
    bar_weighted_basis: &[f64],
) -> Result<f64> {
    if basis_values.len() != bar_weighted_basis.len() {
        return Err(CoreError::ShapeMismatch {
            op: "mace_bessel_cutoff_basis_vjp cotangents",
            expected: basis_values.len(),
            actual: bar_weighted_basis.len(),
        });
    }
    validate_nonempty("mace_bessel_cutoff_basis_vjp basis_values", basis_values)?;
    validate_finite_slice("mace_bessel_cutoff_basis_vjp basis_values", basis_values)?;
    validate_finite_slice(
        "mace_bessel_cutoff_basis_vjp bar_weighted_basis",
        bar_weighted_basis,
    )?;

    let cutoff = mace_polynomial_cutoff_forward(distance, radius, cutoff_p)?;
    let bar_cutoff = basis_values
        .iter()
        .zip(bar_weighted_basis)
        .map(|(basis, bar)| basis * bar)
        .sum::<f64>();
    let bar_basis = bar_weighted_basis
        .iter()
        .map(|bar| bar * cutoff.value)
        .collect::<Vec<_>>();
    Ok(cutoff.derivative * bar_cutoff
        + mace_bessel_basis_vjp(distance, radius, basis_values, &bar_basis)?)
}

pub fn mace_bessel_cutoff_basis_forward_deriv2(
    distance: f64,
    radius: f64,
    cutoff_p: usize,
    out: &mut [RadialScalarDeriv2],
) -> Result<()> {
    if out.is_empty() {
        return Err(CoreError::InvalidRadial(
            "mace_bessel_cutoff_basis_forward_deriv2 out must not be empty".to_owned(),
        ));
    }
    let cutoff = mace_polynomial_cutoff_forward_deriv2(distance, radius, cutoff_p)?;
    let mut bessel = vec![
        RadialScalarDeriv2 {
            value: 0.0,
            derivative: 0.0,
            second_derivative: 0.0,
        };
        out.len()
    ];
    mace_bessel_basis_forward_deriv2(distance, radius, &mut bessel)?;
    for (target, basis) in out.iter_mut().zip(bessel) {
        *target = RadialScalarDeriv2 {
            value: cutoff.value * basis.value,
            derivative: cutoff.derivative * basis.value + cutoff.value * basis.derivative,
            second_derivative: cutoff.second_derivative * basis.value
                + 2.0 * cutoff.derivative * basis.derivative
                + cutoff.value * basis.second_derivative,
        };
    }
    Ok(())
}

pub fn mace_agnesi_transform_forward(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
) -> Result<RadialScalarOutput> {
    mace_agnesi_transform_forward_from_parts(
        distance,
        src_atomic_number,
        dst_atomic_number,
        MACE_AGNESI_Q,
        MACE_AGNESI_P,
        MACE_AGNESI_A,
        &MACE_COVALENT_RADII,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mace_agnesi_transform_forward_from_parts(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    q: f64,
    p: f64,
    a: f64,
    covalent_radii: &[f64],
) -> Result<RadialScalarOutput> {
    validate_positive_finite("mace_agnesi_transform_forward distance", distance)?;
    validate_positive_finite("mace_agnesi_transform_forward q", q)?;
    validate_positive_finite("mace_agnesi_transform_forward p", p)?;
    validate_positive_finite("mace_agnesi_transform_forward a", a)?;
    let src_radius = mace_covalent_radius_from_parts(src_atomic_number, covalent_radii)?;
    let dst_radius = mace_covalent_radius_from_parts(dst_atomic_number, covalent_radii)?;
    let r0 = 0.5 * (src_radius + dst_radius);
    let x = distance / r0;
    let x_q = x.powf(q);
    let x_q_minus_p = x.powf(q - p);
    let denom = 1.0 + x_q_minus_p;
    let inner = a * x_q / denom;
    let value = 1.0 / (1.0 + inner);

    let d_inner_dx =
        a * (q * x.powf(q - 1.0) * denom - x_q * (q - p) * x.powf(q - p - 1.0)) / (denom * denom);
    let derivative = -d_inner_dx / ((1.0 + inner) * (1.0 + inner)) / r0;
    Ok(RadialScalarOutput { value, derivative })
}

pub fn mace_agnesi_transform_forward_deriv2(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
) -> Result<RadialScalarDeriv2> {
    mace_agnesi_transform_forward_deriv2_from_parts(
        distance,
        src_atomic_number,
        dst_atomic_number,
        MACE_AGNESI_Q,
        MACE_AGNESI_P,
        MACE_AGNESI_A,
        &MACE_COVALENT_RADII,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mace_agnesi_transform_forward_deriv2_from_parts(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    q: f64,
    p: f64,
    a: f64,
    covalent_radii: &[f64],
) -> Result<RadialScalarDeriv2> {
    validate_positive_finite("mace_agnesi_transform_forward_deriv2 distance", distance)?;
    validate_positive_finite("mace_agnesi_transform_forward_deriv2 q", q)?;
    validate_positive_finite("mace_agnesi_transform_forward_deriv2 p", p)?;
    validate_positive_finite("mace_agnesi_transform_forward_deriv2 a", a)?;
    let src_radius = mace_covalent_radius_from_parts(src_atomic_number, covalent_radii)?;
    let dst_radius = mace_covalent_radius_from_parts(dst_atomic_number, covalent_radii)?;
    let r0 = 0.5 * (src_radius + dst_radius);
    let x = distance / r0;
    let n = x.powf(q);
    let n_first = q * x.powf(q - 1.0);
    let n_second = q * (q - 1.0) * x.powf(q - 2.0);
    let t_exp = q - p;
    let t = x.powf(t_exp);
    let t_first = t_exp * x.powf(t_exp - 1.0);
    let t_second = t_exp * (t_exp - 1.0) * x.powf(t_exp - 2.0);
    let h = 1.0 + t;
    let h_first = t_first;
    let h_second = t_second;
    let inner = a * n / h;
    let inner_first = a * (n_first * h - n * h_first) / (h * h);
    let inner_second = a
        * (n_second / h - 2.0 * n_first * h_first / (h * h) - n * h_second / (h * h)
            + 2.0 * n * h_first * h_first / (h * h * h));
    let denom = 1.0 + inner;
    let value = 1.0 / denom;
    let derivative = -inner_first / (denom * denom) / r0;
    let second_derivative = (-inner_second / (denom * denom)
        + 2.0 * inner_first * inner_first / (denom * denom * denom))
        / (r0 * r0);
    Ok(RadialScalarDeriv2 {
        value,
        derivative,
        second_derivative,
    })
}

pub fn mace_zbl_edge_energy_forward(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    cutoff_p: usize,
) -> Result<RadialScalarOutput> {
    mace_zbl_edge_energy_forward_from_parts(
        distance,
        src_atomic_number,
        dst_atomic_number,
        cutoff_p,
        &MACE_ZBL_COEFFICIENTS,
        MACE_ZBL_A_EXP,
        MACE_ZBL_A_PREFACTOR,
        &MACE_COVALENT_RADII,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mace_zbl_edge_energy_forward_from_parts(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    cutoff_p: usize,
    coefficients: &[f64],
    a_exp: f64,
    a_prefactor: f64,
    covalent_radii: &[f64],
) -> Result<RadialScalarOutput> {
    validate_positive_finite("mace_zbl_edge_energy_forward distance", distance)?;
    if coefficients.len() != MACE_ZBL_EXPONENTS.len() {
        return Err(CoreError::InvalidRadial(format!(
            "MACE ZBL coefficients must have {} values, got {}",
            MACE_ZBL_EXPONENTS.len(),
            coefficients.len()
        )));
    }
    validate_positive_finite("mace_zbl_edge_energy_forward a_exp", a_exp)?;
    validate_positive_finite("mace_zbl_edge_energy_forward a_prefactor", a_prefactor)?;
    let src_radius = mace_covalent_radius_from_parts(src_atomic_number, covalent_radii)?;
    let dst_radius = mace_covalent_radius_from_parts(dst_atomic_number, covalent_radii)?;
    let src_z = src_atomic_number as f64;
    let dst_z = dst_atomic_number as f64;
    let screening_length =
        a_prefactor * BOHR_RADIUS_ANGSTROM / (src_z.powf(a_exp) + dst_z.powf(a_exp));
    let r_over_a = distance / screening_length;
    let mut phi = 0.0;
    let mut d_phi = 0.0;
    for index in 0..MACE_ZBL_EXPONENTS.len() {
        let exp_value = (-MACE_ZBL_EXPONENTS[index] * r_over_a).exp();
        phi += coefficients[index] * exp_value;
        d_phi += coefficients[index] * exp_value * (-MACE_ZBL_EXPONENTS[index] / screening_length);
    }

    let prefactor = MACE_ZBL_ENERGY_FACTOR * src_z * dst_z;
    let base = prefactor * phi / distance;
    let d_base = prefactor * (d_phi * distance - phi) / (distance * distance);
    let cutoff = mace_polynomial_cutoff_forward(distance, src_radius + dst_radius, cutoff_p)?;
    Ok(RadialScalarOutput {
        value: 0.5 * base * cutoff.value,
        derivative: 0.5 * (d_base * cutoff.value + base * cutoff.derivative),
    })
}

pub fn mace_zbl_edge_energy_forward_deriv2(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    cutoff_p: usize,
) -> Result<RadialScalarDeriv2> {
    mace_zbl_edge_energy_forward_deriv2_from_parts(
        distance,
        src_atomic_number,
        dst_atomic_number,
        cutoff_p,
        &MACE_ZBL_COEFFICIENTS,
        MACE_ZBL_A_EXP,
        MACE_ZBL_A_PREFACTOR,
        &MACE_COVALENT_RADII,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mace_zbl_edge_energy_forward_deriv2_from_parts(
    distance: f64,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    cutoff_p: usize,
    coefficients: &[f64],
    a_exp: f64,
    a_prefactor: f64,
    covalent_radii: &[f64],
) -> Result<RadialScalarDeriv2> {
    validate_positive_finite("mace_zbl_edge_energy_forward_deriv2 distance", distance)?;
    if coefficients.len() != MACE_ZBL_EXPONENTS.len() {
        return Err(CoreError::InvalidRadial(format!(
            "MACE ZBL coefficients must have {} values, got {}",
            MACE_ZBL_EXPONENTS.len(),
            coefficients.len()
        )));
    }
    validate_positive_finite("mace_zbl_edge_energy_forward_deriv2 a_exp", a_exp)?;
    validate_positive_finite(
        "mace_zbl_edge_energy_forward_deriv2 a_prefactor",
        a_prefactor,
    )?;
    let src_radius = mace_covalent_radius_from_parts(src_atomic_number, covalent_radii)?;
    let dst_radius = mace_covalent_radius_from_parts(dst_atomic_number, covalent_radii)?;
    let src_z = src_atomic_number as f64;
    let dst_z = dst_atomic_number as f64;
    let screening_length =
        a_prefactor * BOHR_RADIUS_ANGSTROM / (src_z.powf(a_exp) + dst_z.powf(a_exp));
    let r_over_a = distance / screening_length;
    let mut phi = 0.0;
    let mut d_phi = 0.0;
    let mut dd_phi = 0.0;
    for index in 0..MACE_ZBL_EXPONENTS.len() {
        let factor = MACE_ZBL_EXPONENTS[index] / screening_length;
        let exp_value = (-MACE_ZBL_EXPONENTS[index] * r_over_a).exp();
        phi += coefficients[index] * exp_value;
        d_phi += coefficients[index] * exp_value * -factor;
        dd_phi += coefficients[index] * exp_value * factor * factor;
    }

    let prefactor = MACE_ZBL_ENERGY_FACTOR * src_z * dst_z;
    let distance2 = distance * distance;
    let distance3 = distance2 * distance;
    let base = prefactor * phi / distance;
    let d_base = prefactor * (d_phi / distance - phi / distance2);
    let dd_base = prefactor * (dd_phi / distance - 2.0 * d_phi / distance2 + 2.0 * phi / distance3);
    let cutoff =
        mace_polynomial_cutoff_forward_deriv2(distance, src_radius + dst_radius, cutoff_p)?;
    Ok(RadialScalarDeriv2 {
        value: 0.5 * base * cutoff.value,
        derivative: 0.5 * (d_base * cutoff.value + base * cutoff.derivative),
        second_derivative: 0.5
            * (dd_base * cutoff.value
                + 2.0 * d_base * cutoff.derivative
                + base * cutoff.second_derivative),
    })
}

pub fn mace_agnesi_bessel_cutoff_basis_forward_deriv2(
    distance: f64,
    radius: f64,
    cutoff_p: usize,
    src_atomic_number: usize,
    dst_atomic_number: usize,
    out: &mut [RadialScalarDeriv2],
) -> Result<RadialScalarDeriv2> {
    if out.is_empty() {
        return Err(CoreError::InvalidRadial(
            "mace_agnesi_bessel_cutoff_basis_forward_deriv2 out must not be empty".to_owned(),
        ));
    }
    let radial_distance =
        mace_agnesi_transform_forward_deriv2(distance, src_atomic_number, dst_atomic_number)?;
    let cutoff = mace_polynomial_cutoff_forward_deriv2(distance, radius, cutoff_p)?;
    let mut bessel = vec![
        RadialScalarDeriv2 {
            value: 0.0,
            derivative: 0.0,
            second_derivative: 0.0,
        };
        out.len()
    ];
    mace_bessel_basis_forward_deriv2(radial_distance.value, radius, &mut bessel)?;
    for (target, basis) in out.iter_mut().zip(bessel) {
        let bessel_first = basis.derivative * radial_distance.derivative;
        let bessel_second =
            basis.second_derivative * radial_distance.derivative * radial_distance.derivative
                + basis.derivative * radial_distance.second_derivative;
        *target = RadialScalarDeriv2 {
            value: cutoff.value * basis.value,
            derivative: cutoff.derivative * basis.value + cutoff.value * bessel_first,
            second_derivative: cutoff.second_derivative * basis.value
                + 2.0 * cutoff.derivative * bessel_first
                + cutoff.value * bessel_second,
        };
    }
    Ok(radial_distance)
}

/// Forward pass for `b_k(d) = exp(-gamma * (d - mu_k)^2)`.
pub fn gaussian_radial_basis_forward(
    distance: f64,
    centers: &[f64],
    gamma: f64,
    out: &mut [f64],
) -> Result<()> {
    if out.len() != centers.len() {
        return Err(CoreError::ShapeMismatch {
            op: "gaussian_radial_basis_forward",
            expected: centers.len(),
            actual: out.len(),
        });
    }
    validate_distance("gaussian_radial_basis_forward distance", distance)?;
    validate_centers("gaussian_radial_basis_forward centers", centers)?;
    validate_positive_finite("gaussian_radial_basis_forward gamma", gamma)?;

    for (value, center) in out.iter_mut().zip(centers) {
        let delta = distance - center;
        *value = (-gamma * delta * delta).exp();
    }

    Ok(())
}

/// VJP for Gaussian radial basis outputs, returning the contribution to `bar_d`.
pub fn gaussian_radial_basis_vjp(
    distance: f64,
    centers: &[f64],
    gamma: f64,
    basis_values: &[f64],
    bar_basis: &[f64],
) -> Result<f64> {
    if basis_values.len() != centers.len() {
        return Err(CoreError::ShapeMismatch {
            op: "gaussian_radial_basis_vjp values",
            expected: centers.len(),
            actual: basis_values.len(),
        });
    }
    if bar_basis.len() != centers.len() {
        return Err(CoreError::ShapeMismatch {
            op: "gaussian_radial_basis_vjp cotangents",
            expected: centers.len(),
            actual: bar_basis.len(),
        });
    }
    validate_distance("gaussian_radial_basis_vjp distance", distance)?;
    validate_centers("gaussian_radial_basis_vjp centers", centers)?;
    validate_positive_finite("gaussian_radial_basis_vjp gamma", gamma)?;

    let mut bar_distance = 0.0;
    for ((center, value), bar_value) in centers.iter().zip(basis_values).zip(bar_basis) {
        bar_distance += bar_value * value * (-2.0 * gamma * (distance - center));
    }

    Ok(bar_distance)
}

/// Convenience forward for cutoff-weighted radial features `r_k = c(d) * b_k(d)`.
pub fn cutoff_gaussian_radial_forward(
    distance: f64,
    cutoff_radius: f64,
    centers: &[f64],
    gamma: f64,
    out: &mut [f64],
) -> Result<CutoffOutput> {
    let cutoff = cosine_cutoff_forward(distance, cutoff_radius)?;
    gaussian_radial_basis_forward(distance, centers, gamma, out)?;
    for value in out {
        *value *= cutoff.value;
    }
    Ok(cutoff)
}

/// VJP for cutoff-weighted radial features `r_k = c(d) * b_k(d)`.
pub fn cutoff_gaussian_radial_vjp(
    distance: f64,
    cutoff_radius: f64,
    centers: &[f64],
    gamma: f64,
    basis_values: &[f64],
    bar_weighted_basis: &[f64],
) -> Result<f64> {
    if basis_values.len() != centers.len() {
        return Err(CoreError::ShapeMismatch {
            op: "cutoff_gaussian_radial_vjp values",
            expected: centers.len(),
            actual: basis_values.len(),
        });
    }
    if bar_weighted_basis.len() != centers.len() {
        return Err(CoreError::ShapeMismatch {
            op: "cutoff_gaussian_radial_vjp cotangents",
            expected: centers.len(),
            actual: bar_weighted_basis.len(),
        });
    }
    validate_centers("cutoff_gaussian_radial_vjp centers", centers)?;
    validate_positive_finite("cutoff_gaussian_radial_vjp gamma", gamma)?;

    let cutoff = cosine_cutoff_forward(distance, cutoff_radius)?;
    let mut bar_cutoff = 0.0;
    let mut bar_distance_from_basis = 0.0;
    for index in 0..centers.len() {
        bar_cutoff += bar_weighted_basis[index] * basis_values[index];
        bar_distance_from_basis += bar_weighted_basis[index]
            * cutoff.value
            * basis_values[index]
            * (-2.0 * gamma * (distance - centers[index]));
    }

    Ok(cosine_cutoff_vjp(distance, cutoff_radius, bar_cutoff)? + bar_distance_from_basis)
}

fn check_edge_envelope_modulation_shapes(
    op: &'static str,
    feature_dim: usize,
    edge_distances: &[f64],
    cutoff: f64,
    exponent: usize,
    edge_features: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    if feature_dim == 0 {
        return Err(CoreError::InvalidRadial(
            "edge envelope feature_dim must be nonzero".to_owned(),
        ));
    }
    validate_positive_finite("edge envelope cutoff", cutoff)?;
    if exponent == 0 {
        return Err(CoreError::InvalidRadial(
            "edge envelope exponent must be positive".to_owned(),
        ));
    }
    if !edge_features.len().is_multiple_of(feature_dim) {
        return Err(CoreError::InvalidRadial(format!(
            "{op} edge_features length {} is not a multiple of feature_dim {feature_dim}",
            edge_features.len()
        )));
    }
    let edges = edge_features.len() / feature_dim;
    check_len(op, edges, edge_distances.len())?;
    check_len(op, edge_features.len(), output_like.len())?;
    validate_finite_slice("edge envelope edge_features", edge_features)?;
    validate_finite_slice("edge envelope edge_distances", edge_distances)?;
    for &distance in edge_distances {
        validate_distance("edge envelope distance", distance)?;
    }
    Ok(edges)
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

fn validate_distance(op: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(CoreError::InvalidRadial(format!(
            "{op} must be finite and non-negative, got {value}"
        )))
    }
}

fn validate_finite(op: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CoreError::InvalidRadial(format!(
            "{op} must be finite, got {value}"
        )))
    }
}

fn validate_vec3(op: &'static str, value: Vec3) -> Result<()> {
    for (axis, component) in value.iter().enumerate() {
        if !component.is_finite() {
            return Err(CoreError::InvalidRadial(format!(
                "{op} axis {axis} must be finite, got {component}"
            )));
        }
    }
    Ok(())
}

fn validate_positive_finite(op: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CoreError::InvalidRadial(format!(
            "{op} must be finite and positive, got {value}"
        )))
    }
}

fn validate_centers(op: &'static str, centers: &[f64]) -> Result<()> {
    if centers.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(CoreError::InvalidRadial(format!(
            "{op} must contain only finite values"
        )))
    }
}

fn validate_nonempty(op: &'static str, values: &[f64]) -> Result<()> {
    if values.is_empty() {
        Err(CoreError::InvalidRadial(format!("{op} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_finite_slice(op: &'static str, values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(CoreError::InvalidRadial(format!(
            "{op} must contain only finite values"
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const DEFAULT_DISTANCE_EPS: f64 = 1.0e-12;

    fn central_difference<F>(x: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = 1.0e-6;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn central_difference_with_step<F>(x: f64, h: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    struct RadialScanConfig {
        name: &'static str,
        start: f64,
        end: f64,
        samples: usize,
        h: f64,
        value_tol: f64,
        derivative_tol: f64,
        self_consistency_tol: f64,
        second_abs_limit: f64,
        second_jump_limit: f64,
    }

    fn assert_radial_1d_scan<Approx, Reference>(
        config: RadialScanConfig,
        approx: Approx,
        reference: Reference,
    ) where
        Approx: Fn(f64) -> RadialScalarOutput,
        Reference: Fn(f64) -> RadialScalarOutput,
    {
        assert!(config.samples >= 8);
        assert!(config.start < config.end);
        let mut max_value_error = 0.0;
        let mut max_derivative_error = 0.0;
        let mut max_self_error = 0.0;
        let mut max_second_abs = 0.0;
        let mut max_second_jump = 0.0;
        let mut previous_second: Option<f64> = None;
        let step = (config.end - config.start) / ((config.samples - 1) as f64);

        for index in 1..config.samples - 1 {
            let r = config.start + (index as f64) * step;
            let h = config
                .h
                .min(0.25 * (r - config.start))
                .min(0.25 * (config.end - r));
            if h <= 0.0 {
                continue;
            }
            let actual = approx(r);
            let expected = reference(r);
            assert!(
                actual.value.is_finite() && actual.derivative.is_finite(),
                "{} produced non-finite value/derivative at r={r}",
                config.name
            );
            max_value_error = f64::max(max_value_error, (actual.value - expected.value).abs());
            max_derivative_error = f64::max(
                max_derivative_error,
                (actual.derivative - expected.derivative).abs(),
            );

            let plus = approx(r + h);
            let minus = approx(r - h);
            let finite_difference = (plus.value - minus.value) / (2.0 * h);
            max_self_error = f64::max(
                max_self_error,
                (actual.derivative - finite_difference).abs(),
            );

            let second = (plus.derivative - minus.derivative) / (2.0 * h);
            assert!(
                second.is_finite(),
                "{} produced non-finite numerical R'' at r={r}",
                config.name
            );
            max_second_abs = f64::max(max_second_abs, second.abs());
            if let Some(previous) = previous_second {
                max_second_jump = f64::max(max_second_jump, (second - previous).abs());
            }
            previous_second = Some(second);
        }

        assert!(
            max_value_error <= config.value_tol,
            "{} max value error {:.3e} exceeded {:.3e}",
            config.name,
            max_value_error,
            config.value_tol
        );
        assert!(
            max_derivative_error <= config.derivative_tol,
            "{} max derivative error {:.3e} exceeded {:.3e}",
            config.name,
            max_derivative_error,
            config.derivative_tol
        );
        assert!(
            max_self_error <= config.self_consistency_tol,
            "{} derivative self-consistency error {:.3e} exceeded {:.3e}",
            config.name,
            max_self_error,
            config.self_consistency_tol
        );
        assert!(
            max_second_abs <= config.second_abs_limit,
            "{} max |R''| {:.3e} exceeded {:.3e}",
            config.name,
            max_second_abs,
            config.second_abs_limit
        );
        assert!(
            max_second_jump <= config.second_jump_limit,
            "{} max R'' grid jump {:.3e} exceeded {:.3e}",
            config.name,
            max_second_jump,
            config.second_jump_limit
        );
    }

    fn assert_radial_derivative_continuity<F>(
        name: &str,
        knot: f64,
        h: f64,
        derivative_tol: f64,
        second_tol: f64,
        f: F,
    ) where
        F: Fn(f64) -> RadialScalarOutput,
    {
        let left = f(knot - h);
        let right = f(knot + h);
        assert!(
            (left.derivative - right.derivative).abs() <= derivative_tol,
            "{name} R' discontinuity across {knot}: left {:.6e}, right {:.6e}",
            left.derivative,
            right.derivative
        );
        let second_left = (f(knot - h).derivative - f(knot - 2.0 * h).derivative) / h;
        let second_right = (f(knot + 2.0 * h).derivative - f(knot + h).derivative) / h;
        assert!(
            (second_left - second_right).abs() <= second_tol,
            "{name} R'' discontinuity across {knot}: left {:.6e}, right {:.6e}",
            second_left,
            second_right
        );
    }

    fn reference_polynomial_cutoff(distance: f64, radius: f64, p: usize) -> RadialScalarOutput {
        if distance >= radius {
            return RadialScalarOutput {
                value: 0.0,
                derivative: 0.0,
            };
        }
        let p_i = p as i32;
        let p_f = p as f64;
        let x = distance / radius;
        let coeff_a = (p_f + 1.0) * (p_f + 2.0) / 2.0;
        let coeff_b = p_f * (p_f + 2.0);
        let coeff_c = p_f * (p_f + 1.0) / 2.0;
        let x_p_minus_1 = if p_i == 1 { 1.0 } else { x.powi(p_i - 1) };
        let x_p = x_p_minus_1 * x;
        let x_p_plus_1 = x_p * x;
        let x_p_plus_2 = x_p_plus_1 * x;
        RadialScalarOutput {
            value: 1.0 - coeff_a * x_p + coeff_b * x_p_plus_1 - coeff_c * x_p_plus_2,
            derivative: (-coeff_a * p_f * x_p_minus_1 + coeff_b * (p_f + 1.0) * x_p
                - coeff_c * (p_f + 2.0) * x_p_plus_1)
                / radius,
        }
    }

    fn reference_bessel_component(distance: f64, radius: f64, index: usize) -> RadialScalarOutput {
        let n = (index + 1) as f64;
        let prefactor = (2.0 / radius).sqrt();
        let weight = std::f64::consts::PI * n / radius;
        let theta = weight * distance;
        let sin = theta.sin();
        let cos = theta.cos();
        RadialScalarOutput {
            value: prefactor * sin / distance,
            derivative: prefactor * (weight * cos * distance - sin) / (distance * distance),
        }
    }

    fn reference_agnesi(distance: f64, src_z: usize, dst_z: usize) -> RadialScalarOutput {
        let src_radius = mace_covalent_radius(src_z).unwrap();
        let dst_radius = mace_covalent_radius(dst_z).unwrap();
        let r0 = 0.5 * (src_radius + dst_radius);
        let x = distance / r0;
        let x_q = x.powf(MACE_AGNESI_Q);
        let x_q_minus_p = x.powf(MACE_AGNESI_Q - MACE_AGNESI_P);
        let denom = 1.0 + x_q_minus_p;
        let inner = MACE_AGNESI_A * x_q / denom;
        let d_inner_dx = MACE_AGNESI_A
            * (MACE_AGNESI_Q * x.powf(MACE_AGNESI_Q - 1.0) * denom
                - x_q
                    * (MACE_AGNESI_Q - MACE_AGNESI_P)
                    * x.powf(MACE_AGNESI_Q - MACE_AGNESI_P - 1.0))
            / (denom * denom);
        RadialScalarOutput {
            value: 1.0 / (1.0 + inner),
            derivative: -d_inner_dx / ((1.0 + inner) * (1.0 + inner)) / r0,
        }
    }

    fn reference_combined_radial_component(
        distance: f64,
        radius: f64,
        cutoff_p: usize,
        src_z: usize,
        dst_z: usize,
        index: usize,
    ) -> RadialScalarOutput {
        let radial_distance = reference_agnesi(distance, src_z, dst_z);
        let cutoff = reference_polynomial_cutoff(distance, radius, cutoff_p);
        let bessel = reference_bessel_component(radial_distance.value, radius, index);
        RadialScalarOutput {
            value: cutoff.value * bessel.value,
            derivative: cutoff.derivative * bessel.value
                + cutoff.value * bessel.derivative * radial_distance.derivative,
        }
    }

    fn production_combined_radial_component(
        distance: f64,
        radius: f64,
        cutoff_p: usize,
        src_z: usize,
        dst_z: usize,
        index: usize,
    ) -> RadialScalarOutput {
        let radial_distance = mace_agnesi_transform_forward(distance, src_z, dst_z).unwrap();
        let cutoff = mace_polynomial_cutoff_forward(distance, radius, cutoff_p).unwrap();
        let mut basis = vec![0.0; index + 1];
        mace_bessel_basis_forward(radial_distance.value, radius, &mut basis).unwrap();
        let mut bar_basis = vec![0.0; index + 1];
        bar_basis[index] = cutoff.value;
        let bar_distance =
            mace_bessel_basis_vjp(radial_distance.value, radius, &basis, &bar_basis).unwrap();
        RadialScalarOutput {
            value: cutoff.value * basis[index],
            derivative: cutoff.derivative * basis[index]
                + radial_distance.derivative * bar_distance,
        }
    }

    fn reference_zbl(
        distance: f64,
        src_z: usize,
        dst_z: usize,
        cutoff_p: usize,
    ) -> RadialScalarOutput {
        let src_radius = mace_covalent_radius(src_z).unwrap();
        let dst_radius = mace_covalent_radius(dst_z).unwrap();
        let src_z_f = src_z as f64;
        let dst_z_f = dst_z as f64;
        let screening_length = MACE_ZBL_A_PREFACTOR * BOHR_RADIUS_ANGSTROM
            / (src_z_f.powf(MACE_ZBL_A_EXP) + dst_z_f.powf(MACE_ZBL_A_EXP));
        let r_over_a = distance / screening_length;
        let mut phi = 0.0;
        let mut d_phi = 0.0;
        for index in 0..MACE_ZBL_COEFFICIENTS.len() {
            let exp_value = (-MACE_ZBL_EXPONENTS[index] * r_over_a).exp();
            phi += MACE_ZBL_COEFFICIENTS[index] * exp_value;
            d_phi += MACE_ZBL_COEFFICIENTS[index]
                * exp_value
                * (-MACE_ZBL_EXPONENTS[index] / screening_length);
        }
        let prefactor = MACE_ZBL_ENERGY_FACTOR * src_z_f * dst_z_f;
        let base = prefactor * phi / distance;
        let d_base = prefactor * (d_phi * distance - phi) / (distance * distance);
        let cutoff = reference_polynomial_cutoff(distance, src_radius + dst_radius, cutoff_p);
        RadialScalarOutput {
            value: 0.5 * base * cutoff.value,
            derivative: 0.5 * (d_base * cutoff.value + base * cutoff.derivative),
        }
    }

    #[test]
    fn distance_basic() {
        let out = distance_forward([3.0, 4.0, 0.0], DEFAULT_DISTANCE_EPS).unwrap();
        assert!((out.distance - 5.0).abs() < 1e-12);
        assert!((out.unit[0] - 0.6).abs() < 1e-12);
        assert!((out.unit[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn distance_vjp_matches_finite_difference() {
        let r = [1.2, -0.7, 2.5];
        let bar_distance = -1.7;
        let analytic = distance_vjp(r, bar_distance, DEFAULT_DISTANCE_EPS).unwrap();
        for axis in 0..3 {
            let fd = central_difference(r[axis], |value| {
                let mut perturbed = r;
                perturbed[axis] = value;
                bar_distance
                    * distance_forward(perturbed, DEFAULT_DISTANCE_EPS)
                        .unwrap()
                        .distance
            });
            assert!((analytic[axis] - fd).abs() < 1.0e-8);
        }
    }

    #[test]
    fn distance_vjp_uses_documented_near_zero_policy() {
        let r = [1.0e-14, -2.0e-14, 0.0];
        let bar_r = distance_vjp(r, 2.0, DEFAULT_DISTANCE_EPS).unwrap();
        assert_eq!(bar_r, [0.02, -0.04, 0.0]);
    }

    #[test]
    fn distance_vjp_accumulates_into_existing_bar_r() {
        let mut bar_r = [10.0, -5.0, 0.25];
        distance_vjp_accumulate([3.0, 4.0, 0.0], 2.0, DEFAULT_DISTANCE_EPS, &mut bar_r).unwrap();

        assert_eq!(bar_r, [11.2, -3.4, 0.25]);
    }

    #[test]
    fn distance_jvp_and_linearized_vjp_match_finite_difference() {
        let r = [1.2, -0.7, 2.5];
        let r_dot = [0.4, -0.3, 0.2];
        let bar_distance = -1.7;
        let bar_distance_dot = 0.6;
        let jvp = distance_forward_jvp(r, r_dot, DEFAULT_DISTANCE_EPS).unwrap();
        let fd_jvp = central_difference_with_step(0.0, 1.0e-5, |eps| {
            distance_forward(
                [
                    r[0] + eps * r_dot[0],
                    r[1] + eps * r_dot[1],
                    r[2] + eps * r_dot[2],
                ],
                DEFAULT_DISTANCE_EPS,
            )
            .unwrap()
            .distance
        });
        assert!((jvp.tangent - fd_jvp).abs() < 1.0e-10);

        let analytic = distance_vjp_linearize(
            r,
            r_dot,
            bar_distance,
            bar_distance_dot,
            DEFAULT_DISTANCE_EPS,
        )
        .unwrap();
        for (axis, &analytic_value) in analytic.iter().enumerate() {
            let fd = central_difference_with_step(0.0, 1.0e-5, |eps| {
                let perturbed_r = [
                    r[0] + eps * r_dot[0],
                    r[1] + eps * r_dot[1],
                    r[2] + eps * r_dot[2],
                ];
                let perturbed_bar = bar_distance + eps * bar_distance_dot;
                distance_vjp(perturbed_r, perturbed_bar, DEFAULT_DISTANCE_EPS).unwrap()[axis]
            });
            assert!((analytic_value - fd).abs() < 5.0e-10);
        }
    }

    #[test]
    fn distance_jvp_and_linearized_vjp_use_near_zero_policy() {
        let r = [1.0e-15, -2.0e-15, 3.0e-15];
        let r_dot = [0.4, -0.3, 0.2];
        let eps = DEFAULT_DISTANCE_EPS;
        let jvp = distance_forward_jvp(r, r_dot, eps).unwrap();
        assert!((jvp.tangent - dot(jvp.unit, r_dot)).abs() < 1.0e-30);

        let analytic = distance_vjp_linearize(r, r_dot, 2.0, -0.5, eps).unwrap();
        for axis in 0..3 {
            let expected = -0.5 * jvp.unit[axis] + 2.0 * r_dot[axis] / eps;
            assert!((analytic[axis] - expected).abs() < 1.0e-3);
        }
    }

    #[test]
    fn distance_rejects_invalid_eps_and_non_finite_values() {
        assert!(matches!(
            distance_forward([0.0, 0.0, 0.0], 0.0),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            distance_forward([f64::NAN, 0.0, 0.0], DEFAULT_DISTANCE_EPS),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            distance_vjp([1.0, 0.0, 0.0], f64::INFINITY, DEFAULT_DISTANCE_EPS),
            Err(CoreError::InvalidRadial(_))
        ));
    }

    #[test]
    fn cosine_cutoff_zero_at_and_beyond_cutoff() {
        let at = cosine_cutoff_forward(5.0, 5.0).unwrap();
        let beyond = cosine_cutoff_forward(6.0, 5.0).unwrap();
        assert_eq!(at.value, 0.0);
        assert_eq!(at.derivative, 0.0);
        assert_eq!(beyond.value, 0.0);
        assert_eq!(beyond.derivative, 0.0);
    }

    #[test]
    fn cosine_cutoff_derivative_matches_finite_difference() {
        let distance = 1.3;
        let radius = 4.0;
        let out = cosine_cutoff_forward(distance, radius).unwrap();
        let fd = central_difference(distance, |d| {
            cosine_cutoff_forward(d, radius).unwrap().value
        });
        assert!((out.derivative - fd).abs() < 1.0e-10);
    }

    #[test]
    fn gaussian_radial_basis_forward_and_vjp_match_finite_difference() {
        let centers = [0.5, 1.0, 1.5];
        let gamma = 0.7;
        let distance = 1.2;
        let bar_basis = [0.3, -0.4, 0.8];
        let mut values = [0.0; 3];
        gaussian_radial_basis_forward(distance, &centers, gamma, &mut values).unwrap();

        let analytic =
            gaussian_radial_basis_vjp(distance, &centers, gamma, &values, &bar_basis).unwrap();
        let fd = central_difference(distance, |d| {
            let mut perturbed = [0.0; 3];
            gaussian_radial_basis_forward(d, &centers, gamma, &mut perturbed).unwrap();
            perturbed
                .iter()
                .zip(bar_basis)
                .map(|(value, bar_value)| *value * bar_value)
                .sum::<f64>()
        });

        assert!((analytic - fd).abs() < 1.0e-10);
    }

    #[test]
    fn cutoff_weighted_radial_vjp_matches_finite_difference() {
        let centers = [0.5, 1.0, 1.5, 2.0];
        let gamma = 0.8;
        let distance = 1.4;
        let cutoff_radius = 3.0;
        let bar_weighted = [0.2, -0.6, 0.1, 0.5];

        let mut basis_values = [0.0; 4];
        gaussian_radial_basis_forward(distance, &centers, gamma, &mut basis_values).unwrap();
        let analytic = cutoff_gaussian_radial_vjp(
            distance,
            cutoff_radius,
            &centers,
            gamma,
            &basis_values,
            &bar_weighted,
        )
        .unwrap();

        let fd = central_difference(distance, |d| {
            let mut weighted = [0.0; 4];
            cutoff_gaussian_radial_forward(d, cutoff_radius, &centers, gamma, &mut weighted)
                .unwrap();
            weighted
                .iter()
                .zip(bar_weighted)
                .map(|(value, bar_value)| *value * bar_value)
                .sum::<f64>()
        });

        assert!((analytic - fd).abs() < 1.0e-10);
    }

    #[test]
    fn mace_agnesi_transform_vjp_matches_finite_difference() {
        let distance = 0.74;
        let output = mace_agnesi_transform_forward(distance, 6, 8).unwrap();
        let fd = central_difference(distance, |d| {
            mace_agnesi_transform_forward(d, 6, 8).unwrap().value
        });
        assert!((output.derivative - fd).abs() < 1.0e-8);
    }

    #[test]
    fn mace_zbl_edge_energy_vjp_matches_finite_difference() {
        let distance = 0.57;
        let output = mace_zbl_edge_energy_forward(distance, 1, 8, 6).unwrap();
        let fd = central_difference(distance, |d| {
            mace_zbl_edge_energy_forward(d, 1, 8, 6).unwrap().value
        });
        assert!((output.derivative - fd).abs() < 1.0e-7);
    }

    #[test]
    fn mace_bessel_cutoff_basis_vjp_matches_finite_difference() {
        let distance = 0.74;
        let radius = 5.0;
        let cutoff_p = 6;
        let bar_embedding = [0.2, -0.3, 0.4, -0.5];
        let mut basis = [0.0; 4];
        mace_bessel_basis_forward(distance, radius, &mut basis).unwrap();
        let analytic =
            mace_bessel_cutoff_basis_vjp(distance, radius, cutoff_p, &basis, &bar_embedding)
                .unwrap();

        let fd = central_difference(distance, |d| {
            let mut embedding = [0.0; 4];
            mace_bessel_cutoff_basis_forward(d, radius, cutoff_p, &mut embedding).unwrap();
            embedding
                .iter()
                .zip(bar_embedding)
                .map(|(value, bar)| value * bar)
                .sum::<f64>()
        });

        assert!((analytic - fd).abs() < 1.0e-9);
    }

    #[test]
    fn mace_polynomial_cutoff_deriv2_matches_finite_difference() {
        let distance = 1.3;
        let radius = 6.0;
        let cutoff_p = 6;
        let deriv2 = mace_polynomial_cutoff_forward_deriv2(distance, radius, cutoff_p).unwrap();
        let existing = mace_polynomial_cutoff_forward(distance, radius, cutoff_p).unwrap();
        assert!((deriv2.value - existing.value).abs() < 1.0e-14);
        assert!((deriv2.derivative - existing.derivative).abs() < 1.0e-14);
        let fd = central_difference(distance, |d| {
            mace_polynomial_cutoff_forward(d, radius, cutoff_p)
                .unwrap()
                .derivative
        });
        assert!((deriv2.second_derivative - fd).abs() < 1.0e-9);
    }

    #[test]
    fn mace_bessel_basis_deriv2_matches_finite_difference() {
        let distance = 0.91;
        let radius = 5.0;
        let mut deriv2 = vec![
            RadialScalarDeriv2 {
                value: 0.0,
                derivative: 0.0,
                second_derivative: 0.0,
            };
            10
        ];
        mace_bessel_basis_forward_deriv2(distance, radius, &mut deriv2).unwrap();
        let mut values = vec![0.0; deriv2.len()];
        mace_bessel_basis_forward(distance, radius, &mut values).unwrap();
        for index in [0, 3, 9] {
            assert!((deriv2[index].value - values[index]).abs() < 1.0e-14);
            let fd_first = central_difference(distance, |d| {
                let mut basis = vec![0.0; deriv2.len()];
                mace_bessel_basis_forward(d, radius, &mut basis).unwrap();
                basis[index]
            });
            let fd_second = central_difference(distance, |d| {
                let mut basis_deriv2 = vec![
                    RadialScalarDeriv2 {
                        value: 0.0,
                        derivative: 0.0,
                        second_derivative: 0.0,
                    };
                    deriv2.len()
                ];
                mace_bessel_basis_forward_deriv2(d, radius, &mut basis_deriv2).unwrap();
                basis_deriv2[index].derivative
            });
            assert!((deriv2[index].derivative - fd_first).abs() < 1.0e-8);
            assert!((deriv2[index].second_derivative - fd_second).abs() < 2.0e-7);
        }
    }

    #[test]
    fn mace_bessel_cutoff_basis_deriv2_matches_finite_difference() {
        let distance = 1.17;
        let radius = 6.0;
        let cutoff_p = 6;
        let mut deriv2 = vec![
            RadialScalarDeriv2 {
                value: 0.0,
                derivative: 0.0,
                second_derivative: 0.0,
            };
            10
        ];
        mace_bessel_cutoff_basis_forward_deriv2(distance, radius, cutoff_p, &mut deriv2).unwrap();
        for index in [0, 4, 9] {
            let fd_first = central_difference(distance, |d| {
                let mut basis = vec![0.0; deriv2.len()];
                mace_bessel_cutoff_basis_forward(d, radius, cutoff_p, &mut basis).unwrap();
                basis[index]
            });
            let fd_second = central_difference(distance, |d| {
                let mut basis_deriv2 = vec![
                    RadialScalarDeriv2 {
                        value: 0.0,
                        derivative: 0.0,
                        second_derivative: 0.0,
                    };
                    deriv2.len()
                ];
                mace_bessel_cutoff_basis_forward_deriv2(d, radius, cutoff_p, &mut basis_deriv2)
                    .unwrap();
                basis_deriv2[index].derivative
            });
            assert!((deriv2[index].derivative - fd_first).abs() < 1.0e-8);
            assert!((deriv2[index].second_derivative - fd_second).abs() < 3.0e-7);
        }
    }

    #[test]
    fn mace_agnesi_transform_deriv2_matches_finite_difference() {
        for (distance, src_z, dst_z) in [(0.74, 1, 1), (1.35, 1, 98), (2.2, 26, 8)] {
            let deriv2 = mace_agnesi_transform_forward_deriv2(distance, src_z, dst_z).unwrap();
            let existing = mace_agnesi_transform_forward(distance, src_z, dst_z).unwrap();
            assert!((deriv2.value - existing.value).abs() < 1.0e-14);
            assert!((deriv2.derivative - existing.derivative).abs() < 1.0e-14);
            let fd = central_difference_with_step(distance, 3.0e-6, |d| {
                mace_agnesi_transform_forward(d, src_z, dst_z)
                    .unwrap()
                    .derivative
            });
            assert!(
                (deriv2.second_derivative - fd).abs() < 2.0e-8,
                "Agnesi R'' mismatch for {src_z}-{dst_z} at {distance}: analytic {:.8e}, fd {:.8e}",
                deriv2.second_derivative,
                fd
            );
        }
    }

    #[test]
    fn mace_zbl_edge_energy_deriv2_matches_finite_difference() {
        for (distance, src_z, dst_z) in [(0.57, 1, 8), (1.1, 8, 26), (1.8, 6, 98)] {
            let deriv2 = mace_zbl_edge_energy_forward_deriv2(distance, src_z, dst_z, 6).unwrap();
            let existing = mace_zbl_edge_energy_forward(distance, src_z, dst_z, 6).unwrap();
            assert!((deriv2.value - existing.value).abs() < 1.0e-12);
            assert!((deriv2.derivative - existing.derivative).abs() < 1.0e-10);
            let fd = central_difference_with_step(distance, 2.0e-6, |d| {
                mace_zbl_edge_energy_forward(d, src_z, dst_z, 6)
                    .unwrap()
                    .derivative
            });
            assert!(
                (deriv2.second_derivative - fd).abs() < 2.0e-4,
                "ZBL R'' mismatch for {src_z}-{dst_z} at {distance}: analytic {:.8e}, fd {:.8e}",
                deriv2.second_derivative,
                fd
            );
        }
    }

    #[test]
    fn mace_agnesi_bessel_cutoff_basis_deriv2_matches_finite_difference() {
        let distance = 1.24;
        let radius = 6.0;
        let cutoff_p = 6;
        let src_z = 8;
        let dst_z = 1;
        let mut deriv2 = vec![
            RadialScalarDeriv2 {
                value: 0.0,
                derivative: 0.0,
                second_derivative: 0.0,
            };
            10
        ];
        let radial_distance = mace_agnesi_bessel_cutoff_basis_forward_deriv2(
            distance,
            radius,
            cutoff_p,
            src_z,
            dst_z,
            &mut deriv2,
        )
        .unwrap();
        let existing_radial_distance =
            mace_agnesi_transform_forward(distance, src_z, dst_z).unwrap();
        assert!((radial_distance.value - existing_radial_distance.value).abs() < 1.0e-14);
        assert!((radial_distance.derivative - existing_radial_distance.derivative).abs() < 1.0e-14);

        for index in [0, 4, 9] {
            let fd_first = central_difference(distance, |d| {
                production_combined_radial_component(d, radius, cutoff_p, src_z, dst_z, index).value
            });
            let fd_second = central_difference_with_step(distance, 3.0e-6, |d| {
                production_combined_radial_component(d, radius, cutoff_p, src_z, dst_z, index)
                    .derivative
            });
            assert!((deriv2[index].derivative - fd_first).abs() < 1.0e-8);
            assert!(
                (deriv2[index].second_derivative - fd_second).abs() < 3.0e-6,
                "combined radial R'' mismatch at index {index}: analytic {:.8e}, fd {:.8e}",
                deriv2[index].second_derivative,
                fd_second
            );
        }
    }

    #[test]
    fn mace_osaka26_radial_1d_scans_cover_values_derivatives_and_smoothness() {
        let radius = 6.0;
        let cutoff_p = 6;
        let src_z = 8;
        let dst_z = 1;
        let base_scan = |name| RadialScanConfig {
            name,
            start: 5.0e-2,
            end: radius,
            samples: 401,
            h: 1.0e-5,
            value_tol: 1.0e-12,
            derivative_tol: 1.0e-10,
            self_consistency_tol: 2.5e-8,
            second_abs_limit: 5.0e4,
            second_jump_limit: 5.0e3,
        };

        assert_radial_1d_scan(
            RadialScanConfig {
                name: "MACE Agnesi distance transform",
                second_abs_limit: 1.0e1,
                second_jump_limit: 2.0,
                ..base_scan("MACE Agnesi distance transform")
            },
            |r| mace_agnesi_transform_forward(r, src_z, dst_z).unwrap(),
            |r| reference_agnesi(r, src_z, dst_z),
        );

        for index in [0, 4, 9] {
            assert_radial_1d_scan(
                RadialScanConfig {
                    name: "MACE Bessel after Agnesi transform",
                    derivative_tol: 1.0e-9,
                    self_consistency_tol: 2.0e-6,
                    second_abs_limit: 1.0e4,
                    second_jump_limit: 1.0e3,
                    ..base_scan("MACE Bessel after Agnesi transform")
                },
                |r| {
                    let radial_distance = mace_agnesi_transform_forward(r, src_z, dst_z).unwrap();
                    let mut basis = vec![0.0; index + 1];
                    mace_bessel_basis_forward(radial_distance.value, radius, &mut basis).unwrap();
                    let mut bar_basis = vec![0.0; index + 1];
                    bar_basis[index] = 1.0;
                    let derivative =
                        mace_bessel_basis_vjp(radial_distance.value, radius, &basis, &bar_basis)
                            .unwrap()
                            * radial_distance.derivative;
                    RadialScalarOutput {
                        value: basis[index],
                        derivative,
                    }
                },
                |r| {
                    let radial_distance = reference_agnesi(r, src_z, dst_z);
                    let bessel = reference_bessel_component(radial_distance.value, radius, index);
                    RadialScalarOutput {
                        value: bessel.value,
                        derivative: bessel.derivative * radial_distance.derivative,
                    }
                },
            );

            assert_radial_1d_scan(
                RadialScanConfig {
                    name: "MACE combined radial embedding",
                    derivative_tol: 1.0e-9,
                    self_consistency_tol: 2.0e-6,
                    second_abs_limit: 1.0e4,
                    second_jump_limit: 1.0e3,
                    ..base_scan("MACE combined radial embedding")
                },
                |r| production_combined_radial_component(r, radius, cutoff_p, src_z, dst_z, index),
                |r| reference_combined_radial_component(r, radius, cutoff_p, src_z, dst_z, index),
            );
        }

        assert_radial_1d_scan(
            RadialScanConfig {
                name: "MACE polynomial cutoff",
                start: 0.0,
                second_abs_limit: 1.0,
                second_jump_limit: 0.05,
                ..base_scan("MACE polynomial cutoff")
            },
            |r| {
                let cutoff = mace_polynomial_cutoff_forward(r, radius, cutoff_p).unwrap();
                RadialScalarOutput {
                    value: cutoff.value,
                    derivative: cutoff.derivative,
                }
            },
            |r| reference_polynomial_cutoff(r, radius, cutoff_p),
        );
        assert_radial_derivative_continuity(
            "MACE polynomial cutoff",
            radius,
            1.0e-5,
            1.0e-9,
            1.0e-2,
            |r| {
                let cutoff = mace_polynomial_cutoff_forward(r, radius, cutoff_p).unwrap();
                RadialScalarOutput {
                    value: cutoff.value,
                    derivative: cutoff.derivative,
                }
            },
        );

        assert_radial_1d_scan(
            RadialScanConfig {
                name: "MACE active ZBL pair repulsion",
                start: 0.2,
                value_tol: 1.0e-10,
                derivative_tol: 1.0e-8,
                self_consistency_tol: 2.5e-5,
                second_abs_limit: 2.0e5,
                second_jump_limit: 5.0e4,
                ..base_scan("MACE active ZBL pair repulsion")
            },
            |r| mace_zbl_edge_energy_forward(r, src_z, dst_z, cutoff_p).unwrap(),
            |r| reference_zbl(r, src_z, dst_z, cutoff_p),
        );
        let zbl_cutoff =
            mace_covalent_radius(src_z).unwrap() + mace_covalent_radius(dst_z).unwrap();
        assert_radial_derivative_continuity(
            "MACE active ZBL pair repulsion cutoff",
            zbl_cutoff,
            1.0e-5,
            1.0e-7,
            1.0e-2,
            |r| mace_zbl_edge_energy_forward(r, src_z, dst_z, cutoff_p).unwrap(),
        );
    }

    #[test]
    fn mace_polynomial_cutoff_zero_at_and_beyond_cutoff() {
        let at = mace_polynomial_cutoff_forward(1.07, 1.07, 6).unwrap();
        let beyond = mace_polynomial_cutoff_forward(1.2, 1.07, 6).unwrap();
        assert_eq!(at.value, 0.0);
        assert_eq!(at.derivative, 0.0);
        assert_eq!(beyond.value, 0.0);
        assert_eq!(beyond.derivative, 0.0);
    }

    #[test]
    fn matches_pinned_mace_radial_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../fixtures/mace/v0_3_16_4d2da09/mace_radial_reference_f64.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema_version"], "mace-radial-reference-v1");

        let species_atomic_numbers = json_usize_vec(&fixture["graph"]["species_atomic_numbers"]);
        let node_type_indices = json_usize_vec(&fixture["graph"]["node_type_indices"]);
        let edge_index = json_usize_matrix(&fixture["graph"]["edge_index"]);
        let distances = json_f64_column(&fixture["graph"]["distances"]);
        let agnesi_outputs = json_f64_column(&fixture["agnesi"]["output"]);
        let agnesi_cotangent = json_f64_column(&fixture["agnesi"]["cotangent"]);
        let agnesi_vjp = json_f64_column(&fixture["agnesi"]["vjp_distances"]);
        let bessel_basis = json_f64_matrix(&fixture["bessel"]["basis"]);
        let bessel_cutoff = json_f64_column(&fixture["bessel"]["cutoff"]);
        let bessel_embedding = json_f64_matrix(&fixture["bessel"]["embedding"]);
        let bessel_cotangent = json_f64_matrix(&fixture["bessel"]["cotangent_embedding"]);
        let bessel_vjp = json_f64_column(&fixture["bessel"]["vjp_distances"]);
        let zbl_outputs = json_f64_vec(&fixture["zbl"]["output_node_energy"]);
        let zbl_cotangent = json_f64_column(&fixture["zbl"]["cotangent_node_energy"]);
        let zbl_vjp = json_f64_column(&fixture["zbl"]["vjp_distances"]);

        let mut zbl_node_energy = vec![0.0; node_type_indices.len()];
        let mut actual_zbl_vjp = vec![0.0; distances.len()];
        for edge in 0..distances.len() {
            let src = edge_index[0][edge];
            let dst = edge_index[1][edge];
            let src_z = species_atomic_numbers[node_type_indices[src]];
            let dst_z = species_atomic_numbers[node_type_indices[dst]];

            let agnesi = mace_agnesi_transform_forward(distances[edge], src_z, dst_z).unwrap();
            assert_close(agnesi.value, agnesi_outputs[edge], 1.0e-12);
            assert_close(
                agnesi.derivative * agnesi_cotangent[edge],
                agnesi_vjp[edge],
                1.0e-10,
            );

            let mut actual_bessel_basis = vec![0.0; bessel_basis[edge].len()];
            mace_bessel_basis_forward(distances[edge], 5.0, &mut actual_bessel_basis).unwrap();
            for (basis_index, actual) in actual_bessel_basis.iter().enumerate() {
                assert_close(*actual, bessel_basis[edge][basis_index], 1.0e-12);
            }
            let mut actual_bessel_embedding = vec![0.0; bessel_embedding[edge].len()];
            let cutoff = mace_bessel_cutoff_basis_forward(
                distances[edge],
                5.0,
                6,
                &mut actual_bessel_embedding,
            )
            .unwrap();
            assert_close(cutoff.value, bessel_cutoff[edge], 1.0e-12);
            for (basis_index, actual) in actual_bessel_embedding.iter().enumerate() {
                assert_close(*actual, bessel_embedding[edge][basis_index], 1.0e-12);
            }
            let actual_bessel_vjp = mace_bessel_cutoff_basis_vjp(
                distances[edge],
                5.0,
                6,
                &actual_bessel_basis,
                &bessel_cotangent[edge],
            )
            .unwrap();
            assert_close(actual_bessel_vjp, bessel_vjp[edge], 1.0e-10);

            let zbl = mace_zbl_edge_energy_forward(distances[edge], src_z, dst_z, 6).unwrap();
            zbl_node_energy[dst] += zbl.value;
            actual_zbl_vjp[edge] = zbl.derivative * zbl_cotangent[dst];
        }

        for node in 0..zbl_outputs.len() {
            assert_close(zbl_node_energy[node], zbl_outputs[node], 1.0e-10);
        }
        for edge in 0..zbl_vjp.len() {
            assert_close(actual_zbl_vjp[edge], zbl_vjp[edge], 1.0e-8);
        }
    }

    #[test]
    fn radial_shape_mismatch_is_reported() {
        let err = gaussian_radial_basis_forward(1.0, &[0.0, 1.0], 1.0, &mut [0.0])
            .expect_err("shape mismatch should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn invalid_radial_parameters_are_reported() {
        let mut out = [0.0; 1];

        let cutoff_err =
            cosine_cutoff_forward(1.0, 0.0).expect_err("non-positive cutoff radius should fail");
        assert!(matches!(cutoff_err, CoreError::InvalidRadial(_)));

        let gamma_err = gaussian_radial_basis_forward(1.0, &[0.5], 0.0, &mut out)
            .expect_err("non-positive gamma should fail");
        assert!(matches!(gamma_err, CoreError::InvalidRadial(_)));

        let center_err = gaussian_radial_basis_forward(1.0, &[f64::NAN], 1.0, &mut out)
            .expect_err("non-finite center should fail");
        assert!(matches!(center_err, CoreError::InvalidRadial(_)));
    }

    #[test]
    fn radial_vjp_shape_mismatches_are_reported() {
        let centers = [0.0, 1.0];
        let values = [1.0, 0.5];
        let bars = [0.25, -0.75];

        let value_err = gaussian_radial_basis_vjp(1.0, &centers, 1.0, &[1.0], &bars)
            .expect_err("basis value shape mismatch should fail");
        assert!(matches!(value_err, CoreError::ShapeMismatch { .. }));

        let cotangent_err = gaussian_radial_basis_vjp(1.0, &centers, 1.0, &values, &[0.25])
            .expect_err("basis cotangent shape mismatch should fail");
        assert!(matches!(cotangent_err, CoreError::ShapeMismatch { .. }));

        let cutoff_value_err = cutoff_gaussian_radial_vjp(1.0, 2.0, &centers, 1.0, &[1.0], &bars)
            .expect_err("cutoff basis value shape mismatch should fail");
        assert!(matches!(cutoff_value_err, CoreError::ShapeMismatch { .. }));

        let cutoff_cotangent_err =
            cutoff_gaussian_radial_vjp(1.0, 2.0, &centers, 1.0, &values, &[0.25])
                .expect_err("cutoff basis cotangent shape mismatch should fail");
        assert!(matches!(
            cutoff_cotangent_err,
            CoreError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn mace_radial_rejects_invalid_inputs() {
        let mut out = [0.0; 1];
        assert!(matches!(
            mace_agnesi_transform_forward(0.0, 1, 8),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            mace_agnesi_transform_forward(0.5, 200, 8),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            mace_zbl_edge_energy_forward(0.5, 1, 200, 6),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            mace_polynomial_cutoff_forward(0.5, 1.0, 0),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            mace_bessel_basis_forward(0.0, 5.0, &mut out),
            Err(CoreError::InvalidRadial(_))
        ));
        assert!(matches!(
            mace_bessel_basis_forward(0.5, 5.0, &mut []),
            Err(CoreError::InvalidRadial(_))
        ));
    }

    #[test]
    fn mace_covalent_radii_cover_osaka26_species() {
        assert_close(mace_covalent_radius(96).unwrap(), 1.69, 1.0e-12);
        assert_close(mace_covalent_radius(98).unwrap(), 0.20, 1.0e-12);
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:.16e}, got {actual:.16e}, diff {:.3e}, tolerance {tolerance:.3e}",
            (actual - expected).abs()
        );
    }

    fn json_f64_vec(value: &serde_json::Value) -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_f64().unwrap())
            .collect()
    }

    fn json_f64_column(value: &serde_json::Value) -> Vec<f64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap()[0].as_f64().unwrap())
            .collect()
    }

    fn json_usize_vec(value: &serde_json::Value) -> Vec<usize> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap() as usize)
            .collect()
    }

    fn json_usize_matrix(value: &serde_json::Value) -> Vec<Vec<usize>> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(json_usize_vec)
            .collect()
    }

    fn json_f64_matrix(value: &serde_json::Value) -> Vec<Vec<f64>> {
        value.as_array().unwrap().iter().map(json_f64_vec).collect()
    }
}
