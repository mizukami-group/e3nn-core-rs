//! Real spherical harmonics for the v0.1 CPU/f64 force path.
//!
//! The implemented convention is fixture-pinned against e3nn `0.6.0` with
//! `normalization="component"` and `normalize=true`.

use crate::geometry::{dot, Vec3};
use crate::{CoreError, Result};

const DEFAULT_EPS: f64 = 1.0e-12;
#[cfg(test)]
const SQRT_3: f64 = 1.732_050_807_568_877_2;
#[cfg(test)]
const SQRT_5: f64 = 2.236_067_977_499_79;
#[cfg(test)]
const HALF_SQRT_5: f64 = 1.118_033_988_749_895;
#[cfg(test)]
const HALF_SQRT_15: f64 = 1.936_491_673_103_708_5;

#[derive(Clone, Copy, Debug)]
struct ShPolynomialTerm {
    output: usize,
    coeff: f64,
    x_pow: u8,
    y_pow: u8,
    z_pow: u8,
}

// Coefficients are fitted from the pinned e3nn reference implementation for
// normalization="component" and normalize=false, then used on either raw r or
// normalized unit vectors according to SphericalHarmonicsPlan::normalize_input.
const SH_COMPONENT_TERMS: &[ShPolynomialTerm] = &[
    ShPolynomialTerm {
        output: 0,
        coeff: 1.0,
        x_pow: 0,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 1,
        coeff: 1.732_050_807_568_876_5,
        x_pow: 1,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 2,
        coeff: 1.732_050_807_568_877,
        x_pow: 0,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 3,
        coeff: 1.732_050_807_568_876_5,
        x_pow: 0,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 4,
        coeff: 3.872_983_346_207_422_4,
        x_pow: 1,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 5,
        coeff: 3.872_983_346_207_419,
        x_pow: 1,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 6,
        coeff: -1.118_033_988_749_894_5,
        x_pow: 2,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 6,
        coeff: 2.236_067_977_499_789,
        x_pow: 0,
        y_pow: 2,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 6,
        coeff: -1.118_033_988_749_895,
        x_pow: 0,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 7,
        coeff: 3.872_983_346_207_416_6,
        x_pow: 0,
        y_pow: 1,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 8,
        coeff: -1.936_491_673_103_708_5,
        x_pow: 2,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 8,
        coeff: 1.936_491_673_103_707_8,
        x_pow: 0,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 9,
        coeff: -2.091_650_066_335_189_4,
        x_pow: 3,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 9,
        coeff: 6.274_950_199_005_567,
        x_pow: 1,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 10,
        coeff: 10.246_950_765_959_598,
        x_pow: 1,
        y_pow: 1,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 11,
        coeff: -1.620_185_174_601_965_8,
        x_pow: 3,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 11,
        coeff: 6.480_740_698_407_86,
        x_pow: 1,
        y_pow: 2,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 11,
        coeff: -1.620_185_174_601_965,
        x_pow: 1,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 12,
        coeff: -3.968_626_966_596_884_7,
        x_pow: 2,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 12,
        coeff: 2.645_751_311_064_59,
        x_pow: 0,
        y_pow: 3,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 12,
        coeff: -3.968_626_966_596_887_4,
        x_pow: 0,
        y_pow: 1,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 13,
        coeff: -1.620_185_174_601_965_8,
        x_pow: 2,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 13,
        coeff: 6.480_740_698_407_861,
        x_pow: 0,
        y_pow: 2,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 13,
        coeff: -1.620_185_174_601_964_6,
        x_pow: 0,
        y_pow: 0,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 14,
        coeff: -5.123_475_382_979_797,
        x_pow: 2,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 14,
        coeff: 5.123_475_382_979_8,
        x_pow: 0,
        y_pow: 1,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 15,
        coeff: -6.274_950_199_005_564_5,
        x_pow: 2,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 15,
        coeff: 2.091_650_066_335_187,
        x_pow: 0,
        y_pow: 0,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 16,
        coeff: -8.874_119_674_649_426,
        x_pow: 3,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 16,
        coeff: 8.874_119_674_649_423,
        x_pow: 1,
        y_pow: 0,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 17,
        coeff: -6.274_950_199_005_573,
        x_pow: 3,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 17,
        coeff: 18.824_850_597_016_71,
        x_pow: 1,
        y_pow: 1,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 18,
        coeff: -3.354_101_966_249_685,
        x_pow: 3,
        y_pow: 0,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 18,
        coeff: 20.124_611_797_498_11,
        x_pow: 1,
        y_pow: 2,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 18,
        coeff: -3.354_101_966_249_683_4,
        x_pow: 1,
        y_pow: 0,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 19,
        coeff: -7.115_124_735_378_858,
        x_pow: 3,
        y_pow: 1,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 19,
        coeff: 9.486_832_980_505_14,
        x_pow: 1,
        y_pow: 3,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 19,
        coeff: -7.115_124_735_378_856,
        x_pow: 1,
        y_pow: 1,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: 1.124_999_999_999_999_3,
        x_pow: 4,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: -8.999_999_999_999_995,
        x_pow: 2,
        y_pow: 2,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: 2.249_999_999_999_998_7,
        x_pow: 2,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: 3.000_000_000_000_001_8,
        x_pow: 0,
        y_pow: 4,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: -9.000_000_000_000_004,
        x_pow: 0,
        y_pow: 2,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 20,
        coeff: 1.125_000_000_000_000_2,
        x_pow: 0,
        y_pow: 0,
        z_pow: 4,
    },
    ShPolynomialTerm {
        output: 21,
        coeff: -7.115_124_735_378_857,
        x_pow: 2,
        y_pow: 1,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 21,
        coeff: 9.486_832_980_505_136,
        x_pow: 0,
        y_pow: 3,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 21,
        coeff: -7.115_124_735_378_851,
        x_pow: 0,
        y_pow: 1,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 22,
        coeff: 1.677_050_983_124_842,
        x_pow: 4,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 22,
        coeff: -10.062_305_898_749_049,
        x_pow: 2,
        y_pow: 2,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 22,
        coeff: 10.062_305_898_749_054,
        x_pow: 0,
        y_pow: 2,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 22,
        coeff: -1.677_050_983_124_841_7,
        x_pow: 0,
        y_pow: 0,
        z_pow: 4,
    },
    ShPolynomialTerm {
        output: 23,
        coeff: -18.824_850_597_016_706,
        x_pow: 2,
        y_pow: 1,
        z_pow: 1,
    },
    ShPolynomialTerm {
        output: 23,
        coeff: 6.274_950_199_005_566,
        x_pow: 0,
        y_pow: 1,
        z_pow: 3,
    },
    ShPolynomialTerm {
        output: 24,
        coeff: 2.218_529_918_662_357,
        x_pow: 4,
        y_pow: 0,
        z_pow: 0,
    },
    ShPolynomialTerm {
        output: 24,
        coeff: -13.311_179_511_974_13,
        x_pow: 2,
        y_pow: 0,
        z_pow: 2,
    },
    ShPolynomialTerm {
        output: 24,
        coeff: 2.218_529_918_662_355_7,
        x_pow: 0,
        y_pow: 0,
        z_pow: 4,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShNormalization {
    Component,
    Norm,
    Integral,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalHarmonicsPlan {
    pub l_max: usize,
    pub normalize_input: bool,
    pub normalization: ShNormalization,
    pub eps: f64,
}

impl SphericalHarmonicsPlan {
    pub fn new(
        l_max: usize,
        normalize_input: bool,
        normalization: ShNormalization,
    ) -> Result<Self> {
        if l_max > 4 {
            return Err(CoreError::Unsupported(format!(
                "spherical harmonics l_max={l_max}; T15 supports l_max <= 4"
            )));
        }
        if normalization != ShNormalization::Component {
            return Err(CoreError::Unsupported(
                "only component spherical-harmonics normalization is implemented".to_owned(),
            ));
        }

        Ok(Self {
            l_max,
            normalize_input,
            normalization,
            eps: DEFAULT_EPS,
        })
    }

    pub fn component(l_max: usize, normalize_input: bool) -> Result<Self> {
        Self::new(l_max, normalize_input, ShNormalization::Component)
    }

    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.checked_output_dim().unwrap_or(usize::MAX)
    }

    pub fn checked_output_dim(&self) -> Result<usize> {
        self.l_max
            .checked_add(1)
            .and_then(|value| value.checked_mul(value))
            .ok_or_else(|| {
                CoreError::Unsupported(format!(
                    "spherical harmonics l_max={} overflows output dimension",
                    self.l_max
                ))
            })
    }

    pub fn forward(&self, r: Vec3, out: &mut [f64]) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if out.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_forward",
                expected: output_dim,
                actual: out.len(),
            });
        }

        let input = self.input_vector(r);
        component_forward(self.l_max, input, out);
        Ok(())
    }

    pub fn forward_unchecked(&self, r: Vec3, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.output_dim());
        let input = self.input_vector(r);
        component_forward(self.l_max, input, out);
    }

    pub fn forward_f32(&self, r: Vec3, out: &mut [f32]) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if out.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_forward_f32",
                expected: output_dim,
                actual: out.len(),
            });
        }

        let input = self.input_vector_f32(r);
        component_forward_f32(self.l_max, input, out);
        Ok(())
    }

    pub fn forward_f32_unchecked(&self, r: Vec3, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.output_dim());
        let input = self.input_vector_f32(r);
        component_forward_f32(self.l_max, input, out);
    }

    pub fn vjp_edge_vector(&self, r: Vec3, grad_y: &[f64], grad_r: &mut Vec3) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if grad_y.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector",
                expected: output_dim,
                actual: grad_y.len(),
            });
        }

        let input = self.input_vector(r);
        let grad_input = component_vjp_input(self.l_max, input, grad_y);
        if self.normalize_input {
            accumulate_normalized_vjp(r, input, grad_input, self.eps, grad_r);
        } else {
            grad_r[0] += grad_input[0];
            grad_r[1] += grad_input[1];
            grad_r[2] += grad_input[2];
        }
        Ok(())
    }

    pub fn jvp_edge_vector(&self, r: Vec3, r_dot: Vec3, out_dot: &mut [f64]) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if out_dot.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_jvp_edge_vector",
                expected: output_dim,
                actual: out_dot.len(),
            });
        }

        let input = self.input_vector(r);
        let input_dot = if self.normalize_input {
            normalized_input_jvp(r, input, r_dot, self.eps)
        } else {
            r_dot
        };
        component_jvp_input(self.l_max, input, input_dot, out_dot);
        Ok(())
    }

    pub fn vjp_edge_vector_linearize(
        &self,
        r: Vec3,
        r_dot: Vec3,
        grad_y: &[f64],
        grad_y_dot: &[f64],
        grad_r_dot: &mut Vec3,
    ) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if grad_y.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector_linearize grad_y",
                expected: output_dim,
                actual: grad_y.len(),
            });
        }
        if grad_y_dot.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector_linearize grad_y_dot",
                expected: output_dim,
                actual: grad_y_dot.len(),
            });
        }

        let input = self.input_vector(r);
        let input_dot = if self.normalize_input {
            normalized_input_jvp(r, input, r_dot, self.eps)
        } else {
            r_dot
        };
        let grad_input = component_vjp_input(self.l_max, input, grad_y);
        let grad_input_dot =
            component_vjp_input_linearize(self.l_max, input, input_dot, grad_y, grad_y_dot);
        if self.normalize_input {
            accumulate_normalized_vjp_linearize(
                r,
                input,
                r_dot,
                input_dot,
                grad_input,
                grad_input_dot,
                self.eps,
                grad_r_dot,
            );
        } else {
            for axis in 0..3 {
                grad_r_dot[axis] += grad_input_dot[axis];
            }
        }
        Ok(())
    }

    pub fn vjp_input_gradient_edge_vector(&self, r: Vec3, grad_y: &[f64]) -> Result<Vec3> {
        let output_dim = self.checked_output_dim()?;
        if grad_y.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_input_gradient_edge_vector",
                expected: output_dim,
                actual: grad_y.len(),
            });
        }

        let input = self.input_vector(r);
        Ok(component_vjp_input(self.l_max, input, grad_y))
    }

    pub fn vjp_edge_vector_linearize_with_input_gradient(
        &self,
        r: Vec3,
        r_dot: Vec3,
        grad_input: Vec3,
        grad_y: &[f64],
        grad_y_dot: &[f64],
        grad_r_dot: &mut Vec3,
    ) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if grad_y.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector_linearize_with_input_gradient grad_y",
                expected: output_dim,
                actual: grad_y.len(),
            });
        }
        if grad_y_dot.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector_linearize_with_input_gradient grad_y_dot",
                expected: output_dim,
                actual: grad_y_dot.len(),
            });
        }

        let (input, input_dot, distance, denom) = if self.normalize_input {
            let distance = dot(r, r).sqrt();
            let denom = distance.max(self.eps);
            let input = [r[0] / denom, r[1] / denom, r[2] / denom];
            let input_dot = if distance > self.eps {
                let radial_tangent = dot(input, r_dot);
                [
                    (r_dot[0] - input[0] * radial_tangent) / distance,
                    (r_dot[1] - input[1] * radial_tangent) / distance,
                    (r_dot[2] - input[2] * radial_tangent) / distance,
                ]
            } else {
                [
                    r_dot[0] / self.eps,
                    r_dot[1] / self.eps,
                    r_dot[2] / self.eps,
                ]
            };
            (input, input_dot, distance, denom)
        } else {
            (r, r_dot, 0.0, 1.0)
        };
        let grad_input_dot =
            component_vjp_input_linearize(self.l_max, input, input_dot, grad_y, grad_y_dot);
        if self.normalize_input {
            let projection = dot(input, grad_input);
            let projection_dot = dot(input_dot, grad_input) + dot(input, grad_input_dot);
            let denom_dot = if distance > self.eps {
                dot(input, r_dot)
            } else {
                0.0
            };
            for axis in 0..3 {
                let numerator = grad_input[axis] - input[axis] * projection;
                let numerator_dot = grad_input_dot[axis]
                    - input_dot[axis] * projection
                    - input[axis] * projection_dot;
                grad_r_dot[axis] += numerator_dot / denom - numerator * denom_dot / (denom * denom);
            }
        } else {
            for axis in 0..3 {
                grad_r_dot[axis] += grad_input_dot[axis];
            }
        }
        Ok(())
    }

    pub fn vjp_edge_vector_unchecked(&self, r: Vec3, grad_y: &[f64], grad_r: &mut Vec3) {
        debug_assert_eq!(grad_y.len(), self.output_dim());
        let input = self.input_vector(r);
        let grad_input = component_vjp_input(self.l_max, input, grad_y);
        if self.normalize_input {
            accumulate_normalized_vjp(r, input, grad_input, self.eps, grad_r);
        } else {
            grad_r[0] += grad_input[0];
            grad_r[1] += grad_input[1];
            grad_r[2] += grad_input[2];
        }
    }

    pub fn vjp_edge_vector_f32(&self, r: Vec3, grad_y: &[f32], grad_r: &mut Vec3) -> Result<()> {
        let output_dim = self.checked_output_dim()?;
        if grad_y.len() != output_dim {
            return Err(CoreError::ShapeMismatch {
                op: "spherical_harmonics_vjp_edge_vector_f32",
                expected: output_dim,
                actual: grad_y.len(),
            });
        }

        let input = self.input_vector_f32(r);
        let grad_input = component_vjp_input_f32(self.l_max, input, grad_y);
        if self.normalize_input {
            accumulate_normalized_vjp_f32(r, input, grad_input, self.eps as f32, grad_r);
        } else {
            grad_r[0] += f64::from(grad_input[0]);
            grad_r[1] += f64::from(grad_input[1]);
            grad_r[2] += f64::from(grad_input[2]);
        }
        Ok(())
    }

    pub fn vjp_edge_vector_f32_unchecked(&self, r: Vec3, grad_y: &[f32], grad_r: &mut Vec3) {
        debug_assert_eq!(grad_y.len(), self.output_dim());
        let input = self.input_vector_f32(r);
        let grad_input = component_vjp_input_f32(self.l_max, input, grad_y);
        if self.normalize_input {
            accumulate_normalized_vjp_f32(r, input, grad_input, self.eps as f32, grad_r);
        } else {
            grad_r[0] += f64::from(grad_input[0]);
            grad_r[1] += f64::from(grad_input[1]);
            grad_r[2] += f64::from(grad_input[2]);
        }
    }

    fn input_vector(&self, r: Vec3) -> Vec3 {
        if self.normalize_input {
            let distance = dot(r, r).sqrt();
            let denom = distance.max(self.eps);
            [r[0] / denom, r[1] / denom, r[2] / denom]
        } else {
            r
        }
    }

    fn input_vector_f32(&self, r: Vec3) -> [f32; 3] {
        let input = [r[0] as f32, r[1] as f32, r[2] as f32];
        if self.normalize_input {
            let distance = (input[0] * input[0] + input[1] * input[1] + input[2] * input[2]).sqrt();
            let denom = distance.max(self.eps as f32);
            [input[0] / denom, input[1] / denom, input[2] / denom]
        } else {
            input
        }
    }
}

fn component_forward(l_max: usize, r: Vec3, out: &mut [f64]) {
    let dim = (l_max + 1) * (l_max + 1);
    out[..dim].fill(0.0);
    let px = pow_small_table(r[0]);
    let py = pow_small_table(r[1]);
    let pz = pow_small_table(r[2]);
    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        out[term.output] += term.coeff
            * px[term.x_pow as usize]
            * py[term.y_pow as usize]
            * pz[term.z_pow as usize];
    }
}

fn component_forward_f32(l_max: usize, r: [f32; 3], out: &mut [f32]) {
    let dim = (l_max + 1) * (l_max + 1);
    out[..dim].fill(0.0);
    let px = pow_small_table_f32(r[0]);
    let py = pow_small_table_f32(r[1]);
    let pz = pow_small_table_f32(r[2]);
    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        out[term.output] += (term.coeff as f32)
            * px[term.x_pow as usize]
            * py[term.y_pow as usize]
            * pz[term.z_pow as usize];
    }
}

fn component_jvp_input(l_max: usize, r: Vec3, r_dot: Vec3, out_dot: &mut [f64]) {
    let dim = (l_max + 1) * (l_max + 1);
    out_dot[..dim].fill(0.0);
    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        out_dot[term.output] += term.coeff * monomial_directional_derivative(term, r, r_dot);
    }
}

fn component_vjp_input(l_max: usize, r: Vec3, grad_y: &[f64]) -> Vec3 {
    let dim = (l_max + 1) * (l_max + 1);
    let mut grad = [0.0; 3];

    let px = pow_small_table(r[0]);
    let py = pow_small_table(r[1]);
    let pz = pow_small_table(r[2]);
    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        let bar = grad_y[term.output];
        if term.x_pow > 0 {
            grad[0] += bar
                * term.coeff
                * f64::from(term.x_pow)
                * px[(term.x_pow - 1) as usize]
                * py[term.y_pow as usize]
                * pz[term.z_pow as usize];
        }
        if term.y_pow > 0 {
            grad[1] += bar
                * term.coeff
                * f64::from(term.y_pow)
                * px[term.x_pow as usize]
                * py[(term.y_pow - 1) as usize]
                * pz[term.z_pow as usize];
        }
        if term.z_pow > 0 {
            grad[2] += bar
                * term.coeff
                * f64::from(term.z_pow)
                * px[term.x_pow as usize]
                * py[term.y_pow as usize]
                * pz[(term.z_pow - 1) as usize];
        }
    }

    grad
}

fn component_vjp_input_linearize(
    l_max: usize,
    r: Vec3,
    r_dot: Vec3,
    grad_y: &[f64],
    grad_y_dot: &[f64],
) -> Vec3 {
    let dim = (l_max + 1) * (l_max + 1);
    let mut grad_dot = component_vjp_input(l_max, r, grad_y_dot);

    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        let bar = grad_y[term.output];
        let powers = [term.x_pow, term.y_pow, term.z_pow];
        for grad_axis in 0..3 {
            if powers[grad_axis] == 0 {
                continue;
            }
            let mut grad_powers = powers;
            grad_powers[grad_axis] -= 1;
            let coeff = term.coeff * f64::from(powers[grad_axis]);
            grad_dot[grad_axis] +=
                bar * coeff * monomial_directional_derivative_from_powers(grad_powers, r, r_dot);
        }
    }

    grad_dot
}

fn component_vjp_input_f32(l_max: usize, r: [f32; 3], grad_y: &[f32]) -> [f32; 3] {
    let dim = (l_max + 1) * (l_max + 1);
    let mut grad = [0.0; 3];

    let px = pow_small_table_f32(r[0]);
    let py = pow_small_table_f32(r[1]);
    let pz = pow_small_table_f32(r[2]);
    for term in SH_COMPONENT_TERMS {
        if term.output >= dim {
            break;
        }
        let bar = grad_y[term.output];
        let coeff = term.coeff as f32;
        if term.x_pow > 0 {
            grad[0] += bar
                * coeff
                * f32::from(term.x_pow)
                * px[(term.x_pow - 1) as usize]
                * py[term.y_pow as usize]
                * pz[term.z_pow as usize];
        }
        if term.y_pow > 0 {
            grad[1] += bar
                * coeff
                * f32::from(term.y_pow)
                * px[term.x_pow as usize]
                * py[(term.y_pow - 1) as usize]
                * pz[term.z_pow as usize];
        }
        if term.z_pow > 0 {
            grad[2] += bar
                * coeff
                * f32::from(term.z_pow)
                * px[term.x_pow as usize]
                * py[term.y_pow as usize]
                * pz[(term.z_pow - 1) as usize];
        }
    }

    grad
}

fn pow_small(value: f64, power: u8) -> f64 {
    match power {
        0 => 1.0,
        1 => value,
        2 => value * value,
        3 => value * value * value,
        4 => {
            let square = value * value;
            square * square
        }
        _ => unreachable!("SH polynomial powers are <= 4"),
    }
}

/// Precomputed monomial powers `[value^0 ..= value^4]` for one coordinate.
///
/// Entries reproduce `pow_small`'s exact f64 grouping bit-for-bit: power 3 is
/// `(value*value)*value` and power 4 is `(value*value)*(value*value)`, sharing
/// the single `square = value * value` product. Indexing this table is therefore
/// numerically identical to calling `pow_small(value, power)` for `power <= 4`.
#[inline]
fn pow_small_table(value: f64) -> [f64; 5] {
    let square = value * value;
    [1.0, value, square, square * value, square * square]
}

fn monomial_directional_derivative(term: &ShPolynomialTerm, r: Vec3, r_dot: Vec3) -> f64 {
    monomial_directional_derivative_from_powers([term.x_pow, term.y_pow, term.z_pow], r, r_dot)
}

fn monomial_directional_derivative_from_powers(powers: [u8; 3], r: Vec3, r_dot: Vec3) -> f64 {
    let mut directional = 0.0;
    for axis in 0..3 {
        if powers[axis] == 0 {
            continue;
        }
        let mut term = f64::from(powers[axis]) * r_dot[axis];
        for component in 0..3 {
            let power = if component == axis {
                powers[component] - 1
            } else {
                powers[component]
            };
            term *= pow_small(r[component], power);
        }
        directional += term;
    }
    directional
}

#[allow(dead_code)]
fn pow_small_f32(value: f32, power: u8) -> f32 {
    match power {
        0 => 1.0,
        1 => value,
        2 => value * value,
        3 => value * value * value,
        4 => {
            let square = value * value;
            square * square
        }
        _ => unreachable!("SH polynomial powers are <= 4"),
    }
}

/// Precomputed monomial powers `[value^0 ..= value^4]` for one coordinate (f32).
///
/// Entries reproduce `pow_small_f32`'s exact f32 grouping bit-for-bit, sharing
/// the single `square = value * value` product. Indexing this table is therefore
/// numerically identical to calling `pow_small_f32(value, power)` for `power <= 4`.
#[inline]
fn pow_small_table_f32(value: f32) -> [f32; 5] {
    let square = value * value;
    [1.0, value, square, square * value, square * square]
}

fn accumulate_normalized_vjp(r: Vec3, unit: Vec3, grad_unit: Vec3, eps: f64, grad_r: &mut Vec3) {
    let denom = dot(r, r).sqrt().max(eps);
    let projection = dot(unit, grad_unit);
    grad_r[0] += (grad_unit[0] - unit[0] * projection) / denom;
    grad_r[1] += (grad_unit[1] - unit[1] * projection) / denom;
    grad_r[2] += (grad_unit[2] - unit[2] * projection) / denom;
}

fn normalized_input_jvp(r: Vec3, unit: Vec3, r_dot: Vec3, eps: f64) -> Vec3 {
    let distance = dot(r, r).sqrt();
    if distance > eps {
        let radial_tangent = dot(unit, r_dot);
        [
            (r_dot[0] - unit[0] * radial_tangent) / distance,
            (r_dot[1] - unit[1] * radial_tangent) / distance,
            (r_dot[2] - unit[2] * radial_tangent) / distance,
        ]
    } else {
        [r_dot[0] / eps, r_dot[1] / eps, r_dot[2] / eps]
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_normalized_vjp_linearize(
    r: Vec3,
    unit: Vec3,
    r_dot: Vec3,
    unit_dot: Vec3,
    grad_unit: Vec3,
    grad_unit_dot: Vec3,
    eps: f64,
    grad_r_dot: &mut Vec3,
) {
    let distance = dot(r, r).sqrt();
    let denom = distance.max(eps);
    let projection = dot(unit, grad_unit);
    let projection_dot = dot(unit_dot, grad_unit) + dot(unit, grad_unit_dot);
    let denom_dot = if distance > eps {
        dot(unit, r_dot)
    } else {
        0.0
    };
    for axis in 0..3 {
        let numerator = grad_unit[axis] - unit[axis] * projection;
        let numerator_dot =
            grad_unit_dot[axis] - unit_dot[axis] * projection - unit[axis] * projection_dot;
        grad_r_dot[axis] += numerator_dot / denom - numerator * denom_dot / (denom * denom);
    }
}

fn accumulate_normalized_vjp_f32(
    r: Vec3,
    unit: [f32; 3],
    grad_unit: [f32; 3],
    eps: f32,
    grad_r: &mut Vec3,
) {
    let r = [r[0] as f32, r[1] as f32, r[2] as f32];
    let denom = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(eps);
    let projection = unit[0] * grad_unit[0] + unit[1] * grad_unit[1] + unit[2] * grad_unit[2];
    grad_r[0] += f64::from((grad_unit[0] - unit[0] * projection) / denom);
    grad_r[1] += f64::from((grad_unit[1] - unit[1] * projection) / denom);
    grad_r[2] += f64::from((grad_unit[2] - unit[2] * projection) / denom);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-12;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected:.16e}, got {actual:.16e}, diff {:.3e}",
            (actual - expected).abs()
        );
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual_value - *expected_value).abs() <= tol,
                "index {index}: expected {expected_value:.16e}, got {actual_value:.16e}, diff {:.3e}",
                (*actual_value - *expected_value).abs()
            );
        }
    }

    fn central_difference<F>(x: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = 1.0e-6;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn scalar_projection(plan: &SphericalHarmonicsPlan, r: Vec3, grad_y: &[f64]) -> f64 {
        let mut y = vec![0.0; plan.output_dim()];
        plan.forward(r, &mut y).unwrap();
        y.iter()
            .zip(grad_y)
            .map(|(value, grad_value)| *value * grad_value)
            .sum()
    }

    fn perturb_vec3(r: Vec3, r_dot: Vec3, h: f64) -> Vec3 {
        [
            r[0] + h * r_dot[0],
            r[1] + h * r_dot[1],
            r[2] + h * r_dot[2],
        ]
    }

    fn perturb_slice(values: &[f64], tangents: &[f64], h: f64) -> Vec<f64> {
        values
            .iter()
            .zip(tangents)
            .map(|(value, tangent)| value + h * tangent)
            .collect()
    }

    #[test]
    fn output_dim_is_lmax_plus_one_squared() {
        assert_eq!(
            SphericalHarmonicsPlan::component(0, true)
                .unwrap()
                .output_dim(),
            1
        );
        assert_eq!(
            SphericalHarmonicsPlan::component(1, true)
                .unwrap()
                .output_dim(),
            4
        );
        assert_eq!(
            SphericalHarmonicsPlan::component(2, true)
                .unwrap()
                .output_dim(),
            9
        );
        assert_eq!(
            SphericalHarmonicsPlan::component(3, true)
                .unwrap()
                .output_dim(),
            16
        );
        assert_eq!(
            SphericalHarmonicsPlan::component(4, true)
                .unwrap()
                .output_dim(),
            25
        );
    }

    #[test]
    fn rejects_unsupported_lmax_and_normalization() {
        assert!(SphericalHarmonicsPlan::component(5, true).is_err());
        assert!(SphericalHarmonicsPlan::new(4, true, ShNormalization::Integral).is_err());
        assert!(SphericalHarmonicsPlan::new(4, true, ShNormalization::Norm).is_err());
    }

    #[test]
    fn public_sh_plan_fields_do_not_panic_on_overflow() {
        let plan = SphericalHarmonicsPlan {
            l_max: usize::MAX,
            normalize_input: true,
            normalization: ShNormalization::Component,
            eps: DEFAULT_EPS,
        };

        assert_eq!(plan.output_dim(), usize::MAX);
        assert!(matches!(
            plan.checked_output_dim(),
            Err(CoreError::Unsupported(_))
        ));
        assert!(matches!(
            plan.forward([1.0, 0.0, 0.0], &mut []),
            Err(CoreError::Unsupported(_))
        ));
    }

    #[test]
    fn normalized_forward_matches_fixture_simple_axes() {
        let plan = SphericalHarmonicsPlan::component(2, true).unwrap();
        let mut out = [0.0; 9];

        plan.forward([1.0, 0.0, 0.0], &mut out).unwrap();
        assert_vec_close(
            &out,
            &[
                1.0,
                SQRT_3,
                0.0,
                0.0,
                0.0,
                0.0,
                -HALF_SQRT_5,
                0.0,
                -HALF_SQRT_15,
            ],
            TOL,
        );

        plan.forward([0.0, 1.0, 0.0], &mut out).unwrap();
        assert_vec_close(
            &out,
            &[1.0, 0.0, SQRT_3, 0.0, 0.0, 0.0, SQRT_5, 0.0, 0.0],
            TOL,
        );

        plan.forward([0.0, 0.0, 1.0], &mut out).unwrap();
        assert_vec_close(
            &out,
            &[
                1.0,
                0.0,
                0.0,
                SQRT_3,
                0.0,
                0.0,
                -HALF_SQRT_5,
                0.0,
                HALF_SQRT_15,
            ],
            TOL,
        );
    }

    #[test]
    fn normalized_forward_matches_fixture_mixed_vector() {
        let plan = SphericalHarmonicsPlan::component(2, true).unwrap();
        let mut out = [0.0; 9];

        plan.forward([1.0, 2.0, 3.0], &mut out).unwrap();

        assert_vec_close(
            &out,
            &[
                1.0,
                0.462_910_049_886_275_7,
                0.925_820_099_772_551_4,
                1.388_730_149_658_827_1,
                0.829_925_002_758_732_3,
                0.553_283_335_172_488_2,
                -0.159_719_141_249_984_9,
                1.659_850_005_517_464_5,
                1.106_566_670_344_976_4,
            ],
            TOL,
        );
    }

    #[test]
    fn unnormalized_forward_uses_homogeneous_polynomials() {
        let plan = SphericalHarmonicsPlan::component(2, false).unwrap();
        let mut out = [0.0; 9];

        plan.forward([2.0, 0.0, 0.0], &mut out).unwrap();

        assert_vec_close(
            &out,
            &[
                1.0,
                2.0 * SQRT_3,
                0.0,
                0.0,
                0.0,
                0.0,
                -4.0 * HALF_SQRT_5,
                0.0,
                -4.0 * HALF_SQRT_15,
            ],
            TOL,
        );
    }

    #[test]
    fn normalized_forward_is_scale_invariant() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let mut out = vec![0.0; plan.output_dim()];
        let mut scaled_out = vec![0.0; plan.output_dim()];

        plan.forward([0.7, -1.1, 2.3], &mut out).unwrap();
        plan.forward([7.0, -11.0, 23.0], &mut scaled_out).unwrap();

        assert_vec_close(&out, &scaled_out, TOL);
    }

    #[test]
    fn normalized_forward_zero_vector_uses_eps_policy() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let mut out = vec![f64::NAN; plan.output_dim()];

        plan.forward([0.0, 0.0, 0.0], &mut out).unwrap();

        let mut expected = vec![0.0; plan.output_dim()];
        expected[0] = 1.0;
        assert_vec_close(&out, &expected, TOL);
    }

    #[test]
    fn forward_reports_shape_mismatch() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let err = plan
            .forward([1.0, 0.0, 0.0], &mut [0.0; 24])
            .expect_err("short output should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn vjp_reports_shape_mismatch() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let mut grad_r = [0.0; 3];
        let err = plan
            .vjp_edge_vector([1.0, 0.0, 0.0], &[0.0; 24], &mut grad_r)
            .expect_err("short cotangent should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn jvp_reports_shape_mismatch() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let err = plan
            .jvp_edge_vector([1.0, 0.0, 0.0], [0.1, 0.0, 0.0], &mut [0.0; 24])
            .expect_err("short tangent output should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn linearized_vjp_reports_shape_mismatch() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let mut grad_r_dot = [0.0; 3];
        let err = plan
            .vjp_edge_vector_linearize(
                [1.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                &[0.0; 24],
                &[0.0; 25],
                &mut grad_r_dot,
            )
            .expect_err("short cotangent should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));

        let err = plan
            .vjp_edge_vector_linearize(
                [1.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                &[0.0; 25],
                &[0.0; 24],
                &mut grad_r_dot,
            )
            .expect_err("short cotangent tangent should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn jvp_matches_finite_difference() {
        let r = [0.7, -1.1, 2.3];
        let r_dot = [-0.31, 0.17, 0.23];
        let h = 1.0e-6;
        for normalize_input in [true, false] {
            for l_max in 0..=4 {
                let plan = SphericalHarmonicsPlan::component(l_max, normalize_input).unwrap();
                let mut actual = vec![0.0; plan.output_dim()];
                plan.jvp_edge_vector(r, r_dot, &mut actual).unwrap();

                let mut plus = vec![0.0; plan.output_dim()];
                let mut minus = vec![0.0; plan.output_dim()];
                plan.forward(perturb_vec3(r, r_dot, h), &mut plus).unwrap();
                plan.forward(perturb_vec3(r, r_dot, -h), &mut minus)
                    .unwrap();
                for index in 0..plan.output_dim() {
                    let expected = (plus[index] - minus[index]) / (2.0 * h);
                    assert_close(actual[index], expected, 2.0e-8);
                }
            }
        }
    }

    #[test]
    fn normalized_vjp_matches_finite_difference() {
        let r = [0.7, -1.1, 2.3];
        for l_max in 0..=4 {
            let plan = SphericalHarmonicsPlan::component(l_max, true).unwrap();
            let grad_y = (0..plan.output_dim())
                .map(|index| 0.3 - 0.17 * index as f64)
                .collect::<Vec<_>>();
            let mut grad_r = [0.25, -0.5, 0.75];
            let initial = grad_r;

            plan.vjp_edge_vector(r, &grad_y, &mut grad_r).unwrap();

            for axis in 0..3 {
                let fd = central_difference(r[axis], |value| {
                    let mut perturbed = r;
                    perturbed[axis] = value;
                    scalar_projection(&plan, perturbed, &grad_y)
                });
                assert_close(grad_r[axis] - initial[axis], fd, 1.0e-7);
            }
        }
    }

    #[test]
    fn unnormalized_vjp_matches_finite_difference() {
        let r = [0.7, -1.1, 2.3];
        for l_max in 0..=4 {
            let plan = SphericalHarmonicsPlan::component(l_max, false).unwrap();
            let grad_y = (0..plan.output_dim())
                .map(|index| -0.2 + 0.11 * index as f64)
                .collect::<Vec<_>>();
            let mut grad_r = [0.0; 3];

            plan.vjp_edge_vector(r, &grad_y, &mut grad_r).unwrap();

            for axis in 0..3 {
                let fd = central_difference(r[axis], |value| {
                    let mut perturbed = r;
                    perturbed[axis] = value;
                    scalar_projection(&plan, perturbed, &grad_y)
                });
                assert_close(grad_r[axis], fd, 1.0e-7);
            }
        }
    }

    #[test]
    fn lmax_zero_vjp_has_no_directional_gradient() {
        let plan = SphericalHarmonicsPlan::component(0, true).unwrap();
        let mut grad_r = [0.25, -0.5, 0.75];

        plan.vjp_edge_vector([0.7, -1.1, 2.3], &[3.0], &mut grad_r)
            .unwrap();

        assert_vec_close(&grad_r, &[0.25, -0.5, 0.75], TOL);
    }

    #[test]
    fn normalized_vjp_uses_documented_near_zero_policy() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let grad_y = (0..plan.output_dim())
            .map(|index| 0.5 - 0.03 * index as f64)
            .collect::<Vec<_>>();
        let mut grad_r = [0.0; 3];

        plan.vjp_edge_vector([1.0e-14, -2.0e-14, 0.0], &grad_y, &mut grad_r)
            .unwrap();

        for value in grad_r {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn linearized_vjp_matches_finite_difference() {
        let r = [0.7, -1.1, 2.3];
        let r_dot = [-0.31, 0.17, 0.23];
        let h = 1.0e-6;
        for normalize_input in [true, false] {
            for l_max in 0..=4 {
                let plan = SphericalHarmonicsPlan::component(l_max, normalize_input).unwrap();
                let grad_y = (0..plan.output_dim())
                    .map(|index| 0.2 - 0.13 * index as f64)
                    .collect::<Vec<_>>();
                let grad_y_dot = (0..plan.output_dim())
                    .map(|index| -0.4 + 0.07 * index as f64)
                    .collect::<Vec<_>>();
                let mut actual = [0.125, -0.25, 0.5];
                let initial = actual;
                plan.vjp_edge_vector_linearize(r, r_dot, &grad_y, &grad_y_dot, &mut actual)
                    .unwrap();

                let mut plus = [0.0; 3];
                plan.vjp_edge_vector(
                    perturb_vec3(r, r_dot, h),
                    &perturb_slice(&grad_y, &grad_y_dot, h),
                    &mut plus,
                )
                .unwrap();
                let mut minus = [0.0; 3];
                plan.vjp_edge_vector(
                    perturb_vec3(r, r_dot, -h),
                    &perturb_slice(&grad_y, &grad_y_dot, -h),
                    &mut minus,
                )
                .unwrap();

                for axis in 0..3 {
                    let expected = (plus[axis] - minus[axis]) / (2.0 * h);
                    assert_close(actual[axis] - initial[axis], expected, 2.0e-7);
                }
            }
        }
    }

    #[test]
    fn normalized_jvp_and_linearized_vjp_use_near_zero_policy() {
        let plan = SphericalHarmonicsPlan::component(4, true).unwrap();
        let r = [1.0e-14, -2.0e-14, 0.0];
        let r_dot = [0.1, -0.2, 0.3];
        let grad_y = (0..plan.output_dim())
            .map(|index| 0.5 - 0.03 * index as f64)
            .collect::<Vec<_>>();
        let grad_y_dot = (0..plan.output_dim())
            .map(|index| -0.25 + 0.02 * index as f64)
            .collect::<Vec<_>>();
        let mut out_dot = vec![f64::NAN; plan.output_dim()];
        let mut grad_r_dot = [0.0; 3];

        plan.jvp_edge_vector(r, r_dot, &mut out_dot).unwrap();
        plan.vjp_edge_vector_linearize(r, r_dot, &grad_y, &grad_y_dot, &mut grad_r_dot)
            .unwrap();

        for value in out_dot.into_iter().chain(grad_r_dot) {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn matches_pinned_e3nn_reference_fixture_forward_and_vjp() {
        let plan = SphericalHarmonicsPlan::component(2, true).unwrap();
        let vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-0.25, 0.5, 1.25],
        ];
        let expected_y = [
            [
                1.0,
                1.732_050_807_568_877_2,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.118_033_988_749_895,
                0.0,
                -1.936_491_673_103_708_5,
            ],
            [
                1.0,
                0.0,
                1.732_050_807_568_877_2,
                0.0,
                0.0,
                0.0,
                2.236_067_977_499_79,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                1.732_050_807_568_877_2,
                0.0,
                0.0,
                -1.118_033_988_749_895,
                0.0,
                1.936_491_673_103_708_5,
            ],
            [
                1.0,
                0.462_910_049_886_275_7,
                0.925_820_099_772_551_4,
                1.388_730_149_658_827_1,
                0.829_925_002_758_732_3,
                0.553_283_335_172_488_2,
                -0.159_719_141_249_984_9,
                1.659_850_005_517_464_5,
                1.106_566_670_344_976_4,
            ],
            [
                1.0,
                -0.316_227_766_016_837_9,
                0.632_455_532_033_675_8,
                1.581_138_830_084_189_8,
                -0.645_497_224_367_902_8,
                -0.258_198_889_747_161_1,
                -0.670_820_393_249_937,
                1.290_994_448_735_805_6,
                1.549_193_338_482_966_8,
            ],
        ];
        let grad_y = [
            [
                0.132_863_086_518_157_8,
                -2.346_708_427_076_086_5,
                -0.629_684_518_115_028_8,
                0.092_224_689_360_762_6,
                0.668_226_939_091_551_2,
                1.500_310_164_098_042_5,
                -0.026_791_140_938_767_4,
                -0.755_389_823_251_759_7,
                -1.956_314_574_102_543_4,
            ],
            [
                -0.163_995_153_437_277_6,
                1.339_811_819_636_179_7,
                -2.366_463_809_791_415_7,
                1.402_540_674_087_945,
                -0.018_046_557_564_722_7,
                -0.887_446_587_682_219_4,
                1.372_028_158_686_479,
                -0.912_154_105_324_030_4,
                -0.573_025_508_806_298_3,
            ],
            [
                -0.068_906_442_780_713_4,
                0.037_865_913_958_100_9,
                -1.398_188_584_705_106_3,
                0.900_915_359_783_638_6,
                1.430_260_501_275_640_4,
                0.973_992_458_594_263_3,
                0.733_996_060_031_544_7,
                0.398_791_586_665_318_4,
                -1.348_799_720_114_300_3,
            ],
            [
                0.475_632_098_278_814_8,
                -1.077_649_493_079_394_5,
                1.544_237_666_119_022_5,
                0.066_308_942_964_494_6,
                -1.585_394_068_305_914_7,
                0.013_700_365_978_926_4,
                0.470_682_155_162_159_3,
                0.943_607_880_525_705_3,
                1.111_759_365_211_411,
            ],
            [
                0.885_314_629_541_607_4,
                -1.224_787_760_148_560_4,
                1.509_339_624_982_938_5,
                -0.817_906_440_391_703_3,
                -0.397_451_900_940_021_6,
                -1.375_773_622_932_331_2,
                -0.403_426_343_828_501_4,
                -0.547_818_736_709_611,
                0.854_455_003_847_582_7,
            ],
        ];
        let expected_grad_vectors = [
            [0.0, 4.720_030_701_582_681, 2.747_769_654_273_833_7],
            [-1.116_443_710_550_664_4, 0.0, -1.103_485_951_892_473_3],
            [5.604_960_789_029_29, -0.877_220_493_509_654_8, 0.0],
            [
                -2.464_709_661_821_501_6,
                1.252_582_849_458_124,
                -0.013_485_345_698_248_8,
            ],
            [
                -3.269_202_510_431_332_3,
                -0.088_337_434_526_623_2,
                -0.618_505_528_275_616_9,
            ],
        ];

        for fixture_index in 0..vectors.len() {
            let mut out = [0.0; 9];
            plan.forward(vectors[fixture_index], &mut out).unwrap();
            assert_vec_close(&out, &expected_y[fixture_index], 1.0e-12);

            let mut grad_r = [0.0; 3];
            plan.vjp_edge_vector(vectors[fixture_index], &grad_y[fixture_index], &mut grad_r)
                .unwrap();
            assert_vec_close(&grad_r, &expected_grad_vectors[fixture_index], 1.0e-12);
        }
    }
}
