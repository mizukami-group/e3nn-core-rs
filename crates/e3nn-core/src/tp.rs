//! Sparse tensor product paths for the v0.1 CPU/f64 force path.
//!
//! This module consumes explicit CG entries. It does not generate Wigner 3j
//! coefficients or implement a broad e3nn tensor-product API.

use crate::{CoreError, Irreps, LayoutSegment, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgEntry {
    pub i_m: usize,
    pub j_m: usize,
    pub k_m: usize,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorProductInstruction {
    pub in1_segment: usize,
    pub in2_segment: usize,
    pub out_segment: usize,
    pub cg_offset: usize,
    pub cg_len: usize,
    pub weight_offset: Option<usize>,
    pub path_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorProductPlan {
    pub irreps_in1: Irreps,
    pub irreps_in2: Irreps,
    pub irreps_out: Irreps,
    pub instructions: Vec<TensorProductInstruction>,
    pub cg_entries: Vec<CgEntry>,
    pub weight_numel: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorProductLayouts {
    pub in1: Vec<LayoutSegment>,
    pub in2: Vec<LayoutSegment>,
    pub out: Vec<LayoutSegment>,
}

impl TensorProductPlan {
    pub fn new(
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: Vec<TensorProductInstruction>,
        cg_entries: Vec<CgEntry>,
        weight_numel: usize,
    ) -> Result<Self> {
        let plan = Self {
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            cg_entries,
            weight_numel,
        };
        plan.validate()?;
        Ok(plan)
    }

    #[must_use]
    pub fn input1_dim(&self) -> usize {
        self.irreps_in1.dim()
    }

    #[must_use]
    pub fn input2_dim(&self) -> usize {
        self.irreps_in2.dim()
    }

    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.irreps_out.dim()
    }

    pub fn forward(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        self.check_forward_shapes(x, y, w_edge, out)?;
        out.fill(0.0);
        self.forward_accumulate_checked(x, y, w_edge, out)?;
        Ok(())
    }

    pub fn forward_accumulate(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        self.check_forward_shapes(x, y, w_edge, out)?;
        self.forward_accumulate_checked(x, y, w_edge, out)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vjp(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        grad_out: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
        grad_w_edge: Option<&mut [f64]>,
    ) -> Result<()> {
        self.validate()?;
        let layouts = self.checked_layouts()?;
        self.vjp_with_layouts(
            &layouts,
            x,
            y,
            w_edge,
            grad_out,
            grad_x,
            grad_y,
            grad_w_edge,
        )
    }

    pub fn checked_layouts(&self) -> Result<TensorProductLayouts> {
        Ok(TensorProductLayouts {
            in1: self.irreps_in1.checked_layout()?,
            in2: self.irreps_in2.checked_layout()?,
            out: self.irreps_out.checked_layout()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vjp_with_layouts(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        grad_out: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
        grad_w_edge: Option<&mut [f64]>,
    ) -> Result<()> {
        let mut comp_x = vec![0.0; grad_x.len()];
        let mut comp_y = vec![0.0; grad_y.len()];
        self.vjp_with_layouts_and_compensation(
            layouts,
            x,
            y,
            w_edge,
            grad_out,
            grad_x,
            grad_y,
            grad_w_edge,
            &mut comp_x,
            &mut comp_y,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vjp_with_layouts_and_compensation(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        grad_out: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
        grad_w_edge: Option<&mut [f64]>,
        comp_x: &mut [f64],
        comp_y: &mut [f64],
    ) -> Result<()> {
        self.check_vjp_shapes(x, y, w_edge, grad_out, grad_x, grad_y, &grad_w_edge)?;
        check_len("tensor_product_vjp comp_x", self.input1_dim(), comp_x.len())?;
        check_len("tensor_product_vjp comp_y", self.input2_dim(), comp_y.len())?;
        comp_x.fill(0.0);
        comp_y.fill(0.0);
        self.vjp_layouts_checked_with_compensation(
            layouts,
            x,
            y,
            w_edge,
            grad_out,
            grad_x,
            grad_y,
            grad_w_edge,
            comp_x,
            comp_y,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn vjp_layouts_checked_with_compensation(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        grad_out: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
        mut grad_w_edge: Option<&mut [f64]>,
        comp_x: &mut [f64],
        comp_y: &mut [f64],
    ) {
        for instruction in &self.instructions {
            let in1 = layouts.in1[instruction.in1_segment];
            let in2 = layouts.in2[instruction.in2_segment];
            let out = layouts.out[instruction.out_segment];
            let weight = instruction_weight(instruction, w_edge);

            for cg in self.cg_slice(instruction) {
                let x_index = in1.start + cg.i_m;
                let y_index = in2.start + cg.j_m;
                let out_index = out.start + cg.k_m;
                let scale = instruction.path_weight * cg.value;
                let bar_out = grad_out[out_index];

                kahan_add(
                    &mut grad_x[x_index],
                    &mut comp_x[x_index],
                    scale * weight * y[y_index] * bar_out,
                );
                kahan_add(
                    &mut grad_y[y_index],
                    &mut comp_y[y_index],
                    scale * weight * x[x_index] * bar_out,
                );
                if let (Some(weight_offset), Some(grad_w)) =
                    (instruction.weight_offset, grad_w_edge.as_deref_mut())
                {
                    grad_w[weight_offset] += scale * x[x_index] * y[y_index] * bar_out;
                }
            }
        }
    }

    fn validate(&self) -> Result<()> {
        if self.instructions.is_empty() {
            return Err(invalid_tp(
                "tensor product must contain at least one instruction",
            ));
        }

        let in1_layout = self.irreps_in1.checked_layout()?;
        let in2_layout = self.irreps_in2.checked_layout()?;
        let out_layout = self.irreps_out.checked_layout()?;

        let mut dynamic_weight_seen = vec![false; self.weight_numel];
        let mut dynamic_weight_count = 0usize;
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            if !instruction.path_weight.is_finite() {
                return Err(invalid_tp(format!(
                    "instruction {instruction_index} has non-finite path_weight"
                )));
            }
            if instruction.cg_len == 0 {
                return Err(invalid_tp(format!(
                    "instruction {instruction_index} has empty CG slice"
                )));
            }
            let in1 = checked_segment(
                &in1_layout,
                instruction.in1_segment,
                instruction_index,
                "in1",
            )?;
            let in2 = checked_segment(
                &in2_layout,
                instruction.in2_segment,
                instruction_index,
                "in2",
            )?;
            let out = checked_segment(
                &out_layout,
                instruction.out_segment,
                instruction_index,
                "out",
            )?;
            validate_coupling(instruction_index, in1, in2, out)?;

            let cg_end = instruction
                .cg_offset
                .checked_add(instruction.cg_len)
                .ok_or_else(|| {
                    invalid_tp(format!(
                        "instruction {instruction_index} CG slice overflows"
                    ))
                })?;
            if cg_end > self.cg_entries.len() {
                return Err(invalid_tp(format!(
                    "instruction {instruction_index} CG slice ends at {cg_end}, but only {} entries exist",
                    self.cg_entries.len()
                )));
            }
            for (relative_index, cg) in self.cg_entries[instruction.cg_offset..cg_end]
                .iter()
                .enumerate()
            {
                if cg.i_m >= in1.len || cg.j_m >= in2.len || cg.k_m >= out.len {
                    return Err(invalid_tp(format!(
                        "instruction {instruction_index} CG entry {relative_index} index out of bounds"
                    )));
                }
                if !cg.value.is_finite() {
                    return Err(invalid_tp(format!(
                        "instruction {instruction_index} CG entry {relative_index} has non-finite value"
                    )));
                }
            }
            if let Some(weight_offset) = instruction.weight_offset {
                if weight_offset >= self.weight_numel {
                    return Err(invalid_tp(format!(
                        "instruction {instruction_index} uses weight offset {weight_offset}, but weight_numel is {}",
                        self.weight_numel
                    )));
                }
                if dynamic_weight_seen[weight_offset] {
                    return Err(invalid_tp(format!(
                        "dynamic weight offset {weight_offset} is used more than once"
                    )));
                }
                dynamic_weight_seen[weight_offset] = true;
                dynamic_weight_count += 1;
            }
        }

        if dynamic_weight_count == 0 {
            if self.weight_numel != 0 {
                return Err(invalid_tp(format!(
                    "static tensor product must have weight_numel 0, got {}",
                    self.weight_numel
                )));
            }
        } else {
            for (offset, seen) in dynamic_weight_seen.iter().enumerate() {
                if !seen {
                    return Err(invalid_tp(format!(
                        "dynamic weight offset {offset} is declared but unused"
                    )));
                }
            }
        }

        Ok(())
    }

    fn check_forward_shapes(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out: &[f64],
    ) -> Result<()> {
        check_len("tensor_product_forward x", self.input1_dim(), x.len())?;
        check_len("tensor_product_forward y", self.input2_dim(), y.len())?;
        check_len("tensor_product_forward out", self.output_dim(), out.len())?;
        self.check_weight_shape("tensor_product_forward w_edge", w_edge)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_vjp_shapes(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        grad_out: &[f64],
        grad_x: &[f64],
        grad_y: &[f64],
        grad_w_edge: &Option<&mut [f64]>,
    ) -> Result<()> {
        check_len("tensor_product_vjp x", self.input1_dim(), x.len())?;
        check_len("tensor_product_vjp y", self.input2_dim(), y.len())?;
        check_len(
            "tensor_product_vjp grad_out",
            self.output_dim(),
            grad_out.len(),
        )?;
        check_len("tensor_product_vjp grad_x", self.input1_dim(), grad_x.len())?;
        check_len("tensor_product_vjp grad_y", self.input2_dim(), grad_y.len())?;
        self.check_weight_shape("tensor_product_vjp w_edge", w_edge)?;
        if self.uses_dynamic_weights() {
            match grad_w_edge {
                Some(grad_w) => check_len(
                    "tensor_product_vjp grad_w_edge",
                    self.weight_numel,
                    grad_w.len(),
                )?,
                None => {
                    return Err(CoreError::ShapeMismatch {
                        op: "tensor_product_vjp grad_w_edge",
                        expected: self.weight_numel,
                        actual: 0,
                    });
                }
            }
        } else if let Some(grad_w) = grad_w_edge {
            check_len(
                "tensor_product_vjp grad_w_edge",
                self.weight_numel,
                grad_w.len(),
            )?;
        }
        Ok(())
    }

    fn check_weight_shape(&self, op: &'static str, w_edge: Option<&[f64]>) -> Result<()> {
        if self.uses_dynamic_weights() {
            match w_edge {
                Some(weights) => check_len(op, self.weight_numel, weights.len())?,
                None => {
                    return Err(CoreError::ShapeMismatch {
                        op,
                        expected: self.weight_numel,
                        actual: 0,
                    });
                }
            }
        } else if let Some(weights) = w_edge {
            check_len(op, self.weight_numel, weights.len())?;
        }
        Ok(())
    }

    fn uses_dynamic_weights(&self) -> bool {
        self.instructions
            .iter()
            .any(|instruction| instruction.weight_offset.is_some())
    }

    fn forward_accumulate_checked(
        &self,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
    ) -> Result<()> {
        let layouts = self.checked_layouts()?;
        self.forward_accumulate_layouts_checked(&layouts, x, y, w_edge, out_values);
        Ok(())
    }

    pub fn forward_accumulate_with_layouts(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
    ) -> Result<()> {
        self.check_forward_shapes(x, y, w_edge, out_values)?;
        self.forward_accumulate_layouts_checked(layouts, x, y, w_edge, out_values);
        Ok(())
    }

    pub fn forward_accumulate_with_layouts_and_compensation(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
        compensation: &mut [f64],
    ) -> Result<()> {
        self.check_forward_shapes(x, y, w_edge, out_values)?;
        check_len(
            "tensor_product_forward compensation",
            out_values.len(),
            compensation.len(),
        )?;
        compensation.fill(0.0);
        self.forward_accumulate_layouts_checked_with_compensation(
            layouts,
            x,
            y,
            w_edge,
            out_values,
            compensation,
        );
        Ok(())
    }

    pub fn forward_accumulate_with_layouts_and_existing_compensation(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
        compensation: &mut [f64],
    ) -> Result<()> {
        self.check_forward_shapes(x, y, w_edge, out_values)?;
        check_len(
            "tensor_product_forward compensation",
            out_values.len(),
            compensation.len(),
        )?;
        self.forward_accumulate_layouts_checked_with_compensation(
            layouts,
            x,
            y,
            w_edge,
            out_values,
            compensation,
        );
        Ok(())
    }

    fn forward_accumulate_layouts_checked(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
    ) {
        let mut compensation = vec![0.0; out_values.len()];
        self.forward_accumulate_layouts_checked_with_compensation(
            layouts,
            x,
            y,
            w_edge,
            out_values,
            &mut compensation,
        );
    }

    fn forward_accumulate_layouts_checked_with_compensation(
        &self,
        layouts: &TensorProductLayouts,
        x: &[f64],
        y: &[f64],
        w_edge: Option<&[f64]>,
        out_values: &mut [f64],
        compensation: &mut [f64],
    ) {
        for instruction in &self.instructions {
            let in1 = layouts.in1[instruction.in1_segment];
            let in2 = layouts.in2[instruction.in2_segment];
            let out = layouts.out[instruction.out_segment];
            let weight = instruction_weight(instruction, w_edge);
            let scale = instruction.path_weight * weight;

            for cg in self.cg_slice(instruction) {
                let idx = out.start + cg.k_m;
                let term = scale * cg.value * x[in1.start + cg.i_m] * y[in2.start + cg.j_m];
                kahan_add(&mut out_values[idx], &mut compensation[idx], term);
            }
        }
    }

    fn cg_slice(&self, instruction: &TensorProductInstruction) -> &[CgEntry] {
        let end = instruction.cg_offset + instruction.cg_len;
        &self.cg_entries[instruction.cg_offset..end]
    }
}

fn checked_segment<'a>(
    layout: &'a [LayoutSegment],
    segment_index: usize,
    instruction_index: usize,
    name: &str,
) -> Result<&'a LayoutSegment> {
    layout.get(segment_index).ok_or_else(|| {
        invalid_tp(format!(
            "instruction {instruction_index} references missing {name} segment {segment_index}"
        ))
    })
}

fn validate_coupling(
    instruction_index: usize,
    in1: &LayoutSegment,
    in2: &LayoutSegment,
    out: &LayoutSegment,
) -> Result<()> {
    if out.irrep.parity != in1.irrep.parity.multiply(in2.irrep.parity) {
        return Err(invalid_tp(format!(
            "instruction {instruction_index} violates parity coupling: {} x {} -> {}",
            in1.irrep, in2.irrep, out.irrep
        )));
    }
    let min_l = in1.irrep.l.abs_diff(in2.irrep.l);
    let max_l = in1.irrep.l + in2.irrep.l;
    if out.irrep.l < min_l || out.irrep.l > max_l {
        return Err(invalid_tp(format!(
            "instruction {instruction_index} violates triangular coupling: {} x {} -> {}",
            in1.irrep, in2.irrep, out.irrep
        )));
    }
    Ok(())
}

#[allow(clippy::expect_used)]
fn instruction_weight(instruction: &TensorProductInstruction, w_edge: Option<&[f64]>) -> f64 {
    instruction
        .weight_offset
        .and_then(|offset| w_edge.map(|weights| weights[offset]))
        .unwrap_or(1.0)
}

fn check_len(op: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(CoreError::ShapeMismatch {
            op,
            expected,
            actual,
        })
    }
}

fn invalid_tp(message: impl Into<String>) -> CoreError {
    CoreError::InvalidTensorProduct(message.into())
}

#[inline(always)]
fn kahan_add(sum: &mut f64, compensation: &mut f64, term: f64) {
    let y = term - *compensation;
    let t = *sum + y;
    *compensation = (t - *sum) - y;
    *sum = t;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Irrep, IrrepMul, Parity};

    const SQRT_3: f64 = 1.732_050_807_568_877_2;
    const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;

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

    fn scalar_irreps(mul: usize) -> Irreps {
        Irreps::new(vec![IrrepMul::new(
            mul,
            Irrep::new(0, Parity::Even).unwrap(),
        )
        .unwrap()])
        .unwrap()
    }

    fn vector_irreps() -> Irreps {
        Irreps::new(vec![
            IrrepMul::new(1, Irrep::new(1, Parity::Odd).unwrap()).unwrap()
        ])
        .unwrap()
    }

    fn scalar_dynamic_plan() -> TensorProductPlan {
        TensorProductPlan::new(
            scalar_irreps(1),
            scalar_irreps(1),
            scalar_irreps(1),
            vec![TensorProductInstruction {
                in1_segment: 0,
                in2_segment: 0,
                out_segment: 0,
                cg_offset: 0,
                cg_len: 1,
                weight_offset: Some(0),
                path_weight: 2.0,
            }],
            vec![CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: 1.0,
            }],
            1,
        )
        .unwrap()
    }

    fn vector_dot_plan() -> TensorProductPlan {
        TensorProductPlan::new(
            vector_irreps(),
            vector_irreps(),
            scalar_irreps(1),
            vec![TensorProductInstruction {
                in1_segment: 0,
                in2_segment: 0,
                out_segment: 0,
                cg_offset: 0,
                cg_len: 3,
                weight_offset: None,
                path_weight: 1.0,
            }],
            vec![
                CgEntry {
                    i_m: 0,
                    j_m: 0,
                    k_m: 0,
                    value: INV_SQRT_3,
                },
                CgEntry {
                    i_m: 1,
                    j_m: 1,
                    k_m: 0,
                    value: INV_SQRT_3,
                },
                CgEntry {
                    i_m: 2,
                    j_m: 2,
                    k_m: 0,
                    value: INV_SQRT_3,
                },
            ],
            0,
        )
        .unwrap()
    }

    fn e3nn_tensor_product_smoke_plan() -> TensorProductPlan {
        let irreps_in1 = "2x0e + 1x1o".parse::<Irreps>().unwrap();
        let irreps_in2 = "1x0e + 1x1o".parse::<Irreps>().unwrap();
        let irreps_out = "2x0e + 1x1o".parse::<Irreps>().unwrap();
        let in1_segment = [0, 0, 1, 1, 0, 1, 2, 2, 2];
        let in2_segment = [0, 0, 0, 0, 1, 1, 0, 1, 1];
        let out_segment = [0, 1, 0, 1, 2, 2, 2, 0, 1];
        let cg_offset = [0, 0, 0, 0, 1, 1, 4, 7, 7];
        let cg_len = [1, 1, 1, 1, 3, 3, 3, 3, 3];
        let path_weight = [
            INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, 1.0, 1.0, 1.0, INV_SQRT_3, INV_SQRT_3,
        ];
        let instructions = (0..9)
            .map(|index| TensorProductInstruction {
                in1_segment: in1_segment[index],
                in2_segment: in2_segment[index],
                out_segment: out_segment[index],
                cg_offset: cg_offset[index],
                cg_len: cg_len[index],
                weight_offset: Some(index),
                path_weight: path_weight[index],
            })
            .collect();
        let cg_entries = vec![
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: 1.0,
            },
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 0,
                j_m: 1,
                k_m: 1,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 0,
                j_m: 2,
                k_m: 2,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 1,
                j_m: 0,
                k_m: 1,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 2,
                j_m: 0,
                k_m: 2,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 1,
                j_m: 1,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 2,
                j_m: 2,
                k_m: 0,
                value: INV_SQRT_3,
            },
        ];

        TensorProductPlan::new(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            cg_entries,
            9,
        )
        .unwrap()
    }

    fn scalar_projection(
        plan: &TensorProductPlan,
        x: &[f64],
        y: &[f64],
        w: Option<&[f64]>,
        grad_out: &[f64],
    ) -> f64 {
        let mut out = vec![0.0; plan.output_dim()];
        plan.forward(x, y, w, &mut out).unwrap();
        out.iter()
            .zip(grad_out)
            .map(|(value, grad)| *value * grad)
            .sum()
    }

    #[test]
    fn scalar_scalar_forward_zeroes_output() {
        let plan = scalar_dynamic_plan();
        let mut out = [99.0];

        plan.forward(&[3.0], &[-4.0], Some(&[0.5]), &mut out)
            .unwrap();

        assert_vec_close(&out, &[-12.0], 1.0e-12);
    }

    #[test]
    fn forward_accumulate_adds_to_existing_output() {
        let plan = scalar_dynamic_plan();
        let mut out = [1.25];

        plan.forward_accumulate(&[3.0], &[-4.0], Some(&[0.5]), &mut out)
            .unwrap();

        assert_vec_close(&out, &[-10.75], 1.0e-12);
    }

    #[test]
    fn hand_checkable_vector_dot_to_scalar_forward() {
        let plan = vector_dot_plan();
        let mut out = [0.0];

        plan.forward(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], None, &mut out)
            .unwrap();

        assert_close(out[0], (4.0 + 10.0 + 18.0) / SQRT_3, 1.0e-12);
    }

    #[test]
    fn rejects_invalid_plan_metadata() {
        let err = TensorProductPlan::new(
            vector_irreps(),
            scalar_irreps(1),
            scalar_irreps(1),
            vec![TensorProductInstruction {
                in1_segment: 0,
                in2_segment: 0,
                out_segment: 0,
                cg_offset: 0,
                cg_len: 1,
                weight_offset: None,
                path_weight: 1.0,
            }],
            vec![CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: 1.0,
            }],
            0,
        )
        .expect_err("1o x 0e cannot couple to 0e");
        assert!(matches!(err, CoreError::InvalidTensorProduct(_)));
    }

    #[test]
    fn reports_forward_shape_mismatch() {
        let plan = scalar_dynamic_plan();
        let err = plan
            .forward(&[], &[-4.0], Some(&[0.5]), &mut [0.0])
            .expect_err("short x should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));

        let err = plan
            .forward(&[3.0], &[-4.0], None, &mut [0.0])
            .expect_err("missing dynamic weight should fail");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn vjp_matches_finite_difference_and_accumulates() {
        let plan = TensorProductPlan::new(
            vector_irreps(),
            vector_irreps(),
            scalar_irreps(1),
            vec![TensorProductInstruction {
                in1_segment: 0,
                in2_segment: 0,
                out_segment: 0,
                cg_offset: 0,
                cg_len: 3,
                weight_offset: Some(0),
                path_weight: 1.25,
            }],
            vector_dot_plan().cg_entries,
            1,
        )
        .unwrap();
        let x = [0.7, -1.1, 2.3];
        let y = [-0.4, 0.9, 1.7];
        let w = [0.6];
        let grad_out = [-1.3];
        let mut grad_x = [0.25, -0.5, 0.75];
        let mut grad_y = [1.0, -1.25, 1.5];
        let mut grad_w = [-0.3];
        let initial_grad_x = grad_x;
        let initial_grad_y = grad_y;
        let initial_grad_w = grad_w;

        plan.vjp(
            &x,
            &y,
            Some(&w),
            &grad_out,
            &mut grad_x,
            &mut grad_y,
            Some(&mut grad_w),
        )
        .unwrap();

        for axis in 0..3 {
            let fd_x = central_difference(x[axis], |value| {
                let mut perturbed = x;
                perturbed[axis] = value;
                scalar_projection(&plan, &perturbed, &y, Some(&w), &grad_out)
            });
            assert_close(grad_x[axis] - initial_grad_x[axis], fd_x, 1.0e-8);

            let fd_y = central_difference(y[axis], |value| {
                let mut perturbed = y;
                perturbed[axis] = value;
                scalar_projection(&plan, &x, &perturbed, Some(&w), &grad_out)
            });
            assert_close(grad_y[axis] - initial_grad_y[axis], fd_y, 1.0e-8);
        }

        let fd_w = central_difference(w[0], |value| {
            scalar_projection(&plan, &x, &y, Some(&[value]), &grad_out)
        });
        assert_close(grad_w[0] - initial_grad_w[0], fd_w, 1.0e-8);
    }

    #[test]
    fn reports_vjp_shape_mismatch() {
        let plan = scalar_dynamic_plan();
        let mut grad_x = [0.0];
        let mut grad_y = [0.0];
        let err = plan
            .vjp(
                &[3.0],
                &[-4.0],
                Some(&[0.5]),
                &[1.0],
                &mut grad_x,
                &mut grad_y,
                None,
            )
            .expect_err("missing grad_w_edge should fail for dynamic weights");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }));
    }

    #[test]
    fn matches_pinned_e3nn_tensor_product_fixture_forward_and_vjp() {
        let plan = e3nn_tensor_product_smoke_plan();
        let x = [
            [
                -0.721_447_200_802_191_2,
                2.075_190_455_348_532_3,
                0.048_072_396_066_001_78,
                -0.722_865_442_306_87,
                -2.282_565_894_387_986_3,
            ],
            [
                1.165_170_084_389_448_8,
                0.200_581_857_917_803_92,
                1.481_340_762_187_279,
                -1.829_331_902_610_600_5,
                1.421_374_651_862_311_8,
            ],
            [
                0.682_499_516_285_206_8,
                0.467_355_066_715_404,
                -1.580_563_350_792_569_4,
                1.089_563_661_763_827_6,
                -1.405_496_187_320_962_6,
            ],
            [
                1.091_842_050_913_597_3,
                0.418_199_941_490_111_5,
                -0.581_723_133_619_545_4,
                2.411_024_268_904_223,
                0.277_444_300_314_674_7,
            ],
        ];
        let y = [
            [
                -0.689_132_337_898_742_4,
                1.661_072_955_909_904,
                -0.255_617_871_779_936_9,
                2.867_201_301_561_477_7,
            ],
            [
                -0.248_384_494_209_985_46,
                2.247_983_117_049_306,
                -0.745_599_884_196_589_5,
                -2.196_634_795_232_459,
            ],
            [
                0.384_674_551_031_459_4,
                0.354_169_863_524_918_7,
                0.731_063_890_392_520_1,
                0.608_397_938_921_576_3,
            ],
            [
                0.461_845_548_299_793_2,
                -0.909_442_468_965_574_9,
                -0.478_181_525_662_653_6,
                -1.473_634_252_523_174_7,
            ],
        ];
        let w = [
            [
                0.497_378_254_551_682_1,
                0.838_355_754_102_377_6,
                -0.534_970_294_418_594_3,
                3.103_838_042_222_489_6,
                0.309_604_481_971_132_3,
                -0.902_003_697_949_098_6,
                -0.352_110_727_479_395_3,
                -0.219_968_476_622_872_97,
                -1.440_644_849_296_675_5,
            ],
            [
                0.291_127_937_837_171_9,
                0.160_258_857_020_204_6,
                -0.795_626_768_823_621_6,
                -0.714_334_769_588_129_9,
                0.129_901_956_800_622_91,
                -0.943_444_746_708_808,
                0.737_009_863_087_843_6,
                -1.566_969_545_782_997,
                -0.389_200_260_040_500_35,
            ],
            [
                1.070_833_473_597_221_1,
                1.334_013_938_994_803_8,
                -0.477_883_126_819_653_34,
                0.286_200_422_424_222_5,
                0.351_823_721_315_372_43,
                -1.557_438_616_217_647_5,
                0.814_158_633_734_303,
                0.387_621_195_217_168_76,
                1.555_814_691_129_416_4,
            ],
            [
                0.161_867_548_897_594_75,
                0.016_256_800_291_439_832,
                -0.777_993_088_837_437_1,
                0.887_548_306_953_115_6,
                0.937_792_648_816_573_3,
                -0.047_234_132_117_004_904,
                0.659_403_612_732_893_7,
                -1.569_531_601_474_481_3,
                -0.049_370_431_704_885_88,
            ],
        ];
        let expected_out = [
            [
                1.044_934_528_220_366_4,
                0.693_660_916_846_213_6,
                -2.002_598_995_492_617,
                0.207_941_226_935_802_03,
                -3.788_114_066_091_741,
            ],
            [
                -0.846_714_151_951_731_3,
                -0.210_137_302_572_934_53,
                -0.205_727_566_870_457_93,
                0.209_649_839_833_251_44,
                -0.102_185_714_941_305_13,
            ],
            [
                0.032_817_200_758_838_705,
                -0.088_765_322_567_623_23,
                -0.385_531_082_059_961_67,
                -0.008_859_759_889_044_166,
                -0.425_468_131_362_323_5,
            ],
            [
                0.500_663_134_029_779_8,
                0.120_700_079_101_713_35,
                -0.629_538_483_907_033_4,
                0.146_696_044_893_744_5,
                -0.805_566_985_666_013_4,
            ],
        ];
        let grad_out = [
            [
                -0.524_744_460_323_494,
                0.243_245_673_650_702_07,
                0.495_232_769_297_852_26,
                -1.647_740_917_911_780_8,
                -0.405_281_212_596_509_2,
            ],
            [
                0.258_176_337_948_401_84,
                0.766_850_235_698_523,
                -0.625_598_030_116_999_2,
                -0.028_888_693_566_500_943,
                -0.671_232_814_678_045_7,
            ],
            [
                -1.217_065_996_139_772,
                -0.676_052_905_617_858_2,
                -0.681_630_103_206_196_7,
                -0.808_841_722_918_056,
                0.493_244_412_868_386_3,
            ],
            [
                -0.807_812_843_519_050_6,
                -1.285_722_933_824_572_1,
                -0.631_729_036_051_740_4,
                -0.494_570_992_580_311_06,
                1.035_026_994_367_087,
            ],
        ];
        let expected_grad_x = [
            [
                0.037_325_951_593_864_29,
                -0.454_674_206_172_128_5,
                -0.060_739_842_143_255_035,
                -0.210_815_843_695_925_35,
                -0.281_378_364_177_820_97,
            ],
            [
                -0.021_677_993_568_973_368,
                0.059_175_447_101_233_31,
                -0.460_666_994_690_904_66,
                0.177_775_348_915_049_12,
                0.585_697_232_105_034_4,
            ],
            [
                -0.597_936_194_968_232,
                0.565_142_583_049_663_5,
                -0.303_119_002_188_146_7,
                -0.517_529_762_199_707_1,
                -0.219_792_173_874_716_67,
            ],
            [
                -0.427_151_621_658_397_26,
                -0.117_223_660_180_009_47,
                -0.514_675_305_372_367_9,
                -0.299_170_612_245_178_3,
                -0.471_994_950_998_355_76,
            ],
        ];
        let expected_grad_y = [
            [
                0.829_637_266_133_869_3,
                -0.602_829_046_095_040_5,
                2.049_831_717_904_329,
                0.669_056_084_106_120_8,
            ],
            [
                -0.731_806_218_985_432_3,
                -0.333_451_968_244_359_7,
                0.429_313_026_440_479_5,
                -0.318_401_619_751_461_4,
            ],
            [
                -0.997_888_897_424_844_1,
                0.994_653_222_638_962_6,
                -0.325_568_143_392_485_4,
                0.574_891_316_631_927_2,
            ],
            [
                -0.424_117_987_598_384_04,
                -0.624_411_008_968_649_9,
                0.783_253_157_195_211_6,
                0.723_190_376_606_868,
            ],
        ];
        let expected_grad_w = [
            [
                -0.150_624_083_611_864_16,
                0.069_821_902_766_921_27,
                0.433_259_232_705_322_13,
                -0.200_837_630_300_704_12,
                -0.034_066_539_909_988_9,
                0.097_989_926_898_811_39,
                -0.851_436_648_061_686_2,
                1.098_455_768_361_674,
                -0.509_189_964_932_483_6,
            ],
            [
                -0.043_138_954_497_061_135,
                -0.128_133_808_414_583_13,
                -0.007_426_290_597_038_677_5,
                -0.022_057_996_251_550_31,
                0.060_314_587_571_753_75,
                0.010_383_043_812_035_248,
                0.262_136_927_226_909_07,
                0.135_261_824_544_880_45,
                0.401_762_465_365_761_1,
            ],
            [
                -0.184_480_006_340_255_38,
                -0.102_474_512_237_056_61,
                -0.126_326_339_599_600_66,
                -0.070_171_452_668_348_41,
                -0.209_881_726_691_601_68,
                -0.143_720_670_915_915_48,
                -0.110_419_263_431_595_24,
                0.250_856_872_376_344_9,
                0.139_345_374_862_284_5,
            ],
            [
                -0.235_183_421_813_881_47,
                -0.374_320_266_764_011_3,
                -0.090_080_514_081_420_6,
                -0.143_373_039_651_923_33,
                -0.450_236_296_182_543_07,
                -0.172_450_577_959_251_23,
                -0.143_394_284_830_497_54,
                0.278_080_137_347_753,
                0.442_595_104_667_534_66,
            ],
        ];

        for fixture_index in 0..x.len() {
            let mut out = [0.0; 5];
            plan.forward(
                &x[fixture_index],
                &y[fixture_index],
                Some(&w[fixture_index]),
                &mut out,
            )
            .unwrap();
            assert_vec_close(&out, &expected_out[fixture_index], 1.0e-12);

            let mut grad_x = [0.0; 5];
            let mut grad_y = [0.0; 4];
            let mut grad_w = [0.0; 9];
            plan.vjp(
                &x[fixture_index],
                &y[fixture_index],
                Some(&w[fixture_index]),
                &grad_out[fixture_index],
                &mut grad_x,
                &mut grad_y,
                Some(&mut grad_w),
            )
            .unwrap();
            assert_vec_close(&grad_x, &expected_grad_x[fixture_index], 1.0e-12);
            assert_vec_close(&grad_y, &expected_grad_y[fixture_index], 1.0e-12);
            assert_vec_close(&grad_w, &expected_grad_w[fixture_index], 1.0e-12);
        }
    }
}
