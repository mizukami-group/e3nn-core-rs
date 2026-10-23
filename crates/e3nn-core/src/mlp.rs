//! Fixed scalar MLP forward and input-only VJP for the v0.1 force path.
//!
//! This module intentionally propagates cotangents only to the input features.
//! It does not compute learned parameter gradients.

use crate::{CoreError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Tanh,
}

impl Activation {
    #[must_use]
    pub fn forward(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Tanh => x.tanh(),
        }
    }

    /// Derivative evaluated from this activation's forward output.
    ///
    /// The `output` argument must be the value returned by
    /// `self.forward(pre_activation)`.
    #[must_use]
    pub fn derivative_from_output(self, output: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Tanh => 1.0 - output * output,
        }
    }
}

/// SiLU activation, also known as swish: `x * sigmoid(x)`.
///
/// This is kept as a standalone primitive rather than an `Activation` variant
/// because `FixedMlp` currently stores only post-activation outputs in its VJP
/// cache. SiLU's derivative is not uniquely recoverable from the output alone.
#[must_use]
pub fn silu(value: f64) -> f64 {
    value * sigmoid(value)
}

/// Derivative of [`silu`] with respect to its pre-activation input.
#[must_use]
pub fn silu_derivative(value: f64) -> f64 {
    let sigmoid_value = sigmoid(value);
    sigmoid_value * (1.0 + value * (1.0 - sigmoid_value))
}

#[must_use]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseLayer {
    in_dim: usize,
    out_dim: usize,
    /// Row-major weights with shape `[out_dim, in_dim]`.
    weight: Vec<f64>,
    bias: Vec<f64>,
    activation: Activation,
}

impl DenseLayer {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        weight: Vec<f64>,
        bias: Vec<f64>,
        activation: Activation,
    ) -> Result<Self> {
        let layer = Self {
            in_dim,
            out_dim,
            weight,
            bias,
            activation,
        };
        layer.validate()?;
        Ok(layer)
    }

    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    #[must_use]
    pub fn weight(&self) -> &[f64] {
        &self.weight
    }

    #[must_use]
    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }

    fn validate(&self) -> Result<()> {
        if self.in_dim == 0 || self.out_dim == 0 {
            return Err(invalid_mlp(format!(
                "dense layer dimensions must be nonzero, got {} -> {}",
                self.in_dim, self.out_dim
            )));
        }
        let expected_weight = self.in_dim.checked_mul(self.out_dim).ok_or_else(|| {
            invalid_mlp(format!(
                "dense layer dimensions overflow: {} -> {}",
                self.in_dim, self.out_dim
            ))
        })?;
        if self.weight.len() != expected_weight {
            return Err(invalid_mlp(format!(
                "dense layer weight length mismatch for {} -> {}: expected {}, got {}",
                self.in_dim,
                self.out_dim,
                expected_weight,
                self.weight.len()
            )));
        }
        if self.bias.len() != self.out_dim {
            return Err(invalid_mlp(format!(
                "dense layer bias length mismatch for {} -> {}: expected {}, got {}",
                self.in_dim,
                self.out_dim,
                self.out_dim,
                self.bias.len()
            )));
        }
        if self
            .weight
            .iter()
            .chain(self.bias.iter())
            .any(|value| !value.is_finite())
        {
            return Err(invalid_mlp("dense layer parameters must be finite"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MlpCache {
    activations: Vec<Vec<f64>>,
}

impl MlpCache {
    pub fn clear(&mut self) {
        self.activations.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedMlp {
    layers: Vec<DenseLayer>,
}

impl FixedMlp {
    pub fn new(layers: Vec<DenseLayer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(invalid_mlp("fixed MLP must contain at least one layer"));
        }
        for (index, layer) in layers.iter().enumerate() {
            // Revalidate here because layer fields are private to this module
            // but FixedMlp owns the cross-layer invariants.
            layer.validate()?;
            if index > 0 {
                let previous = &layers[index - 1];
                if previous.out_dim != layer.in_dim {
                    return Err(invalid_mlp(format!(
                        "dense layer {index} input dim {} does not match previous output dim {}",
                        layer.in_dim, previous.out_dim
                    )));
                }
            }
        }
        Ok(Self { layers })
    }

    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].out_dim
    }

    #[must_use]
    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }

    /// Run the fixed MLP forward pass.
    ///
    /// Runtime inputs are assumed finite; non-finite values propagate according
    /// to normal f64 arithmetic. Learned parameters are validated at
    /// construction.
    pub fn forward(&self, x: &[f64], out: &mut [f64], cache: &mut MlpCache) -> Result<()> {
        check_len("fixed_mlp_forward input", self.input_dim(), x.len())?;
        check_len("fixed_mlp_forward output", self.output_dim(), out.len())?;

        // Reuse the persisting cache buffers as ping-pong scratch instead of
        // allocating a fresh input copy and a fresh per-layer activation Vec on
        // every call. Each layer reads its input directly from the previous
        // layer's cache buffer (or `x` for the first layer), so the per-output
        // accumulation order is identical to the previous implementation.
        let layer_count = self.layers.len();
        if cache.activations.len() != layer_count {
            cache.activations.resize_with(layer_count, Vec::new);
        }
        for (layer, activated) in self.layers.iter().zip(cache.activations.iter_mut()) {
            // Every slot is overwritten below, so the fill value is irrelevant;
            // this only ensures the buffer length matches `out_dim` while
            // retaining its capacity across calls.
            activated.resize(layer.out_dim, 0.0);
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let (input, activated): (&[f64], &mut [f64]) = if layer_index == 0 {
                (x, cache.activations[0].as_mut_slice())
            } else {
                let (head, tail) = cache.activations.split_at_mut(layer_index);
                (head[layer_index - 1].as_slice(), tail[0].as_mut_slice())
            };
            for (out_index, activated_value) in activated.iter_mut().enumerate() {
                let row_start = out_index * layer.in_dim;
                let mut pre_activation = layer.bias[out_index];
                for (in_index, input_value) in input.iter().enumerate() {
                    pre_activation += layer.weight[row_start + in_index] * input_value;
                }
                *activated_value = layer.activation.forward(pre_activation);
            }
        }
        out.copy_from_slice(&cache.activations[layer_count - 1]);
        Ok(())
    }

    /// Accumulate the input-only VJP contribution into `grad_x`.
    ///
    /// This intentionally adds to `grad_x` rather than zeroing it, matching the
    /// rest of the force-only VJP primitives.
    pub fn vjp_input_only(
        &self,
        cache: &MlpCache,
        grad_out: &[f64],
        grad_x: &mut [f64],
    ) -> Result<()> {
        check_len(
            "fixed_mlp_vjp_input_only grad_out",
            self.output_dim(),
            grad_out.len(),
        )?;
        check_len(
            "fixed_mlp_vjp_input_only grad_x",
            self.input_dim(),
            grad_x.len(),
        )?;
        self.check_cache(cache)?;

        // Ping-pong between two reused scratch buffers rather than allocating a
        // fresh `grad_input` Vec for every layer. The first (last-layer)
        // iteration reads its cotangent straight from `grad_out`, avoiding the
        // previous `grad_out.to_vec()` copy. `next` is re-zeroed and resized
        // before each layer's accumulation, so the per-input accumulation order
        // is identical to the previous implementation.
        let mut current: Vec<f64> = Vec::new();
        let mut next: Vec<f64> = Vec::new();
        for (iteration, layer_index) in (0..self.layers.len()).rev().enumerate() {
            let layer = &self.layers[layer_index];
            let activation = &cache.activations[layer_index];
            let grad_activation: &[f64] = if iteration == 0 { grad_out } else { &current };

            next.clear();
            next.resize(layer.in_dim, 0.0);
            for out_index in 0..layer.out_dim {
                let grad_pre = grad_activation[out_index]
                    * layer
                        .activation
                        .derivative_from_output(activation[out_index]);
                let row_start = out_index * layer.in_dim;
                for (in_index, grad_input_value) in next.iter_mut().enumerate() {
                    *grad_input_value += layer.weight[row_start + in_index] * grad_pre;
                }
            }

            std::mem::swap(&mut current, &mut next);
        }

        for (target, value) in grad_x.iter_mut().zip(current.iter()) {
            *target += *value;
        }
        Ok(())
    }

    fn check_cache(&self, cache: &MlpCache) -> Result<()> {
        if cache.activations.len() != self.layers.len() {
            return Err(invalid_mlp(
                "MLP cache does not match layer count; run forward before VJP",
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if cache.activations[index].len() != layer.out_dim {
                return Err(invalid_mlp(format!(
                    "MLP cache layer {index} activation length mismatch: expected {}, got {}",
                    layer.out_dim,
                    cache.activations[index].len()
                )));
            }
        }
        Ok(())
    }
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

fn invalid_mlp(message: impl Into<String>) -> CoreError {
    CoreError::InvalidMlp(message.into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

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

    fn sample_mlp() -> FixedMlp {
        FixedMlp::new(vec![
            DenseLayer::new(
                2,
                3,
                vec![0.5, -1.0, 1.5, 0.25, -0.75, 0.8],
                vec![0.1, -0.2, 0.3],
                Activation::Tanh,
            )
            .unwrap(),
            DenseLayer::new(3, 1, vec![1.2, -0.7, 0.4], vec![0.05], Activation::Identity).unwrap(),
        ])
        .unwrap()
    }

    fn scalar_projection(mlp: &FixedMlp, x: &[f64], grad_out: &[f64]) -> f64 {
        let mut cache = MlpCache::default();
        let mut out = vec![0.0; mlp.output_dim()];
        mlp.forward(x, &mut out, &mut cache).unwrap();
        out.iter()
            .zip(grad_out)
            .map(|(value, grad)| value * grad)
            .sum()
    }

    #[test]
    fn activation_derivatives_match_known_values() {
        assert_close(Activation::Identity.derivative_from_output(3.5), 1.0, 0.0);
        let y = 0.25_f64.tanh();
        assert_close(
            Activation::Tanh.derivative_from_output(y),
            1.0 - y * y,
            1.0e-15,
        );
    }

    #[test]
    fn forward_matches_known_values() {
        let mlp = sample_mlp();
        let mut cache = MlpCache::default();
        let mut out = [0.0];

        mlp.forward(&[0.4, -1.2], &mut out, &mut cache).unwrap();

        let h0 = (0.5_f64 * 0.4 + -1.0 * -1.2 + 0.1).tanh();
        let h1 = (1.5_f64 * 0.4 + 0.25 * -1.2 - 0.2).tanh();
        let h2 = (-0.75_f64 * 0.4 + 0.8 * -1.2 + 0.3).tanh();
        let expected = 1.2 * h0 + -0.7 * h1 + 0.4 * h2 + 0.05;
        assert_close(out[0], expected, 1.0e-12);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn vjp_input_only_matches_finite_difference_and_accumulates() {
        let mlp = sample_mlp();
        let x = [0.4, -1.2];
        let grad_out = [-0.8];
        let mut cache = MlpCache::default();
        let mut out = [0.0];
        let mut grad_x = [0.25, -0.5];
        let initial_grad_x = grad_x;

        mlp.forward(&x, &mut out, &mut cache).unwrap();
        mlp.vjp_input_only(&cache, &grad_out, &mut grad_x).unwrap();

        for axis in 0..x.len() {
            let fd = central_difference(x[axis], |value| {
                let mut perturbed = x;
                perturbed[axis] = value;
                scalar_projection(&mlp, &perturbed, &grad_out)
            });
            assert_close(grad_x[axis] - initial_grad_x[axis], fd, 1.0e-8);
        }
    }

    #[test]
    fn identity_only_vjp_is_weight_transpose_product() {
        let mlp = FixedMlp::new(vec![DenseLayer::new(
            2,
            2,
            vec![1.5, -0.5, 0.25, 2.0],
            vec![0.1, -0.2],
            Activation::Identity,
        )
        .unwrap()])
        .unwrap();
        let mut cache = MlpCache::default();
        let mut out = [0.0; 2];
        let mut grad_x = [0.0; 2];

        mlp.forward(&[0.4, -1.2], &mut out, &mut cache).unwrap();
        mlp.vjp_input_only(&cache, &[0.7, -1.1], &mut grad_x)
            .unwrap();

        assert_vec_close(
            &out,
            &[1.5 * 0.4 + -0.5 * -1.2 + 0.1, 0.25 * 0.4 + 2.0 * -1.2 - 0.2],
            1.0e-12,
        );
        assert_vec_close(
            &grad_x,
            &[1.5 * 0.7 + 0.25 * -1.1, -0.5 * 0.7 + 2.0 * -1.1],
            1.0e-12,
        );
    }

    #[test]
    fn zero_grad_output_produces_zero_input_contribution() {
        let mlp = sample_mlp();
        let mut cache = MlpCache::default();
        let mut out = [0.0];
        let mut grad_x = [1.25, -2.5];

        mlp.forward(&[0.4, -1.2], &mut out, &mut cache).unwrap();
        mlp.vjp_input_only(&cache, &[0.0], &mut grad_x).unwrap();

        assert_vec_close(&grad_x, &[1.25, -2.5], 1.0e-12);
    }

    #[test]
    fn reused_cache_matches_fresh_cache_bitwise() {
        let mlp = sample_mlp();
        let inputs = [[0.4, -1.2], [-0.9, 0.6]];
        let grad_out = [-0.8];

        // Reuse a single cache across both inputs (the production path).
        let mut reused_cache = MlpCache::default();
        // Fresh cache per input (the previous allocation behavior).
        for input in &inputs {
            let mut fresh_cache = MlpCache::default();
            let mut reused_out = [0.0];
            let mut fresh_out = [0.0];
            mlp.forward(input, &mut reused_out, &mut reused_cache)
                .unwrap();
            mlp.forward(input, &mut fresh_out, &mut fresh_cache)
                .unwrap();
            assert_eq!(reused_out[0].to_bits(), fresh_out[0].to_bits());

            let mut reused_grad = [0.7, -0.3];
            let mut fresh_grad = [0.7, -0.3];
            mlp.vjp_input_only(&reused_cache, &grad_out, &mut reused_grad)
                .unwrap();
            mlp.vjp_input_only(&fresh_cache, &grad_out, &mut fresh_grad)
                .unwrap();
            assert_eq!(reused_grad[0].to_bits(), fresh_grad[0].to_bits());
            assert_eq!(reused_grad[1].to_bits(), fresh_grad[1].to_bits());
        }
    }

    #[test]
    fn reports_shape_mismatches() {
        let mlp = sample_mlp();
        let mut cache = MlpCache::default();
        let mut out = [0.0];

        assert!(matches!(
            mlp.forward(&[0.4], &mut out, &mut cache),
            Err(CoreError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            mlp.forward(&[0.4, -1.2], &mut [], &mut cache),
            Err(CoreError::ShapeMismatch { .. })
        ));

        mlp.forward(&[0.4, -1.2], &mut out, &mut cache).unwrap();
        assert!(matches!(
            mlp.vjp_input_only(&cache, &[], &mut [0.0; 2]),
            Err(CoreError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            mlp.vjp_input_only(&cache, &[1.0], &mut [0.0; 1]),
            Err(CoreError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_invalid_mlp_definitions_and_stale_cache() {
        assert!(matches!(
            DenseLayer::new(2, 2, vec![1.0; 3], vec![0.0; 2], Activation::Identity),
            Err(CoreError::InvalidMlp(_))
        ));
        assert!(matches!(
            FixedMlp::new(Vec::new()),
            Err(CoreError::InvalidMlp(_))
        ));
        assert!(matches!(
            FixedMlp::new(vec![
                DenseLayer::new(2, 3, vec![0.0; 6], vec![0.0; 3], Activation::Identity).unwrap(),
                DenseLayer::new(2, 1, vec![0.0; 2], vec![0.0; 1], Activation::Identity).unwrap(),
            ]),
            Err(CoreError::InvalidMlp(_))
        ));

        let mlp = sample_mlp();
        let mut cache = MlpCache::default();
        let mut out = [0.0];
        mlp.forward(&[0.4, -1.2], &mut out, &mut cache).unwrap();
        cache.clear();
        assert!(matches!(
            mlp.vjp_input_only(&cache, &[1.0], &mut [0.0; 2]),
            Err(CoreError::InvalidMlp(_))
        ));
    }
}
