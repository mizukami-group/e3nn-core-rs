# Input derivatives

The core provides explicit derivative kernels for fixed-weight inference.
It does not implement training or a general-purpose automatic differentiation
engine. Use the Rust API documentation for the derivative operations supported
by each primitive.

For tensor-product paths, `out_k += path_weight * w_p * C_ijk * x_i * y_j`.
The VJP accumulates into the two input features and dynamic edge weights.
Dynamic weights receive cotangents so a caller can propagate them to distance;
stored learned parameters are fixed.

Spherical harmonics use component normalization. When input normalization is
enabled, the VJP includes the derivative of input normalization. Forward and
VJP convention fixtures cover l_max through 4. See
[mathematical conventions](MATH_CONVENTIONS.md) and [testing](TESTING.md).
