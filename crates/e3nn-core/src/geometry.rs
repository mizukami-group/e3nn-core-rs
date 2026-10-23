/// 3D Cartesian vector used by the v0.1 CPU/f64 implementation.
pub type Vec3 = [f64; 3];

#[must_use]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[must_use]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[must_use]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[must_use]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Project-wide edge convention: `r_ij` = `R_j` + `shift_cart` - `R_i`.
#[must_use]
pub fn edge_vector(pos_i: Vec3, pos_j: Vec3, shift_cart: Vec3) -> Vec3 {
    sub(add(pos_j, shift_cart), pos_i)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn edge_vector_uses_project_convention() {
        let r = edge_vector([1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [0.5, 0.0, -1.0]);
        assert_eq!(r, [3.5, 4.0, 4.0]);
    }
}
