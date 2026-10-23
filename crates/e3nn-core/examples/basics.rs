use e3nn_core::{Irreps, SphericalHarmonicsPlan, Vec3};

fn main() -> e3nn_core::Result<()> {
    // Parse irreps; dimensions and layout
    let irreps: Irreps = "16x0e + 16x1o + 8x2e".parse()?;
    assert_eq!(irreps.dim(), 16 + 48 + 40);

    // Real spherical harmonics (component normalization, l_max <= 4)
    let plan = SphericalHarmonicsPlan::component(2, true)?;
    let r: Vec3 = [0.1, -0.4, 0.9];
    let mut y = vec![0.0_f64; plan.checked_output_dim()?];
    plan.forward(r, &mut y)?;
    assert_eq!(y.len(), 9); // (l_max + 1)^2

    Ok(())
}
