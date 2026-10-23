#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use e3nn_core::{
    CgEntry, Irreps, ShNormalization, SphericalHarmonicsPlan, TensorProductInstruction,
    TensorProductPlan,
};
use serde::Deserialize;

const REFERENCE_COMMIT: &str = "2aa7f58440a06b15352a2cbce01fa4c26f824969";
// `include_str!` requires literal paths; this constant is only for diagnostics.
const FIXTURE_DIR: &str = "fixtures/e3nn_torch/v0_6_0_2aa7f584";
const TOL: f64 = 1.0e-10;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    reference: ReferenceMetadata,
    python: PythonMetadata,
    seed: u64,
    dtype: String,
    conventions: ConventionMetadata,
    fixtures: Vec<FixtureMetadata>,
    generated_at: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceMetadata {
    name: String,
    repository: String,
    commit: String,
    package_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonMetadata {
    python_version: String,
    torch_version: String,
    numpy_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConventionMetadata {
    edge_direction: String,
    spherical_harmonics_normalization: String,
    spherical_harmonics_normalize_input: bool,
    spherical_harmonics_normalize: bool,
    primary_irreps_layout: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureMetadata {
    filename: String,
    operation: String,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    model: Option<serde_json::Value>,
    #[serde(default)]
    conventions: Option<FixtureConventions>,
    #[serde(default)]
    lmax: Option<usize>,
    #[serde(default)]
    irreps_in1: Option<String>,
    #[serde(default)]
    irreps_in2: Option<String>,
    #[serde(default)]
    irreps_out: Option<String>,
    #[serde(default)]
    weight_numel: Option<usize>,
    tensor_shapes: BTreeMap<String, Vec<usize>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureConventions {
    normalization: String,
    normalize: bool,
    normalize_input: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SphericalHarmonicsFixture {
    operation: String,
    lmax: Vec<usize>,
    vectors: Vec<[f64; 3]>,
    y: Vec<Vec<f64>>,
    grad_y: Vec<Vec<f64>>,
    grad_vectors: Vec<[f64; 3]>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorProductFixture {
    operation: String,
    irreps_in1: String,
    irreps_in2: String,
    irreps_out: String,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    w: Vec<Vec<f64>>,
    out: Vec<Vec<f64>>,
    grad_out: Vec<Vec<f64>>,
    grad_x: Vec<Vec<f64>>,
    grad_y: Vec<Vec<f64>>,
    grad_w: Vec<Vec<f64>>,
    weight_numel: Vec<usize>,
    instruction_in1_segment: Vec<usize>,
    instruction_in2_segment: Vec<usize>,
    instruction_out_segment: Vec<usize>,
    instruction_cg_offset: Vec<usize>,
    instruction_cg_len: Vec<usize>,
    instruction_weight_offset: Vec<usize>,
    instruction_path_weight: Vec<f64>,
    cg_i_m: Vec<usize>,
    cg_j_m: Vec<usize>,
    cg_k_m: Vec<usize>,
    cg_value: Vec<f64>,
}

fn load_manifest() -> Manifest {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/manifest.json"
    ))
    .expect("stable e3nn manifest must be valid JSON")
}

fn load_sh_fixture() -> SphericalHarmonicsFixture {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax2_component_normalized_f64.json"
    ))
    .expect("stable spherical-harmonics fixture must be valid JSON")
}

fn load_sh_lmax3_fixture() -> SphericalHarmonicsFixture {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax3_component_normalized_f64.json"
    ))
    .expect("stable lmax3 spherical-harmonics fixture must be valid JSON")
}

fn load_sh_lmax4_fixture() -> SphericalHarmonicsFixture {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/spherical_harmonics_lmax4_component_normalized_f64.json"
    ))
    .expect("stable lmax4 spherical-harmonics fixture must be valid JSON")
}

fn load_tp_fixture() -> TensorProductFixture {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/fully_connected_tensor_product_smoke_f64.json"
    ))
    .expect("stable tensor-product fixture must be valid JSON")
}

fn load_tp_lmax4_fixture() -> TensorProductFixture {
    serde_json::from_str(include_str!(
        "../fixtures/e3nn_torch/v0_6_0_2aa7f584/fully_connected_tensor_product_lmax4_smoke_f64.json"
    ))
    .expect("stable lmax4 tensor-product fixture must be valid JSON")
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= TOL,
        "{context}: expected {expected:.16e}, got {actual:.16e}, diff {:.3e}",
        (actual - expected).abs()
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: vector length mismatch"
    );
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_close(
            *actual_value,
            *expected_value,
            &format!("{context}[{index}]"),
        );
    }
}

fn validate_manifest(manifest: &Manifest) {
    assert_eq!(manifest.schema_version, "0.3.0");
    assert_eq!(manifest.reference.name, "e3nn-torch");
    assert_eq!(
        manifest.reference.repository,
        "https://github.com/e3nn/e3nn"
    );
    assert_eq!(manifest.reference.commit, REFERENCE_COMMIT);
    assert_eq!(manifest.reference.package_version, "0.6.0");
    assert!(!manifest.python.python_version.is_empty());
    assert!(!manifest.python.torch_version.is_empty());
    assert!(!manifest.python.numpy_version.is_empty());
    assert_eq!(manifest.seed, 12345);
    assert_eq!(manifest.dtype, "float64");
    assert!(!manifest.generated_at.is_empty());
    assert_eq!(
        manifest.conventions.edge_direction,
        "r_ij = R_j + shift_ij - R_i"
    );
    assert_eq!(
        manifest.conventions.spherical_harmonics_normalization,
        "component"
    );
    assert!(manifest.conventions.spherical_harmonics_normalize_input);
    assert!(manifest.conventions.spherical_harmonics_normalize);
    assert_eq!(
        manifest.conventions.primary_irreps_layout,
        "e3nn-compatible contiguous blocks"
    );

    let sh = fixture_metadata(
        manifest,
        "spherical_harmonics_lmax2_component_normalized_f64.npz",
    );
    assert_eq!(sh.operation, "spherical_harmonics");
    assert_eq!(sh.lmax, Some(2));
    let conventions = sh
        .conventions
        .as_ref()
        .expect("spherical-harmonics fixture metadata must include conventions");
    assert_eq!(conventions.normalization, "component");
    assert!(conventions.normalize);
    assert!(conventions.normalize_input);
    assert_shape(&sh.tensor_shapes, "vectors", &[5, 3]);
    assert_shape(&sh.tensor_shapes, "y", &[5, 9]);
    assert_shape(&sh.tensor_shapes, "grad_y", &[5, 9]);
    assert_shape(&sh.tensor_shapes, "grad_vectors", &[5, 3]);

    for (filename, lmax, width) in [
        (
            "spherical_harmonics_lmax3_component_normalized_f64.npz",
            3,
            16,
        ),
        (
            "spherical_harmonics_lmax4_component_normalized_f64.npz",
            4,
            25,
        ),
    ] {
        let sh = fixture_metadata(manifest, filename);
        assert_eq!(sh.operation, "spherical_harmonics");
        assert_eq!(sh.lmax, Some(lmax));
        let conventions = sh
            .conventions
            .as_ref()
            .expect("spherical-harmonics fixture metadata must include conventions");
        assert_eq!(conventions.normalization, "component");
        assert!(conventions.normalize);
        assert!(conventions.normalize_input);
        assert_shape(&sh.tensor_shapes, "vectors", &[5, 3]);
        assert_shape(&sh.tensor_shapes, "y", &[5, width]);
        assert_shape(&sh.tensor_shapes, "grad_y", &[5, width]);
        assert_shape(&sh.tensor_shapes, "grad_vectors", &[5, 3]);
    }

    let tp = fixture_metadata(manifest, "fully_connected_tensor_product_smoke_f64.npz");
    assert_eq!(tp.operation, "fully_connected_tensor_product");
    assert_eq!(tp.irreps_in1.as_deref(), Some("2x0e+1x1o"));
    assert_eq!(tp.irreps_in2.as_deref(), Some("1x0e+1x1o"));
    assert_eq!(tp.irreps_out.as_deref(), Some("2x0e+1x1o"));
    assert_eq!(tp.weight_numel, Some(9));
    assert_shape(&tp.tensor_shapes, "x", &[4, 5]);
    assert_shape(&tp.tensor_shapes, "y", &[4, 4]);
    assert_shape(&tp.tensor_shapes, "w", &[4, 9]);
    assert_shape(&tp.tensor_shapes, "out", &[4, 5]);
    assert_shape(&tp.tensor_shapes, "grad_out", &[4, 5]);
    assert_shape(&tp.tensor_shapes, "grad_x", &[4, 5]);
    assert_shape(&tp.tensor_shapes, "grad_y", &[4, 4]);
    assert_shape(&tp.tensor_shapes, "grad_w", &[4, 9]);
    assert_shape(&tp.tensor_shapes, "instructions", &[9]);
    assert_shape(&tp.tensor_shapes, "cg_entries", &[10]);

    let tp = fixture_metadata(
        manifest,
        "fully_connected_tensor_product_lmax4_smoke_f64.npz",
    );
    assert_eq!(tp.operation, "fully_connected_tensor_product");
    assert_eq!(tp.irreps_in1.as_deref(), Some("1x3o+1x4e"));
    assert_eq!(tp.irreps_in2.as_deref(), Some("1x1o+1x2e"));
    assert_eq!(tp.irreps_out.as_deref(), Some("1x2e+1x3o+1x4e"));
    assert_eq!(tp.weight_numel, Some(6));
    assert_shape(&tp.tensor_shapes, "x", &[3, 16]);
    assert_shape(&tp.tensor_shapes, "y", &[3, 8]);
    assert_shape(&tp.tensor_shapes, "w", &[3, 6]);
    assert_shape(&tp.tensor_shapes, "out", &[3, 21]);
    assert_shape(&tp.tensor_shapes, "grad_out", &[3, 21]);
    assert_shape(&tp.tensor_shapes, "grad_x", &[3, 16]);
    assert_shape(&tp.tensor_shapes, "grad_y", &[3, 8]);
    assert_shape(&tp.tensor_shapes, "grad_w", &[3, 6]);
    assert_shape(&tp.tensor_shapes, "instructions", &[6]);
    assert_shape(&tp.tensor_shapes, "cg_entries", &[222]);

    let synthetic = fixture_metadata(manifest, "synthetic_equivariant_v1_energy_force_f64.npz");
    assert_eq!(synthetic.operation, "synthetic_equivariant_energy_force");
    assert_eq!(
        synthetic.architecture.as_deref(),
        Some("synthetic-equivariant-v1")
    );
    assert!(synthetic.model.is_some());
    assert_shape(&synthetic.tensor_shapes, "species", &[3, 3]);
    assert_shape(&synthetic.tensor_shapes, "positions", &[3, 3, 3]);
    assert_shape(&synthetic.tensor_shapes, "pbc", &[3, 3]);
    assert_shape(&synthetic.tensor_shapes, "energy", &[3]);
    assert_shape(&synthetic.tensor_shapes, "forces", &[3, 3, 3]);
    assert_shape(&synthetic.tensor_shapes, "edge_count", &[3]);
    assert_shape(&synthetic.tensor_shapes, "centers", &[4]);
    assert_shape(&synthetic.tensor_shapes, "mlp_layer0_weight", &[3, 4]);
    assert_shape(&synthetic.tensor_shapes, "mlp_layer0_bias", &[3]);
    assert_shape(&synthetic.tensor_shapes, "mlp_layer1_weight", &[1, 3]);
    assert_shape(&synthetic.tensor_shapes, "mlp_layer1_bias", &[1]);
    assert_shape(&synthetic.tensor_shapes, "mlp_activations", &[2]);
    assert_shape(&synthetic.tensor_shapes, "edge_message_instructions", &[1]);
    assert_shape(&synthetic.tensor_shapes, "edge_message_cg_entries", &[3]);
    assert_shape(&synthetic.tensor_shapes, "node_energy_instructions", &[1]);
    assert_shape(&synthetic.tensor_shapes, "node_energy_cg_entries", &[3]);
}

fn fixture_metadata<'a>(manifest: &'a Manifest, filename: &str) -> &'a FixtureMetadata {
    manifest
        .fixtures
        .iter()
        .find(|fixture| fixture.filename == filename)
        .unwrap_or_else(|| {
            panic!("{FIXTURE_DIR}/manifest.json does not describe fixture {filename}")
        })
}

fn assert_shape(shapes: &BTreeMap<String, Vec<usize>>, name: &str, expected: &[usize]) {
    assert_eq!(
        shapes.get(name).map(Vec::as_slice),
        Some(expected),
        "manifest tensor_shapes.{name} mismatch"
    );
}

#[test]
fn stable_manifest_records_supported_reference_and_conventions() {
    validate_manifest(&load_manifest());
}

#[test]
fn spherical_harmonics_fixture_matches_rust_forward_and_vjp() {
    validate_manifest(&load_manifest());
    assert_spherical_harmonics_fixture_matches(&load_sh_fixture(), 2);
    assert_spherical_harmonics_fixture_matches(&load_sh_lmax3_fixture(), 3);
    assert_spherical_harmonics_fixture_matches(&load_sh_lmax4_fixture(), 4);
}

fn assert_spherical_harmonics_fixture_matches(fixture: &SphericalHarmonicsFixture, lmax: usize) {
    assert_eq!(fixture.operation, "spherical_harmonics");
    assert_eq!(fixture.lmax, vec![lmax]);
    assert_eq!(fixture.lmax.len(), 1);

    let plan = SphericalHarmonicsPlan::new(lmax, true, ShNormalization::Component).unwrap();
    assert_eq!(fixture.vectors.len(), fixture.y.len());
    assert_eq!(fixture.vectors.len(), fixture.grad_y.len());
    assert_eq!(fixture.vectors.len(), fixture.grad_vectors.len());
    assert_eq!(fixture.vectors.len(), 5);
    for (name, rows, expected_width) in [
        ("y", &fixture.y, plan.output_dim()),
        ("grad_y", &fixture.grad_y, plan.output_dim()),
    ] {
        assert!(
            rows.iter().all(|row| row.len() == expected_width),
            "spherical_harmonics.{name} rows must all have width {expected_width}"
        );
    }

    for fixture_index in 0..fixture.vectors.len() {
        let mut out = vec![0.0; plan.output_dim()];
        plan.forward(fixture.vectors[fixture_index], &mut out)
            .unwrap();
        assert_vec_close(
            &out,
            &fixture.y[fixture_index],
            &format!("spherical_harmonics.y[{fixture_index}]"),
        );

        let mut grad_r = [0.0; 3];
        plan.vjp_edge_vector(
            fixture.vectors[fixture_index],
            &fixture.grad_y[fixture_index],
            &mut grad_r,
        )
        .unwrap();
        assert_vec_close(
            &grad_r,
            &fixture.grad_vectors[fixture_index],
            &format!("spherical_harmonics.grad_vectors[{fixture_index}]"),
        );
    }
}

#[test]
fn tensor_product_fixture_matches_rust_forward_and_vjp() {
    validate_manifest(&load_manifest());
    assert_tensor_product_fixture_matches(&load_tp_fixture(), 4);
    assert_tensor_product_fixture_matches(&load_tp_lmax4_fixture(), 3);
}

fn assert_tensor_product_fixture_matches(fixture: &TensorProductFixture, expected_batch: usize) {
    assert_eq!(fixture.operation, "fully_connected_tensor_product");
    assert_eq!(fixture.weight_numel.len(), 1);

    let instruction_count = fixture.instruction_in1_segment.len();
    assert_eq!(instruction_count, fixture.instruction_in2_segment.len());
    assert_eq!(instruction_count, fixture.instruction_out_segment.len());
    assert_eq!(instruction_count, fixture.instruction_cg_offset.len());
    assert_eq!(instruction_count, fixture.instruction_cg_len.len());
    assert_eq!(instruction_count, fixture.instruction_weight_offset.len());
    assert_eq!(instruction_count, fixture.instruction_path_weight.len());

    let cg_count = fixture.cg_i_m.len();
    assert_eq!(cg_count, fixture.cg_j_m.len());
    assert_eq!(cg_count, fixture.cg_k_m.len());
    assert_eq!(cg_count, fixture.cg_value.len());

    let instructions = (0..instruction_count)
        .map(|index| TensorProductInstruction {
            in1_segment: fixture.instruction_in1_segment[index],
            in2_segment: fixture.instruction_in2_segment[index],
            out_segment: fixture.instruction_out_segment[index],
            cg_offset: fixture.instruction_cg_offset[index],
            cg_len: fixture.instruction_cg_len[index],
            weight_offset: Some(fixture.instruction_weight_offset[index]),
            path_weight: fixture.instruction_path_weight[index],
        })
        .collect();
    let cg_entries = (0..cg_count)
        .map(|index| CgEntry {
            i_m: fixture.cg_i_m[index],
            j_m: fixture.cg_j_m[index],
            k_m: fixture.cg_k_m[index],
            value: fixture.cg_value[index],
        })
        .collect();
    let plan = TensorProductPlan::new(
        fixture.irreps_in1.parse::<Irreps>().unwrap(),
        fixture.irreps_in2.parse::<Irreps>().unwrap(),
        fixture.irreps_out.parse::<Irreps>().unwrap(),
        instructions,
        cg_entries,
        fixture.weight_numel[0],
    )
    .unwrap();

    assert_eq!(fixture.x.len(), fixture.out.len());
    assert_eq!(fixture.x.len(), fixture.grad_out.len());
    assert_eq!(fixture.x.len(), fixture.grad_x.len());
    assert_eq!(fixture.x.len(), fixture.grad_y.len());
    assert_eq!(fixture.x.len(), fixture.grad_w.len());
    assert_eq!(fixture.x.len(), expected_batch);
    for (name, rows, expected_width) in [
        ("x", &fixture.x, plan.input1_dim()),
        ("y", &fixture.y, plan.input2_dim()),
        ("w", &fixture.w, plan.weight_numel),
        ("out", &fixture.out, plan.output_dim()),
        ("grad_out", &fixture.grad_out, plan.output_dim()),
        ("grad_x", &fixture.grad_x, plan.input1_dim()),
        ("grad_y", &fixture.grad_y, plan.input2_dim()),
        ("grad_w", &fixture.grad_w, plan.weight_numel),
    ] {
        assert!(
            rows.iter().all(|row| row.len() == expected_width),
            "tensor_product.{name} rows must all have width {expected_width}"
        );
    }

    for fixture_index in 0..fixture.x.len() {
        let mut out = vec![0.0; plan.output_dim()];
        plan.forward(
            &fixture.x[fixture_index],
            &fixture.y[fixture_index],
            Some(&fixture.w[fixture_index]),
            &mut out,
        )
        .unwrap();
        assert_vec_close(
            &out,
            &fixture.out[fixture_index],
            &format!("tensor_product.out[{fixture_index}]"),
        );

        let mut grad_x = vec![0.0; plan.input1_dim()];
        let mut grad_y = vec![0.0; plan.input2_dim()];
        let mut grad_w = vec![0.0; plan.weight_numel];
        plan.vjp(
            &fixture.x[fixture_index],
            &fixture.y[fixture_index],
            Some(&fixture.w[fixture_index]),
            &fixture.grad_out[fixture_index],
            &mut grad_x,
            &mut grad_y,
            Some(&mut grad_w),
        )
        .unwrap();
        assert_vec_close(
            &grad_x,
            &fixture.grad_x[fixture_index],
            &format!("tensor_product.grad_x[{fixture_index}]"),
        );
        assert_vec_close(
            &grad_y,
            &fixture.grad_y[fixture_index],
            &format!("tensor_product.grad_y[{fixture_index}]"),
        );
        assert_vec_close(
            &grad_w,
            &fixture.grad_w[fixture_index],
            &format!("tensor_product.grad_w[{fixture_index}]"),
        );
    }
}
