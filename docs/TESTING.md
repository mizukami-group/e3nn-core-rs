# Local validation

Run `bash scripts/check.sh` from the repository root. This checks formatting,
strict Clippy, default and parallel/serde tests, documentation, and the usage
example. On macOS it also checks Accelerate. No CI service is used.

Tests include deterministic seeded properties, central finite differences,
and pinned e3nn forward/VJP fixture comparisons through l=4. The MACE radial
fixture checks radial functions only; this repository does not execute models.
Normal tests require neither Python nor upstream checkouts.

The crate declares Rust 1.89 as its minimum version. Release preparation checked all targets with Rust 1.89.0 and was
validated with Rust 1.95.0 on Apple M3 Ultra (macOS arm64). Linux BLAS providers
require separate native-library validation before claiming support on those
platforms. The checked-in lockfile fixes dependency resolution for this tree.

Fixture provenance is recorded in `crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/manifest.json`
and in the radial JSON. Upstream e3nn reference commit:
`2aa7f58440a06b15352a2cbce01fa4c26f824969` (0.6.0); MACE radial reference:
`4d2da09413ac1407f37cdbb6b81fa28e4c15655e` (0.3.16).
The fixture generators and upstream source trees are not bundled.

The original generation manifest is preserved verbatim as provenance, including
NPZ filenames and synthetic-model metadata. The bundled core fixtures are the
JSON conversions. NPZ files and the unrelated synthetic-model payload are not
included. Numerical fixture payloads and conventions are unchanged.

Version 0.1.0 local validation: 155 default tests, 162 parallel/serde tests,
and 162 Accelerate/parallel/serde tests. Package verification and a registry
dry run are performed without uploading to the registry.
