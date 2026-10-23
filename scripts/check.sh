#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-4}"
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets --features parallel,serde -- -D warnings
cargo test --locked --workspace --features parallel,serde
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
cargo run --locked -p e3nn-core --example basics
if [[ "$(uname -s)" == Darwin ]]; then
  cargo clippy --locked --workspace --all-targets --features accelerate-blas,parallel,serde -- -D warnings
  cargo test --locked --workspace --features accelerate-blas,parallel,serde
fi
