fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(e3nn_core_cblas_backend)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let accelerate = std::env::var_os("CARGO_FEATURE_ACCELERATE_BLAS").is_some();
    let openblas = std::env::var_os("CARGO_FEATURE_OPENBLAS").is_some();
    let intel_mkl = std::env::var_os("CARGO_FEATURE_INTEL_MKL").is_some();

    // e3nn-core's build-script-selected CBLAS backend is intentionally scoped
    // to macOS/Accelerate and Linux x86_64/OpenBLAS|MKL. On other targets
    // (notably Linux aarch64, where `llmlip/armpl` forwards `openblas` to
    // `e3nn-core/openblas`) this cfg stays unset and the so3 CBLAS path uses
    // its scalar fallback. Some so2 helpers still link BLAS directly through
    // feature cfgs, so this cfg is not a crate-wide "no BLAS on aarch64"
    // statement. This is a known, deliberate so3 non-support, not a link bug.
    let cblas_backend = (accelerate && target_os == "macos")
        || ((openblas || intel_mkl) && target_os == "linux" && target_arch == "x86_64");
    if cblas_backend {
        println!("cargo:rustc-cfg=e3nn_core_cblas_backend");
    }

    if accelerate && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if openblas && target_os == "linux" && target_arch == "x86_64" {
        println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
        if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        println!("cargo:rustc-link-lib=dylib=openblas");
    }

    if intel_mkl && target_os == "linux" && target_arch == "x86_64" {
        println!("cargo:rerun-if-env-changed=MKLROOT");
        let mkl_root =
            std::env::var("MKLROOT").unwrap_or_else(|_| "/opt/intel/oneapi/mkl/latest".to_string());
        for candidate in [
            format!("{mkl_root}/lib/intel64"),
            "/usr/lib/x86_64-linux-gnu".to_string(),
        ] {
            if std::path::Path::new(&candidate).is_dir() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
        println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=static=mkl_sequential");
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=static=mkl_sequential");
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
