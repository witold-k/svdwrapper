use std::path::{Path, PathBuf};
#[cfg(feature = "julia")]
use std::env;

#[cfg(any(feature = "cpu", feature = "julia"))]
fn cpu() {
    println!("cargo:rustc-link-lib=openblas");
}

#[cfg(feature = "opencl")]
fn opencl() {
    use std::env;
    use std::path::PathBuf;
    println!("cargo:rustc-link-lib=magma");
    let mut builder = bindgen::Builder::default()
        .allowlist_function("magma_.*");
    // Optional: Include MAGMA headers dynamically if your project requires it
    if std::path::Path::new("/usr/include/magma.h").exists() {
        builder = builder
            .header("/usr/include/magma.h")
            .clang_arg("-I/usr/include");
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate dynamic CUDA/MAGMA bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("magma_bindings.rs"))
        .expect("Couldn't write bindings to target OUT_DIR!");
}

#[cfg(feature = "cuda")]
fn cuda() {
    use std::env;
    use std::path::PathBuf;
    // 2. Dynamic Discovery of CUDA installation path
    let cuda_home = env::var("CUDA_HOME")
    .or_else(|_| env::var("CUDA_PATH"))
    .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_include_path = format!("{}/include", cuda_home);
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);

    let builder = bindgen::Builder::default()
        .header(format!("{}/cuda.h", cuda_include_path))
        .clang_arg(format!("-I{}", cuda_include_path))
        // Force bindgen to respect standard C booleans properly
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        // Use Rust-friendly enums instead of raw integers where possible
        .rustified_enum(".*")
        // Enforce the modern, updated naming convention (allowlist over whitelist)
        .allowlist_function("cu.*")
        .allowlist_function("cusolver.*");

    let bindings = builder
        .generate()
        .expect("Unable to generate dynamic CUDA/MAGMA bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("magma_bindings.rs"))
        .expect("Couldn't write bindings to target OUT_DIR!");
}

#[cfg(feature = "julia")]
fn julia() {
    // 1. User override
    println!("## CHECKING");
    if let Ok(dir) = env::var("JLRS_JULIA_DIR") {
        println!("cargo:rustc-env=JLRS_JULIA_DIR={dir}");
        println!("cargo:rustc-link-search=native={dir}/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}/lib");
        return;
    }

    // 2. Try juliaup default locations
    let home = env::var("HOME").unwrap();
    let candidates = [
        format!("{home}/.julia/juliaup"),
        format!("{home}/.local/share/juliaup"),
        format!("{home}/.juliaup"),
    ];

    for base in candidates {
        let path = PathBuf::from(&base);
        if path.exists() && let Some(julia_dir) = find_julia_dir(&path) {
            println!("cargo:rustc-env=JLRS_JULIA_DIR={}", julia_dir.display());
            if find_libjulia(&julia_dir).is_some() {
                println!("cargo:rustc-env=JLRS_JULIA_DIR={}", julia_dir.display());
                println!("cargo:rustc-link-search=native={}/lib", julia_dir.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", julia_dir.display());
            }
            return;
        }
    }

    panic!("Could not locate Julia installation. Set JLRS_JULIA_DIR manually.");
}

fn find_julia_dir(base: &PathBuf) -> Option<PathBuf> {
    // Recursively search for include/julia/julia_version.h
    for entry in walkdir::WalkDir::new(base).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.ends_with("include/julia/julia_version.h") {
            // Go up 3 levels: include/julia/julia_version.h → include/julia → include → <JULIA_DIR>
            return path.parent()?.parent()?.parent().map(|p| p.to_path_buf());
        }
    }
    None
}

fn find_libjulia(dir: &Path) -> Option<PathBuf> {
    let lib = dir.join("lib").join("libjulia.so");
    if lib.exists() {
        Some(lib)
    } else {
        None
    }
}

fn main() {
    // 1. Tell Cargo to rerun this script ONLY if build.rs or specific env vars change
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(any(feature = "cpu", feature = "julia"))]
    cpu();
    #[cfg(feature = "cuda")]
    cuda();
    #[cfg(feature = "opencl")]
    opencl();
    #[cfg(feature = "julia")]
    julia();
}

