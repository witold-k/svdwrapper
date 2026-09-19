#[cfg(feature = "julia")]
use std::path::{Path, PathBuf};
#[cfg(feature = "julia")]
use std::env;

#[cfg(any(feature = "cpu", feature = "julia"))]
fn cpu() {
    println!("cargo:rustc-link-lib=openblas");
}

#[cfg(feature = "cuda")]
fn cuda() {
    use std::env;
    // 2. Dynamic Discovery of CUDA installation path
    let cuda_home = env::var("CUDA_HOME")
    .or_else(|_| env::var("CUDA_PATH"))
    .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);

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

#[cfg(feature = "julia")]
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

#[cfg(feature = "julia")]
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
    #[cfg(feature = "julia")]
    julia();
}

