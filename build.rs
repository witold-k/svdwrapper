use std::env;
use std::path::PathBuf;

fn main() {
    // 1. Tell Cargo to rerun this script ONLY if build.rs or specific env vars change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // 2. Dynamic Discovery of CUDA installation path
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include_path = format!("{}/include", cuda_home);

    // 3. Linker configurations gated by feature flags
    #[cfg(feature = "cpu")]
    {
        println!("cargo:rustc-link-lib=openblas");
    }

    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=magma");
        println!("cargo:rustc-link-lib=cusolver");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    }

    // 4. Generate bindings safely using bindgen
    let mut builder = bindgen::Builder::default()
        .header(format!("{}/cuda.h", cuda_include_path))
        .clang_arg(format!("-I{}", cuda_include_path))
        // Force bindgen to respect standard C booleans properly
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        // Use Rust-friendly enums instead of raw integers where possible
        .rustified_enum(".*")
        // Enforce the modern, updated naming convention (allowlist over whitelist)
        .allowlist_function("cu.*")
        .allowlist_function("cusolver.*")
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

    // 5. Write bindings to Cargo's official artifact output folder (OUT_DIR)
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("magma_bindings.rs"))
        .expect("Couldn't write bindings to target OUT_DIR!");
}

