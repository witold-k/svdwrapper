# svdwrapper

A highly optimized, hardware-agnostic abstraction layer for computing the Singular Value Decomposition (SVD) of 2D matrices in Rust. This crate allows seamless switching between CPU-based LAPACK routines and GPU-accelerated pipelines (Nvidia CUDA/cuSOLVER and OpenCL) via Cargo features.

Mathematical formulation: A = U * Sigma * Vt

---

## Architectural Overview

The core philosophy of svdwrapper is to separate the high-level matrix API from the underlying hardware-specific execution pipelines. It guarantees API Symmetry: whether you compute an SVD on an industrial server CPU or stream it to a data center GPU, the function signatures and matrix output shapes remain identical.

```text
+-----------------------------------------------------------------+
|                           Client Code                           |
+-----------------------------------------------------------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
SvdManager<f32>::compute_svd                      SvdManager<f64>::compute_svd
        |                                                 |
        v                                                 v
 [Pattern Matching]                                [Pattern Matching]
        |                                                 |
        +-------+                                 +-------+-------+
        |       |                                 |       |       |
        v       v                                 v       v       v
     [cuda]   [cpu]                            [cpu]   [cuda]  [opencl]
   CudaF32Svd CpuSvd                          CpuSvd CudaF64Svd OpenClSvd
  (cuSOLVER) (LAPACK)                        (LAPACK) (gesvdj) (Kernels)
```

Key Structural Highlights:

* Memory-Layout Bridge: ndarray natively stores data in a row-major (C-style) format,
    whereas vendor-provided high-performance libraries (LAPACK, cuSOLVER, Jacobi solvers)
    strictly require column-major (Fortran-style) configurations.
    svdwrapper handles this pivot layer safely under the hood,
    guaranteeing mathematically exact transformations while also handling non-contiguous matrix slices.

* Unified Output Layout: Standard LAPACK/cuSOLVER routines return singular values
    as a reduced 1D vector to minimize bandwidth.
    svdwrapper automatically projects these elements back into a fully populated 2D diagonal
    matrix (Sigma) of dimension M x N, eliminating asymmetric downstream multiplication constraints.

* RAII Resource Management: Hardware handles, primary driver contexts (CUcontext),
    and solver channels (cusolverDnHandle_t) are strictly tied to Rust's type system.
    All active device layers are cleanly de-allocated via Drop semantics upon scope exit, neutralizing GPU memory leaks.

---

## Prerequisites and Installation

Building this crate requires specific system dependencies depending on your active features.
Ensure your host system satisfies the following setup before building.

### 1. System Packages (Ubuntu/Debian)

To compile the underlying bindgen-generated MAGMA or cuSOLVER C bindings,
you must install the LLVM/Clang developer tooling and MAGMA development libraries:

```bash
sudo apt update
sudo apt install -y llvm-dev libclang-dev clang
sudo apt install -y libmagma-dev libmagma2
sudo apt install -y libopenblas-dev gfortran pkg-config
```

### 2. Environment Configuration

For CUDA-accelerated paths, the build system must be able to discover your local CUDA Toolkit installation.
Ensure the following rules are set up:

* CUDA_HOME environment variable must be exported and point to your CUDA directory.
* The Nvidia CUDA Compiler (nvcc) must be globally available inside your PATH.

```bash
Example setup (.bashrc / .zshrc):
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```
---

## Cargo Features

Tailor the crate's footprint to your target deployment infrastructure by enabling or disabling specific backends in your Cargo.toml:

```toml
[dependencies]
svdwrapper = { version = "0.1.0", features = ["cpu", "cuda"] }
```

```text
Feature Flag | Target Architecture   | Underlying Engine               | Precision
-------------+-----------------------+---------------------------------+-----------
"cpu"        | Standard x86_64 / ARM | System LAPACK (OpenBLAS/MKL)    | f64
"cuda"       | Nvidia GPUs           | CUDA Driver API & cuSOLVER      | f32 & f64
"opencl"     | Agnostic Accelerators | Custom OpenCL Compute Kernels   | f64
```

---

## Quick Start and Usage Examples

1. Single-Precision (f32) on Nvidia GPU

```rust
use svdwrapper::{create_backend_f32, Backend};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() -> anyhow::Result<()> {
    // 1. Generate a random rectangular matrix on the host CPU
    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f32>::random((4000, 3000), dist);

    // 2. Instantiate the CUDA f32 pipeline (allocates GPU handles internally)
    let backend = create_backend_f32(Backend::CudaF32);

    // 3. Stream data to device, execute full SVD, and collect results back into RAM
    let (u, sigma, vt) = backend.compute_svd(&a)?;

    println!("Decomposition successful!");
    println!("U matrix shape:     {:?}", u.shape());     // (4000, 4000)
    println!("Sigma matrix shape: {:?}", sigma.shape()); // (4000, 3000)
    println!("V^T matrix shape:   {:?}", vt.shape());    // (3000, 3000)

    Ok(())
}
```

2. Double-Precision (f64) on CPU (LAPACK)

```rust
use svdwrapper::{create_backend_f64, Backend};
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    let a = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0,  6.0,  9.0,
            12.0, 15.0, 18.0,
            21.0, 24.0, 27.0,
            30.0, 33.0, 36.0,
        ],
    )?;

    // Initialize CPU/LAPACK driver
    let backend = create_backend_f64(Backend::Cpu);
    let (u, sigma, vt) = backend.compute_svd(&a)?;

    // Mathematically reconstruct the original matrix: A = U * Sigma * Vt
    let reconstructed = u.dot(&sigma).dot(&vt);
    println!("Reconstruction check passed.");

    Ok(())
}
```

---

## Local Verification and Testing

The crate includes separate integration tests that validate the mathematical correctness of both
rectangular and square transformations down to strict machine epsilon boundaries (diff <= 1e-12 for double-precision).

Running Core CPU Tests:
```bash
cargo test --features cpu
```

Running Nvidia GPU Tests:
Ensure you have the Nvidia CUDA Toolkit and structural GPU drivers configured locally on your development machine before launching:
```bash
cargo test --features cuda
```

Running High-Volume Matrix Benchmarks:
To evaluate hardware execution scaling against massive arrays (10000 x 10000)
without logging flooding data to the standard output, trigger the performance benchmarks under the --release flag:
```bash
cargo test --release -- --ignored --nocapture
```

