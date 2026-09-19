# svdwrapper

`svdwrapper` is an experimental Rust abstraction for computing dense singular value decompositions (SVD) through interchangeable numerical backends.

The intended public model is simple:

```text
A = U * diag(S) * Vt
```

Backends expose the same high-level Rust API and return `(U, S, Vt)`, with the singular values stored as a vector.

The project is under active development. CPU, CUDA and Julia paths already exist; the immediate work is to make those paths consistent, well-tested and documented. OpenCL and ROCm are deliberately lower-priority future work.

## Current status

| Backend | Precision | Status |
| --- | --- | --- |
| CPU / LAPACK | f32, f64 | Implemented; needs broader correctness tests and cleanup |
| NVIDIA CUDA / cuSOLVER | f32, f64 | Implemented; needs stronger error handling, cleanup and broader tests |
| Julia | f32, f64 | Implemented; needs cleanup, documentation and broader tests |
| OpenCL | f32, f64 | Experimental/incomplete; not part of the near-term stabilization target |
| AMD ROCm | - | Placeholder only; future work |

This is not yet a production-ready crate. APIs and backend internals may still change while the core implementation is being consolidated.

## Design goals

The main goal is a small backend-independent interface for dense SVD while keeping backend-specific code isolated.

In particular:

- CPU, CUDA and Julia implementations should expose equivalent semantics.
- Both `f32` and `f64` should be supported where the backend permits it.
- Singular values are represented as a one-dimensional vector rather than an expanded diagonal matrix.
- Rectangular matrices must be handled correctly for both `m > n` and `m < n`.
- Backend resources should follow Rust ownership/RAII rules.
- Errors from numerical libraries, CUDA and runtime setup should be propagated rather than hidden behind panics where practical.
- Consistent test layou - tests live under `tests/`, mirror the relative
  `src/` hierarchy, and use the source filename with a `_test.rs` suffix.

## Cargo features

No backend is enabled by default.

```toml
[dependencies]
svdwrapper = { version = "0.1.0", features = ["cpu"] }
```

Available feature flags:

```text
cpu     CPU SVD through ndarray-linalg / LAPACK
cuda    NVIDIA CUDA SVD through cuSOLVER
julia   Julia-backed SVD through jlrs
opencl  Experimental OpenCL/MAGMA path
rocm    Reserved for a future AMD ROCm backend
```

For example, CPU and CUDA can be enabled together:

```toml
svdwrapper = { version = "0.1.0", features = ["cpu", "cuda"] }
```

## Basic usage

### CPU, f64

```rust
use ndarray::Array2;
use svdwrapper::{create_backend_f64, Backend};

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

    let backend = create_backend_f64(Backend::CpuF64);
    let (u, s, vt) = backend.compute_svd(&a)?;

    println!("U:  {:?}", u.shape());
    println!("S:  {:?}", s.shape());
    println!("Vt: {:?}", vt.shape());

    Ok(())
}
```

For an `m x n` matrix, `S` contains `min(m, n)` singular values. The crate provides helper functions in `svdwrapper::svd` for reconstructing matrices from `U`, `S` and `Vt`.

### CUDA

The CUDA backend currently supports both `f32` and `f64`:

```rust
use svdwrapper::{try_create_backend_f32, Backend};

let backend = try_create_backend_f32(Backend::CudaF32)?;
let (u, s, vt) = backend.compute_svd(&a)?;
```

A working NVIDIA driver and CUDA toolkit are required. The build script looks for `CUDA_HOME` or `CUDA_PATH` and otherwise falls back to `/usr/local/cuda`.

## System dependencies

### CPU

The CPU path uses `ndarray-linalg` and a system BLAS/LAPACK implementation. On Debian/Ubuntu, OpenBLAS can be installed with:

```bash
sudo apt install libopenblas-dev gfortran pkg-config
```

Exact requirements can vary with the BLAS/LAPACK configuration used by `ndarray-linalg`.

### CUDA

The CUDA path requires:

- an NVIDIA GPU and driver,
- a compatible CUDA toolkit,
- CUDA headers and libraries visible to the build.

A typical local setup is:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

### Julia

The Julia backend uses `jlrs`. The build script can discover common Juliaup locations, or the Julia installation can be supplied explicitly:

```bash
export JLRS_JULIA_DIR=/path/to/julia
```

## Testing

Backend-specific tests can be selected through Cargo features:

```bash
cargo test --features cpu
cargo test --features cuda
cargo test --features julia
```

Some large-matrix timing tests are marked `#[ignore]` and are intended for explicit local runs rather than normal correctness testing.

The current test suite already checks basic reconstruction for rectangular matrices. It still needs to be expanded before the crate can be considered mature.

## Near-term roadmap

The next development phase is focused on completing and hardening the already useful backends rather than adding more hardware targets.

Planned near-term work:

1. Make the public documentation and implementation agree on the `(U, S, Vt)` representation.
2. Remove stale comments and backend naming inconsistencies.
3. Improve CUDA initialization, allocation and solver error handling.
4. Reduce avoidable panics in backend construction and return useful errors instead.
5. Expand correctness coverage for:
   - `m > n`, `m < n`, and square matrices,
   - rank-deficient matrices,
   - zero and identity matrices,
   - ill-conditioned inputs,
   - non-contiguous ndarray views,
   - orthogonality of `U` and `V`,
   - reconstruction `A ≈ U * diag(S) * Vt`,
   - non-negative, descending singular values.
6. Bring the Julia path to the same API and testing standard as CPU and CUDA.
7. Add focused documentation and examples once the interfaces stop moving.

## Later work

OpenCL and ROCm are intentionally not near-term goals.

The existing OpenCL/MAGMA code should currently be treated as experimental scaffolding rather than a supported backend. It can be revisited after CPU, CUDA and Julia are consistent and well-tested.

ROCm is a future backend idea only. No working ROCm implementation exists at present.

## Scope

`svdwrapper` is currently concerned with **dense SVD**. Sparse, randomized or truncated SVD algorithms are outside the present scope.

## License

Apache-2.0.
