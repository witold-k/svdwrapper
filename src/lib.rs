// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

//! # svdwrapper
//!
//! A hardware-agnostic abstraction layer for computing the Singular Value Decomposition (SVD)
//! of two-dimensional matrices. This crate enables seamless runtime switching between CPU-based
//! LAPACK routines and GPU-accelerated backends (CUDA/cuSOLVER and OpenCL) via Cargo features.
//!
//! All mathematical evaluations yield a standardized `(U, Sigma, Vt)` tuple,
//! where `Sigma` is cross-platform harmonized as a fully populated 2D diagonal matrix.

pub mod svd;

#[cfg(feature = "cpu")]
pub mod svd_cpu_f32_impl;

#[cfg(feature = "cpu")]
pub mod svd_cpu_f64_impl;

#[cfg(feature = "cuda")]
pub mod svd_cuda_f32_impl;

#[cfg(feature = "cuda")]
pub mod svd_cuda_f64_impl;

#[cfg(feature = "opencl")]
pub mod svd_opencl_f32_impl;

#[cfg(feature = "opencl")]
pub mod svd_opencl_f64_impl;

#[cfg(feature = "julia")]
pub mod svd_julia_f32_impl;

#[cfg(feature = "julia")]
pub mod svd_julia_f64_impl;

use std::marker::PhantomData;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use crate::svd::SvdBackend;

#[cfg(feature = "cpu")]
use crate::svd_cpu_f32_impl::CpuF32Svd;
#[cfg(feature = "cpu")]
use crate::svd_cpu_f64_impl::CpuF64Svd;
#[cfg(feature = "cuda")]
use crate::svd_cuda_f32_impl::CudaF32Svd;
#[cfg(feature = "cuda")]
use crate::svd_cuda_f64_impl::CudaF64Svd;
#[cfg(feature = "julia")]
use crate::svd_julia_f32_impl::JuliaF32Svd;
#[cfg(feature = "julia")]
use crate::svd_julia_f64_impl::JuliaF64Svd;
#[cfg(feature = "opencl")]
use crate::svd_opencl_impl::OpenClSvd;

/// Supported execution backends for numerical SVD processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Classical CPU execution through dense LAPACK libraries (e.g., OpenBLAS, Intel MKL).
    CpuF32,
    /// Classical CPU execution through dense LAPACK libraries (e.g., OpenBLAS, Intel MKL).
    CpuF64,
    /// GPU-accelerated execution with 32-bit floating-point precision via Nvidia cuSOLVER.
    CudaF32,
    /// GPU-accelerated execution with 64-bit floating-point precision via Nvidia cuSOLVER (Jacobi method).
    CudaF64,
    /// Hardware-agnostic GPU/accelerator execution over OpenCL.
    OpenClF32,
    /// Hardware-agnostic GPU/accelerator execution over OpenCL.
    OpenClF64,
    /// use julia as middleware
    JuliaF32,
    /// use julia as middleware
    JuliaF64,
}

pub trait Precision: Sized {
    fn create_manager(backend: Backend) -> SvdManager<Self>;
}

/// The central resource and state manager for calculations on the chosen hardware pipeline.
///
/// This structure encapsulates hardware- and library-specific handles (such as cuSOLVER contexts Engine instances)
/// and exposes a unified, generic interface to the user.
pub enum SvdManager<T> {
    #[cfg(feature = "cpu")]
    CpuF32(CpuF32Svd),
    #[cfg(feature = "cpu")]
    CpuF64(CpuF64Svd),
    #[cfg(feature = "cuda")]
    CudaF32(CudaF32Svd),
    #[cfg(feature = "cuda")]
    CudaF64(CudaF64Svd),
    #[cfg(feature = "opencl")]
    OpenClF32(OpenClF32Svd),
    #[cfg(feature = "opencl")]
    OpenClF64(OpenClF64Svd),
    #[cfg(feature = "julia")]
    JuliaF32(JuliaF32Svd),
    #[cfg(feature = "julia")]
    JuliaF64(JuliaF64Svd),
    /// Internal type marker to accommodate generics without runtime memory overhead.
    _Marker(PhantomData<T>),
}

impl SvdManager<f64> {
    /// Computes the Singular Value Decomposition for a double-precision (`f64`) matrix.
    ///
    /// The method resolves the instantiated backend variant at runtime and routes the mathematical
    /// routine to the appropriate underlying hardware pipeline.
    ///
    /// # Parameters
    ///
    /// * `a` - A reference to a contiguous or fragmented 2D input matrix of type `f64`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` wrapping the initialized `(U, Sigma, Vt)` tuple on success:
    /// * `U` - The left orthogonal singular vector matrix ($M \times M$).
    /// * `Sigma` - The fully populated diagonal matrix containing the singular values ($M \times N$).
    /// * `Vt` - The transposed right orthogonal singular vector matrix ($N \times N$).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The requested backend was not compiled in the active Cargo build profile.
    /// * The underlying numerical algorithm fails to converge.
    /// * Memory allocation boundaries or GPU data transfers encounter failures.
    #[allow(unused_variables)]
    pub fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> anyhow::Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::CpuF64(b) => b.compute_svd(a).map_err(|e| anyhow::anyhow!(e)),
            #[cfg(feature = "cuda")]
            Self::CudaF64(b) => b.compute_svd(a),
            #[cfg(feature = "opencl")]
            Self::OpenF64Cl(b) => b.compute_svd(a),
            #[cfg(feature = "julia")]
            Self::JuliaF64(b) => b.compute_svd(a),
            _ => anyhow::bail!("The requested backend path is either not compiled or inactive for f64 execution."),
        }
    }
}

impl SvdManager<f32> {
    /// Computes the Singular Value Decomposition for a single-precision (`f32`) matrix.
    ///
    /// The method resolves the instantiated backend variant at runtime and routes the mathematical
    /// routine to the appropriate underlying hardware pipeline.
    ///
    /// # Parameters
    ///
    /// * `a` - A reference to a contiguous or fragmented 2D input matrix of type `f32`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` wrapping the initialized `(U, Sigma, Vt)` tuple on success:
    /// * `U` - The left orthogonal singular vector matrix ($M \times M$).
    /// * `Sigma` - The fully populated diagonal matrix containing the singular values ($M \times N$).
    /// * `Vt` - The transposed right orthogonal singular vector matrix ($N \times N$).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The requested backend was not compiled in the active Cargo build profile.
    /// * The underlying numerical algorithm fails to converge.
    /// * Memory allocation boundaries or GPU data transfers encounter failures.
    #[allow(unused_variables)]
    pub fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f32>, Ix2>,
    ) -> anyhow::Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::CpuF32(b) => b.compute_svd(a).map_err(|e| anyhow::anyhow!(e)),
            #[cfg(feature = "cuda")]
            Self::CudaF32(b) => b.compute_svd(a),
            #[cfg(feature = "opencl")]
            Self::OpenF32Cl(b) => b.compute_svd(a),
            #[cfg(feature = "julia")]
            Self::JuliaF32(b) => b.compute_svd(a),
            _ => anyhow::bail!("The requested backend path is either not compiled or inactive for f32 execution."),
        }
    }
}

/// Factory function to instantiate an `SvdManager` tailored for double-precision (`f64`) workloads.
///
/// # Panics
///
/// Panics if the targeted `Backend` has not been compiled via its corresponding Cargo feature flag
/// (e.g., `cuda`, `cpu`, or `opencl`) within the current build environment.
pub fn create_backend_f64(backend: Backend) -> SvdManager<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::CpuF64 => SvdManager::<f64>::CpuF64(CpuF64Svd),
        #[cfg(feature = "cuda")]
        Backend::CudaF64 => SvdManager::<f64>::CudaF64(CudaF64Svd::new().unwrap()),
        #[cfg(feature = "opencl")]
        Backend::OpenClF64 => SvdManager::<f64>::OpenCl(OpenClF64Svd::new().unwrap()),
        #[cfg(feature = "julia")]
        Backend::JuliaF64 => SvdManager::<f64>::OpenCl(JuliaF64Svd::new().unwrap()),
        _ => panic!("The requested f64 backend variant is not compiled in this build configuration."),
    }
}

/// Factory function to instantiate an `SvdManager` tailored for single-precision (`f32`) workloads.
///
/// # Panics
///
/// Panics if:
/// * A non-CUDA backend is requested (as `f32` SVD routines are exclusively implemented via CUDA in this crate).
/// * The `cuda` Cargo feature flag was missing during the compilation stage.
#[allow(unused_variables)]
pub fn create_backend_f32(backend: Backend) -> SvdManager<f32> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::CpuF32 => SvdManager::<f32>::CpuF32(CpuF32Svd),
        #[cfg(feature = "cuda")]
        Backend::CudaF32 => SvdManager::<f32>::CudaF32(CudaF32Svd::new().unwrap()),
        #[cfg(feature = "opencl")]
        Backend::OpenClF32 => SvdManager::<f32>::OpenCl(OpenClF32Svd::new().unwrap()),
        #[cfg(feature = "julia")]
        Backend::JuliaF32 => SvdManager::<f32>::OpenCl(JuliaF32Svd::new().unwrap()),
        _ => panic!("Only CudaF32 supports f32 SVD workloads in this compilation configuration."),
    }
}

