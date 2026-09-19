// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

//! # svdwrapper
//!
//! A hardware-agnostic abstraction layer for computing the Singular Value Decomposition (SVD)
//! of two-dimensional matrices. This crate enables seamless runtime switching between CPU-based
//! LAPACK routines and GPU-accelerated backends (CUDA/cuSOLVER and OpenCL) via Cargo features.
//!
//! All mathematical evaluations yield a standardized `(U, S, Vt)` tuple, where `S`
//! is a one-dimensional singular-value vector. Full and reduced output modes are supported.

pub mod svd;

#[cfg(feature = "cuda")]
mod cuda_common;

#[cfg(feature = "julia")]
mod julia_common;

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
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "julia",
    feature = "opencl"
))]
use crate::svd::{SvdBackend, SvdMode};

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
/// Supported execution backends for numerical SVD processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Classical CPU execution through dense LAPACK libraries (e.g., OpenBLAS, Intel MKL).
    CpuF32,
    /// Classical CPU execution through dense LAPACK libraries (e.g., OpenBLAS, Intel MKL).
    CpuF64,
    /// GPU-accelerated execution with 32-bit floating-point precision via Nvidia cuSOLVER.
    CudaF32,
    /// GPU-accelerated execution with 64-bit floating-point precision via NVIDIA cuSOLVER.
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
    /// Returns a `Result` wrapping the initialized `(U, S, Vt)` tuple on success:
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
        mode: SvdMode,
    ) -> anyhow::Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::CpuF64(b) => b.compute_svd(a, mode).map_err(|e| anyhow::anyhow!(e)),
            #[cfg(feature = "cuda")]
            Self::CudaF64(b) => b.compute_svd(a, mode),
            #[cfg(feature = "opencl")]
            Self::OpenClF64(b) => b.compute_svd(a, mode),
            #[cfg(feature = "julia")]
            Self::JuliaF64(b) => b.compute_svd(a, mode),
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
    /// Returns a `Result` wrapping the initialized `(U, S, Vt)` tuple on success:
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
        mode: SvdMode,
    ) -> anyhow::Result<(Array2<f32>, Array1<f32>, Array2<f32>)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::CpuF32(b) => b.compute_svd(a, mode).map_err(|e| anyhow::anyhow!(e)),
            #[cfg(feature = "cuda")]
            Self::CudaF32(b) => b.compute_svd(a, mode),
            #[cfg(feature = "opencl")]
            Self::OpenClF32(b) => b.compute_svd(a, mode),
            #[cfg(feature = "julia")]
            Self::JuliaF32(b) => b.compute_svd(a, mode),
            _ => anyhow::bail!("The requested backend path is either not compiled or inactive for f32 execution."),
        }
    }
}

/// Fallible factory for a double-precision (`f64`) SVD backend.
///
/// Prefer this function when backend initialization can fail at runtime, notably
/// for CUDA where driver, device, context, and cuSOLVER initialization are fallible.
pub fn try_create_backend_f64(backend: Backend) -> anyhow::Result<SvdManager<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::CpuF64 => Ok(SvdManager::<f64>::CpuF64(CpuF64Svd)),
        #[cfg(feature = "cuda")]
        Backend::CudaF64 => Ok(SvdManager::<f64>::CudaF64(CudaF64Svd::new()?)),
        #[cfg(feature = "julia")]
        Backend::JuliaF64 => Ok(SvdManager::<f64>::JuliaF64(JuliaF64Svd {})),
        _ => anyhow::bail!(
            "The requested f64 backend variant is not compiled in this build configuration."
        ),
    }
}

/// Compatibility factory for a double-precision (`f64`) SVD backend.
///
/// # Panics
///
/// Panics when the requested backend is unavailable or runtime initialization fails.
/// New code should prefer `try_create_backend_f64`.
pub fn create_backend_f64(backend: Backend) -> SvdManager<f64> {
    try_create_backend_f64(backend)
        .unwrap_or_else(|error| panic!("failed to create f64 SVD backend: {error:#}"))
}

/// Fallible factory for a single-precision (`f32`) SVD backend.
///
/// Prefer this function when backend initialization can fail at runtime, notably
/// for CUDA where driver, device, context, and cuSOLVER initialization are fallible.
pub fn try_create_backend_f32(backend: Backend) -> anyhow::Result<SvdManager<f32>> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::CpuF32 => Ok(SvdManager::<f32>::CpuF32(CpuF32Svd)),
        #[cfg(feature = "cuda")]
        Backend::CudaF32 => Ok(SvdManager::<f32>::CudaF32(CudaF32Svd::new()?)),
        #[cfg(feature = "julia")]
        Backend::JuliaF32 => Ok(SvdManager::<f32>::JuliaF32(JuliaF32Svd {})),
        _ => anyhow::bail!(
            "The requested f32 backend variant is not compiled in this build configuration."
        ),
    }
}

/// Compatibility factory for a single-precision (`f32`) SVD backend.
///
/// # Panics
///
/// Panics when the requested backend is unavailable or runtime initialization fails.
/// New code should prefer `try_create_backend_f32`.
pub fn create_backend_f32(backend: Backend) -> SvdManager<f32> {
    try_create_backend_f32(backend)
        .unwrap_or_else(|error| panic!("failed to create f32 SVD backend: {error:#}"))
}
