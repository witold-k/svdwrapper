// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

//! # svdwrapper
//!
//! Backend-independent dense singular value decomposition for f32 and f64.
//! CPU/LAPACK, CUDA/cuSOLVER and Julia backends share the same public API.

mod svd;

#[cfg(feature = "cuda")]
mod cuda_common;
#[cfg(feature = "julia")]
mod julia_common;
#[cfg(feature = "cpu")]
mod svd_cpu_f32_impl;
#[cfg(feature = "cpu")]
mod svd_cpu_f64_impl;
#[cfg(feature = "cuda")]
mod svd_cuda_f32_impl;
#[cfg(feature = "cuda")]
mod svd_cuda_f64_impl;
#[cfg(feature = "julia")]
mod svd_julia_f32_impl;
#[cfg(feature = "julia")]
mod svd_julia_f64_impl;

use ndarray::{ArrayBase, Data, Ix2};
use std::marker::PhantomData;

pub use crate::svd::{SvdMode, SvdResult};

#[cfg(any(feature = "cpu", feature = "cuda", feature = "julia"))]
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

/// Numerical backend used for the decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Cuda,
    Julia,
}

/// A configured SVD implementation for scalar type T.
pub struct Svd<T> {
    implementation: SvdImpl<T>,
}

enum SvdImpl<T> {
    #[cfg(feature = "cpu")]
    CpuF32(CpuF32Svd),
    #[cfg(feature = "cpu")]
    CpuF64(CpuF64Svd),
    #[cfg(feature = "cuda")]
    CudaF32(CudaF32Svd),
    #[cfg(feature = "cuda")]
    CudaF64(CudaF64Svd),
    #[cfg(feature = "julia")]
    JuliaF32(JuliaF32Svd),
    #[cfg(feature = "julia")]
    JuliaF64(JuliaF64Svd),
    _Marker(PhantomData<T>),
}

impl Svd<f32> {
    /// Creates an SVD implementation using the requested backend.
    pub fn new(backend: Backend) -> anyhow::Result<Self> {
        match backend {
            #[cfg(feature = "cpu")]
            Backend::Cpu => Ok(Self { implementation: SvdImpl::CpuF32(CpuF32Svd) }),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(Self { implementation: SvdImpl::CudaF32(CudaF32Svd::new()?) }),
            #[cfg(feature = "julia")]
            Backend::Julia => Ok(Self { implementation: SvdImpl::JuliaF32(JuliaF32Svd) }),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!(
                "the requested backend is not compiled in this build configuration"
            ),
        }
    }

    /// Computes A = U * diag(S) * Vt.
    pub fn compute(
        &self,
        a: &ArrayBase<impl Data<Elem = f32>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<f32> {
        match &self.implementation {
            #[cfg(feature = "cpu")]
            SvdImpl::CpuF32(backend) => backend.compute_svd(a, mode),
            #[cfg(feature = "cuda")]
            SvdImpl::CudaF32(backend) => backend.compute_svd(a, mode),
            #[cfg(feature = "julia")]
            SvdImpl::JuliaF32(backend) => backend.compute_svd(a, mode),
            _ => anyhow::bail!("invalid f32 SVD implementation"),
        }
    }
}

impl Svd<f64> {
    /// Creates an SVD implementation using the requested backend.
    pub fn new(backend: Backend) -> anyhow::Result<Self> {
        match backend {
            #[cfg(feature = "cpu")]
            Backend::Cpu => Ok(Self { implementation: SvdImpl::CpuF64(CpuF64Svd) }),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(Self { implementation: SvdImpl::CudaF64(CudaF64Svd::new()?) }),
            #[cfg(feature = "julia")]
            Backend::Julia => Ok(Self { implementation: SvdImpl::JuliaF64(JuliaF64Svd) }),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!(
                "the requested backend is not compiled in this build configuration"
            ),
        }
    }

    /// Computes A = U * diag(S) * Vt.
    pub fn compute(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<f64> {
        match &self.implementation {
            #[cfg(feature = "cpu")]
            SvdImpl::CpuF64(backend) => backend.compute_svd(a, mode),
            #[cfg(feature = "cuda")]
            SvdImpl::CudaF64(backend) => backend.compute_svd(a, mode),
            #[cfg(feature = "julia")]
            SvdImpl::JuliaF64(backend) => backend.compute_svd(a, mode),
            _ => anyhow::bail!("invalid f64 SVD implementation"),
        }
    }
}
