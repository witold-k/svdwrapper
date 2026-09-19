// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Controls the shape of the singular-vector matrices returned by an SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SvdMode {
    /// Return full singular-vector matrices: U is m x m and Vt is n x n.
    Full,
    /// Return reduced singular-vector matrices with k = min(m, n): U is m x k and Vt is k x n.
    Reduced,
}

/// Standardized SVD result `(U, S, Vt)`.
///
/// `S` contains `min(m, n)` singular values as a one-dimensional vector.
/// `U` and `Vt` use full or reduced shapes according to [`SvdMode`].
pub type SvdResult<T> = Result<(Array2<T>, Array1<T>, Array2<T>), anyhow::Error>;

/// A unified abstraction trait that every underlying mathematical hardware backend must implement.
///
/// This trait ensures cross-platform API symmetry. Whether computing on a local CPU thread via LAPACK
/// or streaming matrices down to a graphics card accelerator using cuSOLVER or OpenCL kernels, the
/// execution signature and shape output invariants remain entirely identical.
pub(crate) trait SvdBackend<T> {
    /// Computes `A = U * diag(S) * Vt` using the requested output mode.
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = T>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<T>;
}

