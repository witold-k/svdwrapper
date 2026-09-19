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

/// Factors produced by a singular value decomposition.
#[derive(Debug, Clone)]
pub struct SvdOutput<T> {
    /// Left singular vectors.
    pub u: Array2<T>,
    /// Singular values in descending order.
    pub s: Array1<T>,
    /// Transposed right singular vectors.
    pub vt: Array2<T>,
}

/// Result of an SVD computation.
pub type SvdResult<T> = Result<SvdOutput<T>, anyhow::Error>;

/// A unified abstraction trait that every underlying mathematical hardware backend must implement.
///
/// This trait ensures API symmetry across the available numerical backends. The execution
/// signature and output-shape invariants remain identical regardless of the implementation.
pub(crate) trait SvdBackend<T> {
    /// Computes `A = U * diag(S) * Vt` using the requested output mode.
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = T>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<T>;
}

