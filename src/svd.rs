// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use ndarray::{Array2, ArrayBase, Data, Ix2};

/// A core type alias representing the standardized output of a Singular Value Decomposition.
///
/// On success, it encapsulates a tuple containing three two-dimensional matrices `(U, Sigma, Vt)`:
/// * `U` - The left orthogonal singular vector matrix ($M \times M$).
/// * `Sigma` - The fully populated diagonal matrix containing the real singular values ($M \times N$).
/// * `Vt` - The transposed right orthogonal singular vector matrix ($N \times N$).
///
/// On failure, it passes an propagated `anyhow::Error` up the execution stack.
pub type SvdResult<T> = Result<(Array2<T>, Array2<T>, Array2<T>), anyhow::Error>;

/// A unified abstraction trait that every underlying mathematical hardware backend must implement.
///
/// This trait ensures cross-platform API symmetry. Whether computing on a local CPU thread via LAPACK
/// or streaming matrices down to a graphics card accelerator using cuSOLVER or OpenCL kernels, the
/// execution signature and shape output invariants remain entirely identical.
pub trait SvdBackend<T> {
    /// Computes the full Singular Value Decomposition ($A = U \cdot \Sigma \cdot V^T$) of a 2D matrix.
    ///
    /// Implementations are required to automatically handle non-square layout configurations ($M \neq N$)
    /// and guarantee that the returned `Sigma` matrix maintains a shape of exactly $M \times N$, rather
    /// than returning a reduced 1D singular value vector.
    ///
    /// # Parameters
    ///
    /// * `a` - A read-only reference to a 2D input matrix slice or array with elements of type `T`.
    ///
    /// # Returns
    ///
    /// Yields an [`SvdResult<T>`] containing the orthogonal states `U`, `Sigma`, and `Vt` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if hardware allocations fail, communication boundaries time out, or the iterative
    /// numerical decomposition subroutine runs out of optimization loops and fails to converge.
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = T>, Ix2>,
    ) -> SvdResult<T>;
}

