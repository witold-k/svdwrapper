// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::anyhow;
use ndarray::{ArrayBase, Data, Ix2};
use ndarray_linalg::{JobSvd, SVDDC};
use crate::svd::{SvdBackend, SvdResult};

pub struct CpuF64Svd;

impl SvdBackend<f64> for CpuF64Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> SvdResult<f64> {
        if a.nrows() == 0 || a.ncols() == 0 {
            return Err(anyhow!("SVD input matrix must be non-empty"));
        }

        let (u, singular_values, vt) = a.svddc(JobSvd::All)?;

        let u = u.ok_or_else(|| anyhow!("LAPACK did not return U"))?;
        let vt = vt.ok_or_else(|| anyhow!("LAPACK did not return V^T"))?;

        Ok((u, singular_values, vt))
    }
}
