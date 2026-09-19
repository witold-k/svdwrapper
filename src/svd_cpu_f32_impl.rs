// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::anyhow;
use ndarray::{ArrayBase, Data, Ix2};
use ndarray_linalg::{JobSvd, SVDDC};
use crate::svd::{SvdBackend, SvdMode, SvdResult};

pub struct CpuF32Svd;

impl SvdBackend<f32> for CpuF32Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f32>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<f32> {
        if a.nrows() == 0 || a.ncols() == 0 {
            return Err(anyhow!("SVD input matrix must be non-empty"));
        }

        let job = match mode {
            SvdMode::Full => JobSvd::All,
            SvdMode::Reduced => JobSvd::Some,
        };
        let (u, singular_values, vt) = a.svddc(job)?;

        let u = u.ok_or_else(|| anyhow!("LAPACK did not return U"))?;
        let vt = vt.ok_or_else(|| anyhow!("LAPACK did not return V^T"))?;

        Ok((u, singular_values, vt))
    }
}
