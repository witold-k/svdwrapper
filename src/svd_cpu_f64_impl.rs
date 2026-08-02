// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::anyhow;
use ndarray::{Array1, ArrayBase, Data, Ix2};
use ndarray_linalg::SVD;
use crate::svd::{SvdBackend, SvdResult};

pub struct CpuF64Svd;

impl SvdBackend<f64> for CpuF64Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> SvdResult<f64> {
        // Berechne SVD via LAPACK-Backend
        let (u, s, vt) = a.svd(true, true)?;

        // FIX 1: Sicheres Handling statt panikgefährdetem unwrap()
        let u_mat = u.ok_or_else(|| anyhow!("U-Matrix wurde von LAPACK nicht berechnet."))?;
        let vt_mat = vt.ok_or_else(|| anyhow!("V^T-Matrix wurde von LAPACK nicht berechnet."))?;

        let m = a.nrows();
        let n = a.ncols();
        let k = std::cmp::min(m, n);

        let mut sigma = Array1::zeros(k as usize);
        for i in 0..(k as usize) {
            sigma[[i]] = s[i];
        }

        Ok((u_mat, sigma, vt_mat))
    }
}
