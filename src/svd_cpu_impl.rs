// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

// src/svd_cpu_impl.rs
use anyhow::anyhow;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::SVD;
use crate::svd::{SvdBackend, SvdResult};

pub struct CpuSvd;

impl SvdBackend<f64> for CpuSvd {
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

        // FIX 2: Sigma muss exakt die Dimension M x N besitzen (wichtig für rechteckige Matrizen)
        let mut sigma = Array2::zeros((m, n));
        for (i, &val) in s.iter().enumerate() {
            if i < m && i < n {
                sigma[[i, i]] = val;
            }
        }

        Ok((u_mat, sigma, vt_mat))
    }
}
