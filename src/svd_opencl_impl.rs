// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

// src/svd_opencl_impl.rs
use anyhow::anyhow;


mod magma {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
    include!("magma_bindings.rs");
}

use crate::svd::SvdBackend;
use ndarray::{Array2, ArrayBase, Data, Ix2};

pub struct OpenClSvd {}

impl OpenClSvd {
    pub fn new() -> Self {
        unsafe { magma::magma_init(); }
        OpenClSvd {}

    }
}

impl Drop for OpenClSvd {
    fn drop(&mut self) {
        unsafe { magma::magma_finalize(); }
    }
}

impl SvdBackend for OpenClSvd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), anyhow::Error> {
        let (m, n) = (a.nrows() as i32, a.ncols() as i32);
        let lda = m;

        // MAGMA expects column-major; ndarray is row-major.
        // Either convert layout or use .reversed_axes() + copy.
        let mut a_col = ndarray::Array2::<f64>::zeros((m as usize, n as usize));
        a_col.assign(&a.t()); // simple but not optimal

        // Allocate outputs
        let min_mn = m.min(n);
        let mut s = vec![0.0f64; min_mn as usize];
        let mut u = vec![0.0f64; (m * m) as usize];
        let mut vt = vec![0.0f64; (n * n) as usize];

        // Workspace query etc. omitted for brevity
        let jobu: u32 = b'A' as u32;
        let jobvt: u32 = b'A' as u32;
        let mut info: i32 = 0;

        unsafe {
            magma::magma_dgesvd(
                jobu,
                jobvt,
                m,
                n,
                a_col.as_mut_ptr(),
                lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                m,
                vt.as_mut_ptr(),
                n,
                std::ptr::null_mut(), // work
                0,                    // lwork
                &mut info,
            );
        }

        if info != 0 {
            return Err(anyhow!("magma_dgesvd failed with info={info}"));
        }

        // Convert back to ndarray
        let u_mat = Array2::from_shape_vec((m as usize, m as usize), u).unwrap();
        let vt_mat = Array2::from_shape_vec((n as usize, n as usize), vt).unwrap();
        let s_mat = {
            let mut diag = Array2::<f64>::zeros((m as usize, n as usize));
            for i in 0..(min_mn as usize) {
                diag[(i, i)] = s[i];
            }
            diag
        };

        Ok((u_mat, s_mat, vt_mat))
    }
}
