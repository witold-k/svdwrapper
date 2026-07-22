// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::anyhow;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use jlrs::prelude::*;
use jlrs::data::layout::tuple::Tuple;
use jlrs::data::managed::array::TypedArray;
use jlrs::memory::target::frame::LocalGcFrame;
use crate::svd::{SvdBackend, SvdResult};

pub struct JuliaF32Svd;

const SVD_JL: &str = include_str!("svd_julia_f32.jl");

fn run_julia_svd<const N: usize>(frame: &mut LocalGcFrame<'_, N>) -> Result<()> {
    // Load the Julia code
    unsafe {
        Value::eval_string(&mut *frame, SVD_JL)
            .map_err(|e| anyhow::anyhow!(
                "Julia exception loading SVD code: {}",
                e.display_string_or("<unknown Julia error>")
            ))?;
    }

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let dims = &[2, 2];
    let julia_matrix = TypedArray::<f32>::from_vec(&mut *frame, a, dims)
        .map_err(|e| anyhow::anyhow!("Jlrs Error: {:?}", e))?
        .map_err(|e| anyhow::anyhow!("Julia Error: {}", e.display_string_or("unknown")))?;

    // Call svd_cuda
    let func = Module::main(frame)
        .global(&mut *frame, "svd_cpu_f32")?
        .as_value();

    let result = unsafe { func.call(&mut *frame, [julia_matrix.as_value()]) }
        .map_err(|e| anyhow::anyhow!(
            "Julia exception calling svd_cuda: {}",
            e.display_string_or("<unknown Julia error>")
        ))?;

    let u = result.get_nth_field(&mut *frame, 0)?.cast::<TypedArray<f32>>()?;
    let s = result.get_nth_field(&mut *frame, 1)?.cast::<TypedArray<f32>>()?;
    let vt = result.get_nth_field(&mut *frame, 2)?.cast::<TypedArray<f32>>()?;

    Ok((u.inline_data().as_slice(), s.inline_data().as_slice(), vt.inline_data().as_slice()))
}

impl SvdBackend<f32> for JuliaF32Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f32>, Ix2>,
    ) -> SvdResult<f32> {
        let (u, s, vt) = Builder::new()
            .n_threads(16)
            .start_mt(|mut mt| {
                mt.with(|handle| {
                    handle.local_scope::<_, 32>(|mut frame| -> Result<()> {
                        run_julia_svd(&mut frame)?;
                    })
                })
            })?;

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
