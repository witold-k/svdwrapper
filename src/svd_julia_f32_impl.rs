// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use crate::julia_common;
use crate::svd::{SvdBackend, SvdResult};
use anyhow::{anyhow, Result};
use jlrs::data::managed::array::dimensions::Dims;
use jlrs::data::managed::array::TypedArray;
use jlrs::memory::target::frame::GcFrame;
use jlrs::prelude::*;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

pub struct JuliaF32Svd;

fn run_julia_svd(
    mut frame: GcFrame<'_>,
    matrix_data: Vec<f32>,
    shape: [usize; 2],
) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>)> {
    let dims = &[shape[0], shape[1]];
    let julia_matrix = TypedArray::<f32>::from_vec(&mut frame, matrix_data, dims)
        .map_err(|e| anyhow!("jlrs error creating Julia matrix: {e:?}"))?
        .map_err(|e| anyhow!("Julia error creating matrix: {}", e.display_string_or("unknown")))?;

    let func = Module::main(&frame)
        .global(&mut frame, "svd_cpu_f32")
        .map_err(|e| anyhow!("failed to resolve Julia function svd_cpu_f32: {e}"))?
        .as_value();

    let result = unsafe { func.call(&mut frame, [julia_matrix.as_value()]) }.map_err(|e| {
        anyhow!(
            "Julia exception calling svd_cpu_f32: {}",
            e.display_string_or("<unknown Julia error>")
        )
    })?;

    let u_jl = result.get_nth_field(&mut frame, 0)?.cast::<TypedArray<f32>>()?;
    let s_jl = result.get_nth_field(&mut frame, 1)?.cast::<TypedArray<f32>>()?;
    let vt_jl = result.get_nth_field(&mut frame, 2)?.cast::<TypedArray<f32>>()?;

    let u_dims = u_jl.dimensions().to_dimensions();
    let vt_dims = vt_jl.dimensions().to_dimensions();
    let u_rows = u_dims.n_elements(0).ok_or_else(|| anyhow!("Julia U result is not two-dimensional"))?;
    let u_cols = u_dims.n_elements(1).ok_or_else(|| anyhow!("Julia U result is not two-dimensional"))?;
    let vt_rows = vt_dims.n_elements(0).ok_or_else(|| anyhow!("Julia Vt result is not two-dimensional"))?;
    let vt_cols = vt_dims.n_elements(1).ok_or_else(|| anyhow!("Julia Vt result is not two-dimensional"))?;

    unsafe {
        let u_slice = u_jl.inline_data().as_slice().to_vec();
        let s_slice = s_jl.inline_data().as_slice().to_vec();
        let vt_slice = vt_jl.inline_data().as_slice().to_vec();

        let u = Array2::from_shape_vec((u_cols, u_rows), u_slice)?.reversed_axes();
        let s = Array1::from_vec(s_slice);
        let vt = Array2::from_shape_vec((vt_cols, vt_rows), vt_slice)?.reversed_axes();

        Ok((u, s, vt))
    }
}

impl SvdBackend<f32> for JuliaF32Svd {
    fn compute_svd(&self, a: &ArrayBase<impl Data<Elem = f32>, Ix2>) -> SvdResult<f32> {
        let shape = [a.nrows(), a.ncols()];
        if shape[0] == 0 || shape[1] == 0 {
            return Err(anyhow!("SVD requires a non-empty matrix"));
        }

        let mut raw_data = Vec::with_capacity(a.len());
        for col in 0..shape[1] {
            for row in 0..shape[0] {
                raw_data.push(a[(row, col)]);
            }
        }

        julia_common::blocking_task(move |frame| run_julia_svd(frame, raw_data, shape))
    }
}
