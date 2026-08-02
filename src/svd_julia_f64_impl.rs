// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use jlrs::prelude::*;
use jlrs::data::managed::array::TypedArray;
use jlrs::memory::target::frame::LocalGcFrame;
use jlrs::data::managed::array::dimensions::Dims;
use crate::svd::{SvdBackend, SvdResult};

pub struct JuliaF64Svd;

const SVD_JL: &str = include_str!("svd_julia_f64.jl");

fn run_julia_svd<const N: usize>(
    frame: &mut LocalGcFrame<'_, N>,
    matrix_data: Vec<f64>,
    shape: [usize; 2],
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    // Load the Julia code
    unsafe {
        Value::eval_string(&mut *frame, SVD_JL).map_err(|e| {
            anyhow!(
                "Julia exception loading SVD code: {}",
                e.display_string_or("<unknown Julia error>")
            )
        })?;
    }

    let dims = &[shape[0], shape[1]];
    let julia_matrix = TypedArray::<f64>::from_vec(&mut *frame, matrix_data, dims)
        .map_err(|e| anyhow!("Jlrs Error: {:?}", e))?
        .map_err(|e| anyhow!("Julia Error: {}", e.display_string_or("unknown")))?;

    // 3. Funktion aufrufen
    let func = Module::main(frame)
        .global(&mut *frame, "svd_cpu_f64")?
        .as_value();

    let result = unsafe { func.call(&mut *frame, [julia_matrix.as_value()]) }.map_err(|e| {
        anyhow!(
            "Julia exception calling svd_cpu_f64: {}",
            e.display_string_or("<unknown Julia error>")
        )
    })?;

    // 4. Tupel-Felder extrahieren (U, S, Vt)
    let u_jl = result.get_nth_field(&mut *frame, 0)?.cast::<TypedArray<f64>>()?;
    let s_jl = result.get_nth_field(&mut *frame, 1)?.cast::<TypedArray<f64>>()?;
    let vt_jl = result.get_nth_field(&mut *frame, 2)?.cast::<TypedArray<f64>>()?;

    let u_dims = u_jl.dimensions().to_dimensions();
    let vt_dims = vt_jl.dimensions().to_dimensions();

    let u_rows = u_dims.n_elements(0).unwrap();
    let u_cols = u_dims.n_elements(1).unwrap();
    let vt_rows = vt_dims.n_elements(0).unwrap();
    let vt_cols = vt_dims.n_elements(1).unwrap();

    unsafe {
        // Rust ndarray expects Row-Major. Julia outputs Column-Major (Fortran order).
        // Passing shapes inverted (cols, rows) then reversing axes produces correct layouts.
        let u_slice = u_jl.inline_data().as_slice().to_vec();
        let s_slice = s_jl.inline_data().as_slice().to_vec();
        let vt_slice = vt_jl.inline_data().as_slice().to_vec();

        // Because Julia is Column-Major, we allocate the shapes inverted (cols, rows)
        // and instantly call `.reversed_axes()` to pivot into standard Row-Major without reallocating.
        let u = Array2::from_shape_vec((u_cols, u_rows), u_slice)?.reversed_axes();
        let s = Array1::from_vec(s_slice);
        let vt = Array2::from_shape_vec((vt_cols, vt_rows), vt_slice)?.reversed_axes();

        Ok((u, s, vt))
    }
}

impl SvdBackend<f64> for JuliaF64Svd {
    fn compute_svd(&self, a: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> SvdResult<f64> {
       let shape = [a.shape()[0], a.shape()[1]];

        // Linearize input array into column-major (Fortran Layout) for Julia compatibility
        let mut raw_data = Vec::with_capacity(a.len());
        for col in 0..shape[1] {
            for row in 0..shape[0] {
                raw_data.push(*a.get((row, col)).unwrap());
            }
        }

        let (u, s, vt) = Builder::new()
            .n_threads(16)
            .start_mt(|mut mt| {
                mt.with(|handle| {
                    handle.local_scope::<_, 32>(|mut frame| {
                        run_julia_svd(&mut frame, raw_data, shape)
                    })
                })
            })
            .expect("Failed to execute Julia runtime thread environment")
            .expect("Inner Julia evaluation failed");

        Ok((u, s, vt))
    }
}

