// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#[cfg(feature = "cuda")]
use ndarray::Array2;
#[cfg(feature = "cuda")]
use ndarray_rand::{rand_distr::Uniform, RandomExt};
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use svdwrapper::{Backend, Svd, SvdMode};

// Intentionally large to exercise the backend under a realistic stress-test workload.
#[cfg(feature = "cuda")]
const SIZE: usize = 10_000;

#[cfg(feature = "cuda")]
fn run_f32() {
    let a = Array2::<f32>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = Svd::<f32>::new(Backend::Cuda).expect("backend initialization failed");

    let start = Instant::now();
    let output = backend.compute(&a, SvdMode::Full).expect("cuda f32 SVD failed");
    let elapsed = start.elapsed();

    println!("cuda f32: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", output.u.dim(), output.s.len(), output.vt.dim());
}

#[cfg(feature = "cuda")]
fn run_f64() {
    let a = Array2::<f64>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = Svd::<f64>::new(Backend::Cuda).expect("backend initialization failed");

    let start = Instant::now();
    let output = backend.compute(&a, SvdMode::Full).expect("cuda f64 SVD failed");
    let elapsed = start.elapsed();

    println!("cuda f64: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", output.u.dim(), output.s.len(), output.vt.dim());
}

#[cfg(feature = "cuda")]
fn main() {
    run_f32();
    run_f64();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires --features cuda");
}
