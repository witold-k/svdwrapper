// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#[cfg(feature = "cuda")]
use ndarray::Array2;
#[cfg(feature = "cuda")]
use ndarray_rand::{rand_distr::Uniform, RandomExt};
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use svdwrapper::{create_backend_f32, create_backend_f64, Backend};

// Intentionally large to exercise the backend under a realistic stress-test workload.
#[cfg(feature = "cuda")]
const SIZE: usize = 10_000;

#[cfg(feature = "cuda")]
fn run_f32() {
    let a = Array2::<f32>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = create_backend_f32(Backend::CudaF32);

    let start = Instant::now();
    let (u, s, vt) = backend.compute_svd(&a).expect("cuda f32 SVD failed");
    let elapsed = start.elapsed();

    println!("cuda f32: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", u.dim(), s.len(), vt.dim());
}

#[cfg(feature = "cuda")]
fn run_f64() {
    let a = Array2::<f64>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = create_backend_f64(Backend::CudaF64);

    let start = Instant::now();
    let (u, s, vt) = backend.compute_svd(&a).expect("cuda f64 SVD failed");
    let elapsed = start.elapsed();

    println!("cuda f64: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", u.dim(), s.len(), vt.dim());
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
