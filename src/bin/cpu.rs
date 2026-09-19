// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#[cfg(feature = "cpu")]
use ndarray::Array2;
#[cfg(feature = "cpu")]
use ndarray_rand::{rand_distr::Uniform, RandomExt};
#[cfg(feature = "cpu")]
use std::time::Instant;
#[cfg(feature = "cpu")]
use svdwrapper::{Backend, Svd, SvdMode};

// Intentionally large to exercise the backend under a realistic stress-test workload.
#[cfg(feature = "cpu")]
const SIZE: usize = 10_000;

#[cfg(feature = "cpu")]
fn run_f32() {
    let a = Array2::<f32>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = Svd::<f32>::new(Backend::Cpu).expect("backend initialization failed");

    let start = Instant::now();
    let output = backend.compute(&a, SvdMode::Full).expect("cpu f32 SVD failed");
    let elapsed = start.elapsed();

    println!("cpu f32: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", output.u.dim(), output.s.len(), output.vt.dim());
}

#[cfg(feature = "cpu")]
fn run_f64() {
    let a = Array2::<f64>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = Svd::<f64>::new(Backend::Cpu).expect("backend initialization failed");

    let start = Instant::now();
    let output = backend.compute(&a, SvdMode::Full).expect("cpu f64 SVD failed");
    let elapsed = start.elapsed();

    println!("cpu f64: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", output.u.dim(), output.s.len(), output.vt.dim());
}

#[cfg(feature = "cpu")]
fn main() {
    run_f32();
    run_f64();
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("This binary requires --features cpu");
}
