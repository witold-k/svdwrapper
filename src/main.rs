// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use svdwrapper::*;
use svdwrapper::Backend;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

fn svd32() {
    let start = Instant::now();

    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f32>::random((10000, 10000), dist);

    let backend = create_backend_f32(Backend::CudaF32);

    let (u, s, vh) = backend.compute_svd(&a).unwrap();

    println!("U =\n{u}");
    println!("S =\n{s}");
    println!("Vh =\n{vh}");

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
}

fn svd64() {
    let start = Instant::now();

    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f64>::random((10000, 10000), dist);

    let backend = create_backend_f64(Backend::CudaF64);

    let (u, s, vh) = backend.compute_svd(&a).unwrap();

    println!("U =\n{u}");
    println!("S =\n{s}");
    println!("Vh =\n{vh}");

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
}

fn main() {
    svd32();
    svd64();
}
