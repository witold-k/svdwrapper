// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "julia")]

use ndarray::{s, Array2, ArrayBase, Data, Ix2};
use svdwrapper::{create_backend_f64, svd::mul_cpu_mat_vec_mat_f64, Backend};

const EPSILON: f64 = 1.0e-10;

fn assert_valid_svd(a: &ArrayBase<impl Data<Elem = f64>, Ix2>) {
    let backend = create_backend_f64(Backend::JuliaF64);
    let (u, singular_values, vt) = backend.compute_svd(a).expect("Julia f64 SVD failed");

    let k = a.nrows().min(a.ncols());
    assert_eq!(u.dim(), (a.nrows(), k));
    assert_eq!(singular_values.len(), k);
    assert_eq!(vt.dim(), (k, a.ncols()));

    let reconstructed = mul_cpu_mat_vec_mat_f64(&u, &singular_values, &vt);
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            let diff = (a[(row, col)] - reconstructed[(row, col)]).abs();
            assert!(diff <= EPSILON, "reconstruction mismatch at ({row}, {col}): diff={diff}");
        }
    }

    let utu = u.t().dot(&u);
    let vv_t = vt.dot(&vt.t());
    for row in 0..k {
        for col in 0..k {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!((utu[(row, col)] - expected).abs() <= EPSILON);
            assert!((vv_t[(row, col)] - expected).abs() <= EPSILON);
        }
    }

    assert!(singular_values.iter().all(|value| *value >= 0.0));
    assert!(singular_values.windows(2).into_iter().all(|pair| pair[0] >= pair[1]));
}

#[test]
fn tall_matrix() {
    let a = Array2::from_shape_vec((4, 3), vec![3.,6.,9.,12.,15.,18.,21.,24.,27.,30.,33.,36.]).unwrap();
    assert_valid_svd(&a);
}

#[test]
fn wide_matrix() {
    let a = Array2::from_shape_vec((3, 4), vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.]).unwrap();
    assert_valid_svd(&a);
}

#[test]
fn square_identity_matrix() {
    assert_valid_svd(&Array2::<f64>::eye(4));
}

#[test]
fn rank_deficient_matrix() {
    let a = Array2::from_shape_vec((3, 3), vec![1.,2.,3.,2.,4.,6.,3.,6.,9.]).unwrap();
    assert_valid_svd(&a);
}

#[test]
fn zero_matrix() {
    assert_valid_svd(&Array2::<f64>::zeros((3, 2)));
}

#[test]
fn ill_conditioned_matrix() {
    let a = Array2::from_diag(&ndarray::arr1(&[1.0, 1.0e-12, 1.0e-12]));
    assert_valid_svd(&a);
}

#[test]
fn non_contiguous_view() {
    let source = Array2::from_shape_vec((3, 6), (1..=18).map(|v| v as f64).collect()).unwrap();
    let view = source.slice(s![.., ..;2]);
    assert_valid_svd(&view);
}

#[test]
fn empty_matrix_is_rejected() {
    let backend = create_backend_f64(Backend::JuliaF64);
    let a = Array2::<f64>::zeros((0, 3));
    assert!(backend.compute_svd(&a).is_err());
}

#[test]
#[ignore = "performance smoke test"]
fn benchmark_repeated_svd() {
    use std::time::Instant;

    const SIZE: usize = 512;
    const REPEATS: usize = 3;

    let backend = create_backend_f64(Backend::JuliaF64);
    let a = Array2::from_shape_fn((SIZE, SIZE), |(row, col)| {
        (((row * 31 + col * 17) % 101) as f64 - 50.0) / 50.0
    });

    let first_start = Instant::now();
    backend.compute_svd(&a).expect("Julia f64 warm-up SVD failed");
    let first_elapsed = first_start.elapsed();

    let repeated_start = Instant::now();
    for _ in 0..REPEATS {
        backend.compute_svd(&a).expect("Julia f64 repeated SVD failed");
    }
    let repeated_elapsed = repeated_start.elapsed();

    println!(
        "Julia f64 {SIZE}x{SIZE}: first={first_elapsed:?}, repeated_total={repeated_elapsed:?}, repeated_avg={:?}",
        repeated_elapsed / REPEATS as u32
    );
}
