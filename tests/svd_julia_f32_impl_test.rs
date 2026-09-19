// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "julia")]

use ndarray::{s, Array2, ArrayBase, Data, Ix2};
use svdwrapper::{Backend, Svd, SvdMode};

const EPSILON: f32 = 2.0e-4;

fn reconstruct(u: &Array2<f32>, s: &ndarray::Array1<f32>, vt: &Array2<f32>) -> Array2<f32> {
    let k = s.len();
    let u = u.slice(ndarray::s![.., ..k]);
    let u_s = &u * &s.view().insert_axis(ndarray::Axis(0));
    u_s.dot(&vt.slice(ndarray::s![..k, ..]))
}

fn assert_valid_svd(a: &ArrayBase<impl Data<Elem = f32>, Ix2>) {
    let backend = Svd::<f32>::new(Backend::Julia).expect("Julia backend initialization failed");
    let output = backend.compute(a, SvdMode::Full).expect("Julia f32 SVD failed");

    let k = a.nrows().min(a.ncols());
    assert_eq!(output.u.dim(), (a.nrows(), a.nrows()));
    assert_eq!(output.s.len(), k);
    assert_eq!(output.vt.dim(), (a.ncols(), a.ncols()));

    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            let diff = (a[(row, col)] - reconstructed[(row, col)]).abs();
            assert!(diff <= EPSILON, "reconstruction mismatch at ({row}, {col}): diff={diff}");
        }
    }

    let utu = output.u.t().dot(&output.u);
    let vv_t = output.vt.dot(&output.vt.t());
    for row in 0..a.nrows() {
        for col in 0..a.nrows() {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!((utu[(row, col)] - expected).abs() <= EPSILON);
        }
    }
    for row in 0..a.ncols() {
        for col in 0..a.ncols() {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!((vv_t[(row, col)] - expected).abs() <= EPSILON);
        }
    }

    assert!(output.s.iter().all(|value| *value >= 0.0));
    assert!(output.s.windows(2).into_iter().all(|pair| pair[0] >= pair[1]));
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
    assert_valid_svd(&Array2::<f32>::eye(4));
}

#[test]
fn rank_deficient_matrix() {
    let a = Array2::from_shape_vec((3, 3), vec![1.,2.,3.,2.,4.,6.,3.,6.,9.]).unwrap();
    assert_valid_svd(&a);
}

#[test]
fn zero_matrix() {
    assert_valid_svd(&Array2::<f32>::zeros((3, 2)));
}

#[test]
fn ill_conditioned_matrix() {
    let a = Array2::from_diag(&ndarray::arr1(&[1.0, 1.0e-5, 1.0e-5]));
    assert_valid_svd(&a);
}

#[test]
fn non_contiguous_view() {
    let source = Array2::from_shape_vec((3, 6), (1..=18).map(|v| v as f32).collect()).unwrap();
    let view = source.slice(s![.., ..;2]);
    assert_valid_svd(&view);
}

#[test]
fn empty_matrix_is_rejected() {
    let backend = Svd::<f32>::new(Backend::Julia).expect("Julia backend initialization failed");
    let a = Array2::<f32>::zeros((0, 3));
    assert!(backend.compute(&a, SvdMode::Full).is_err());
}

#[test]
#[ignore = "performance smoke test"]
fn benchmark_repeated_svd() {
    use std::time::Instant;

    const SIZE: usize = 512;
    const REPEATS: usize = 3;

    let backend = Svd::<f32>::new(Backend::Julia).expect("Julia backend initialization failed");
    let a = Array2::from_shape_fn((SIZE, SIZE), |(row, col)| {
        (((row * 31 + col * 17) % 101) as f32 - 50.0) / 50.0
    });

    let first_start = Instant::now();
    backend.compute(&a, SvdMode::Full).expect("Julia f32 warm-up SVD failed");
    let first_elapsed = first_start.elapsed();

    let repeated_start = Instant::now();
    for _ in 0..REPEATS {
        backend.compute(&a, SvdMode::Full).expect("Julia f32 repeated SVD failed");
    }
    let repeated_elapsed = repeated_start.elapsed();

    println!(
        "Julia f32 {SIZE}x{SIZE}: first={first_elapsed:?}, repeated_total={repeated_elapsed:?}, repeated_avg={:?}",
        repeated_elapsed / REPEATS as u32
    );
}

#[test]
fn reduced_mode_has_reduced_shapes() {
    let a = Array2::<f32>::from_shape_fn((4, 3), |(row, col)| (row * 3 + col + 1) as f32);
    let backend = Svd::<f32>::new(Backend::Julia).expect("Julia backend initialization failed");
    let output = backend
        .compute(&a, SvdMode::Reduced)
        .expect("reduced SVD failed");

    assert_eq!(output.u.dim(), (4, 3));
    assert_eq!(output.s.len(), 3);
    assert_eq!(output.vt.dim(), (3, 3));
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            assert!((a[(row, col)] - reconstructed[(row, col)]).abs() <= EPSILON);
        }
    }
}

#[test]
fn reduced_mode_handles_wide_matrix() {
    let a = Array2::<f32>::from_shape_fn((3, 4), |(row, col)| (row * 4 + col + 1) as f32);
    let backend = Svd::<f32>::new(Backend::Julia).expect("Julia backend initialization failed");
    let output = backend
        .compute(&a, SvdMode::Reduced)
        .expect("reduced wide SVD failed");

    assert_eq!(output.u.dim(), (3, 3));
    assert_eq!(output.s.len(), 3);
    assert_eq!(output.vt.dim(), (3, 4));
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            assert!((a[(row, col)] - reconstructed[(row, col)]).abs() <= EPSILON);
        }
    }
}
