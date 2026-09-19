// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cuda")]

use ndarray::{array, s, Array2};
use svdwrapper::svd::{mul_cpu_mat_vec_mat_f32, SvdBackend};
use svdwrapper::svd_cuda_f32_impl::CudaF32Svd;

const EPSILON: f32 = 2.0e-4;

fn assert_matrix_close(actual: &Array2<f32>, expected: &Array2<f32>, epsilon: f32) {
    assert_eq!(actual.dim(), expected.dim());
    for ((r, c), value) in actual.indexed_iter() {
        let diff = (*value - expected[[r, c]]).abs();
        assert!(
            diff <= epsilon,
            "matrix mismatch at [{r}, {c}]: actual={value}, expected={}, diff={diff}",
            expected[[r, c]]
        );
    }
}

fn assert_orthogonal(matrix: &Array2<f32>, epsilon: f32) {
    let identity = Array2::<f32>::eye(matrix.ncols());
    assert_matrix_close(&matrix.t().dot(matrix), &identity, epsilon);
}

fn assert_valid_svd(a: &Array2<f32>) {
    let backend = CudaF32Svd::new().expect("CUDA backend initialization failed");
    let (u, s, vt) = backend.compute_svd(a).expect("CUDA SVD failed");

    assert_eq!(u.dim(), (a.nrows(), a.nrows()));
    assert_eq!(s.len(), a.nrows().min(a.ncols()));
    assert_eq!(vt.dim(), (a.ncols(), a.ncols()));

    let reconstructed = mul_cpu_mat_vec_mat_f32(&u, &s, &vt);
    assert_matrix_close(&reconstructed, a, EPSILON);
    assert_orthogonal(&u, EPSILON);
    assert_orthogonal(&vt, EPSILON);

    assert!(s.iter().all(|value| *value >= -EPSILON));
    assert!(
        s.windows(2)
            .into_iter()
            .all(|window| window[0] + EPSILON >= window[1]),
        "singular values are not sorted descending: {s:?}"
    );
}

#[test]
fn tall_matrix() {
    assert_valid_svd(&array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [10.0, 11.0, 13.0]
    ]);
}

#[test]
fn wide_matrix() {
    assert_valid_svd(&array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 13.0, 14.0]
    ]);
}

#[test]
fn square_identity_matrix() {
    assert_valid_svd(&Array2::<f32>::eye(4));
}

#[test]
fn rank_deficient_matrix() {
    assert_valid_svd(&array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0]
    ]);
}

#[test]
fn zero_matrix() {
    assert_valid_svd(&Array2::<f32>::zeros((4, 3)));
}

#[test]
fn ill_conditioned_matrix() {
    assert_valid_svd(&array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0e-5, 0.0],
        [0.0, 0.0, 1.0e-5]
    ]);
}

#[test]
fn non_contiguous_view() {
    let source = array![
        [1.0, 99.0, 2.0, 99.0, 3.0],
        [4.0, 99.0, 5.0, 99.0, 6.0],
        [7.0, 99.0, 8.0, 99.0, 10.0],
        [11.0, 99.0, 12.0, 99.0, 13.0]
    ];
    let view = source.slice(s![.., ..;2]);
    assert!(!view.is_standard_layout());

    let backend = CudaF32Svd::new().expect("CUDA backend initialization failed");
    let (u, s, vt) = backend.compute_svd(&view).expect("CUDA SVD failed");
    let reconstructed = mul_cpu_mat_vec_mat_f32(&u, &s, &vt);
    assert_matrix_close(&reconstructed, &view.to_owned(), EPSILON);
}

#[test]
fn empty_matrix_is_rejected() {
    let backend = CudaF32Svd::new().expect("CUDA backend initialization failed");
    let empty = Array2::<f32>::zeros((0, 3));
    let error = backend.compute_svd(&empty).expect_err("empty input must fail");
    assert!(error.to_string().contains("non-empty"));
}
