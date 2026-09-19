// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cuda")]

use ndarray::{array, s, Array2};
use svdwrapper::{Backend, Svd, SvdMode};

const EPSILON: f64 = 1.0e-10;

fn reconstruct(u: &Array2<f64>, s: &ndarray::Array1<f64>, vt: &Array2<f64>) -> Array2<f64> {
    let k = s.len();
    let u = u.slice(ndarray::s![.., ..k]);
    let u_s = &u * &s.view().insert_axis(ndarray::Axis(0));
    u_s.dot(&vt.slice(ndarray::s![..k, ..]))
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, epsilon: f64) {
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

fn assert_orthogonal(matrix: &Array2<f64>, epsilon: f64) {
    let identity = Array2::<f64>::eye(matrix.ncols());
    assert_matrix_close(&matrix.t().dot(matrix), &identity, epsilon);
}

fn assert_valid_svd(a: &Array2<f64>) {
    let backend = Svd::<f64>::new(Backend::Cuda).expect("CUDA backend initialization failed");
    let output = backend.compute(a, SvdMode::Full).expect("CUDA SVD failed");

    assert_eq!(output.u.dim(), (a.nrows(), a.nrows()));
    assert_eq!(output.s.len(), a.nrows().min(a.ncols()));
    assert_eq!(output.vt.dim(), (a.ncols(), a.ncols()));

    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, a, EPSILON);
    assert_orthogonal(&output.u, EPSILON);
    assert_orthogonal(&output.vt, EPSILON);

    assert!(output.s.iter().all(|value| *value >= -EPSILON));
    assert!(
        output.s.windows(2)
            .into_iter()
            .all(|window| window[0] + EPSILON >= window[1]),
        "singular values are not sorted descending: {:?}", output.s
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
    assert_valid_svd(&Array2::<f64>::eye(4));
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
    assert_valid_svd(&Array2::<f64>::zeros((4, 3)));
}

#[test]
fn ill_conditioned_matrix() {
    assert_valid_svd(&array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0e-12, 0.0],
        [0.0, 0.0, 1.0e-12]
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

    let backend = Svd::<f64>::new(Backend::Cuda).expect("CUDA backend initialization failed");
    let output = backend.compute(&view, SvdMode::Full).expect("CUDA SVD failed");
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, &view.to_owned(), EPSILON);
}

#[test]
fn empty_matrix_is_rejected() {
    let backend = Svd::<f64>::new(Backend::Cuda).expect("CUDA backend initialization failed");
    let empty = Array2::<f64>::zeros((0, 3));
    let error = backend.compute(&empty, SvdMode::Full).expect_err("empty input must fail");
    assert!(error.to_string().contains("non-empty"));
}

#[test]
fn reduced_mode_has_reduced_shapes() {
    let a = Array2::<f64>::from_shape_fn((4, 3), |(row, col)| (row * 3 + col + 1) as f64);
    let backend = Svd::<f64>::new(Backend::Cuda).expect("CUDA backend initialization failed");
    let output = backend
        .compute(&a, SvdMode::Reduced)
        .expect("reduced SVD failed");

    assert_eq!(output.u.dim(), (4, 3));
    assert_eq!(output.s.len(), 3);
    assert_eq!(output.vt.dim(), (3, 3));
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, &a, EPSILON);
}

#[test]
fn reduced_mode_handles_wide_matrix() {
    let a = Array2::<f64>::from_shape_fn((3, 4), |(row, col)| (row * 4 + col + 1) as f64);
    let backend = Svd::<f64>::new(Backend::Cuda).expect("Cuda backend initialization failed");
    let output = backend
        .compute(&a, SvdMode::Reduced)
        .expect("reduced wide SVD failed");

    assert_eq!(output.u.dim(), (3, 3));
    assert_eq!(output.s.len(), 3);
    assert_eq!(output.vt.dim(), (3, 4));
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, &a, EPSILON);
}
