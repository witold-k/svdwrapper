// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cpu")]

use ndarray::{array, s, Array2, ArrayBase, Data, Ix2};
use svdwrapper::{Backend, Svd, SvdMode};

const EPSILON: f32 = 2.0e-4;

fn reconstruct(u: &Array2<f32>, s: &ndarray::Array1<f32>, vt: &Array2<f32>) -> Array2<f32> {
    let k = s.len();
    let u = u.slice(ndarray::s![.., ..k]);
    let u_s = &u * &s.view().insert_axis(ndarray::Axis(0));
    u_s.dot(&vt.slice(ndarray::s![..k, ..]))
}

fn assert_matrix_close(actual: &Array2<f32>, expected: &Array2<f32>) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, col), value) in actual.indexed_iter() {
        let diff = (*value - expected[(row, col)]).abs();
        assert!(diff <= EPSILON, "matrix mismatch at ({row}, {col}): diff={diff}");
    }
}

fn assert_valid_svd(a: &ArrayBase<impl Data<Elem = f32>, Ix2>) {
    let backend = Svd::<f32>::new(Backend::Cpu).expect("backend initialization failed");
    let output = backend.compute(a, SvdMode::Full).expect("CPU f32 SVD failed");

    assert_eq!(output.u.dim(), (a.nrows(), a.nrows()));
    assert_eq!(output.s.len(), a.nrows().min(a.ncols()));
    assert_eq!(output.vt.dim(), (a.ncols(), a.ncols()));

    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, &a.to_owned());

    let utu = output.u.t().dot(&output.u);
    let vv_t = output.vt.dot(&output.vt.t());
    assert_matrix_close(&utu, &Array2::<f32>::eye(output.u.ncols()));
    assert_matrix_close(&vv_t, &Array2::<f32>::eye(output.vt.nrows()));

    assert!(output.s.iter().all(|value| *value >= 0.0));
    assert!(output.s.windows(2).into_iter().all(|pair| pair[0] + EPSILON >= pair[1]));
}

#[test]
fn tall_matrix() {
    assert_valid_svd(&array![[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[10.,11.,13.]]);
}

#[test]
fn wide_matrix() {
    assert_valid_svd(&array![[1.,2.,3.,4.],[5.,7.,8.,9.],[10.,11.,13.,14.]]);
}

#[test]
fn square_identity_matrix() {
    assert_valid_svd(&Array2::<f32>::eye(4));
}

#[test]
fn rank_deficient_matrix() {
    assert_valid_svd(&array![[1.,2.,3.],[2.,4.,6.],[3.,6.,9.],[4.,8.,12.]]);
}

#[test]
fn zero_matrix() {
    assert_valid_svd(&Array2::<f32>::zeros((4, 3)));
}

#[test]
fn ill_conditioned_matrix() {
    assert_valid_svd(&array![[1.,0.,0.],[0.,1.0e-5,0.],[0.,0.,1.0e-5]]);
}

#[test]
fn non_contiguous_view() {
    let source = array![
        [1.,99.,2.,99.,3.],
        [4.,99.,5.,99.,6.],
        [7.,99.,8.,99.,10.],
        [11.,99.,12.,99.,13.]
    ];
    let view = source.slice(s![.., ..;2]);
    assert!(!view.is_standard_layout());
    assert_valid_svd(&view);
}

#[test]
fn empty_matrix_is_rejected() {
    let backend = Svd::<f32>::new(Backend::Cpu).expect("backend initialization failed");
    let a = Array2::<f32>::zeros((0, 3));
    assert!(backend.compute(&a, SvdMode::Full).is_err());
}

#[test]
#[ignore = "performance stress test"]
fn benchmark_cpu_large_matrix() {
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use std::time::Instant;

    const SIZE: usize = 10_000;

    let a = Array2::<f32>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = Svd::<f32>::new(Backend::Cpu).expect("backend initialization failed");

    let start = Instant::now();
    let output = backend.compute(&a, SvdMode::Full).expect("CPU f32 stress-test SVD failed");
    let elapsed = start.elapsed();

    println!("CPU f32 {SIZE}x{SIZE}: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", output.u.dim(), output.s.len(), output.vt.dim());
}

#[test]
fn reduced_mode_has_reduced_shapes() {
    let a = Array2::<f32>::from_shape_fn((4, 3), |(row, col)| (row * 3 + col + 1) as f32);
    let backend = Svd::<f32>::new(Backend::Cpu).expect("backend initialization failed");
    let output = backend
        .compute(&a, SvdMode::Reduced)
        .expect("reduced SVD failed");

    assert_eq!(output.u.dim(), (4, 3));
    assert_eq!(output.s.len(), 3);
    assert_eq!(output.vt.dim(), (3, 3));
    let reconstructed = reconstruct(&output.u, &output.s, &output.vt);
    assert_matrix_close(&reconstructed, &a);
}
