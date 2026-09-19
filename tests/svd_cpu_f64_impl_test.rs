// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cpu")]

use ndarray::{array, s, Array2, ArrayBase, Data, Ix2};
use svdwrapper::{create_backend_f64, svd::{mul_cpu_mat_vec_mat_f64, SvdMode}, Backend};

const EPSILON: f64 = 1.0e-10;

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, col), value) in actual.indexed_iter() {
        let diff = (*value - expected[(row, col)]).abs();
        assert!(diff <= EPSILON, "matrix mismatch at ({row}, {col}): diff={diff}");
    }
}

fn assert_valid_svd(a: &ArrayBase<impl Data<Elem = f64>, Ix2>) {
    let backend = create_backend_f64(Backend::CpuF64);
    let (u, singular_values, vt) = backend.compute_svd(a, SvdMode::Full).expect("CPU f64 SVD failed");

    assert_eq!(u.dim(), (a.nrows(), a.nrows()));
    assert_eq!(singular_values.len(), a.nrows().min(a.ncols()));
    assert_eq!(vt.dim(), (a.ncols(), a.ncols()));

    let reconstructed = mul_cpu_mat_vec_mat_f64(&u, &singular_values, &vt);
    assert_matrix_close(&reconstructed, &a.to_owned());

    let utu = u.t().dot(&u);
    let vv_t = vt.dot(&vt.t());
    assert_matrix_close(&utu, &Array2::<f64>::eye(u.ncols()));
    assert_matrix_close(&vv_t, &Array2::<f64>::eye(vt.nrows()));

    assert!(singular_values.iter().all(|value| *value >= 0.0));
    assert!(singular_values.windows(2).into_iter().all(|pair| pair[0] + EPSILON >= pair[1]));
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
    assert_valid_svd(&Array2::<f64>::eye(4));
}

#[test]
fn rank_deficient_matrix() {
    assert_valid_svd(&array![[1.,2.,3.],[2.,4.,6.],[3.,6.,9.],[4.,8.,12.]]);
}

#[test]
fn zero_matrix() {
    assert_valid_svd(&Array2::<f64>::zeros((4, 3)));
}

#[test]
fn ill_conditioned_matrix() {
    assert_valid_svd(&array![[1.,0.,0.],[0.,1.0e-12,0.],[0.,0.,1.0e-12]]);
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
    let backend = create_backend_f64(Backend::CpuF64);
    let a = Array2::<f64>::zeros((0, 3));
    assert!(backend.compute_svd(&a, SvdMode::Full).is_err());
}

#[test]
#[ignore = "performance stress test"]
fn benchmark_cpu_large_matrix() {
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use std::time::Instant;

    const SIZE: usize = 10_000;

    let a = Array2::<f64>::random((SIZE, SIZE), Uniform::new(1.0, 10.0).unwrap());
    let backend = create_backend_f64(Backend::CpuF64);

    let start = Instant::now();
    let (u, singular_values, vt) = backend.compute_svd(&a, SvdMode::Full).expect("CPU f64 stress-test SVD failed");
    let elapsed = start.elapsed();

    println!("CPU f64 {SIZE}x{SIZE}: U={:?}, S={}, Vt={:?}, elapsed={elapsed:?}", u.dim(), singular_values.len(), vt.dim());
}

#[test]
fn reduced_mode_has_reduced_shapes() {
    let a = Array2::<f64>::from_shape_fn((4, 3), |(row, col)| (row * 3 + col + 1) as f64);
    let backend = create_backend_f64(Backend::CpuF64);
    let (u, s, vt) = backend
        .compute_svd(&a, SvdMode::Reduced)
        .expect("reduced SVD failed");

    assert_eq!(u.dim(), (4, 3));
    assert_eq!(s.len(), 3);
    assert_eq!(vt.dim(), (3, 3));
    let reconstructed = mul_cpu_mat_vec_mat_f64(&u, &s, &vt);
    assert_matrix_close(&reconstructed, &a);
}
