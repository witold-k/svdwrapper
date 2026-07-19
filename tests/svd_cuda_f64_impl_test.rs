// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cuda")]

use svdwrapper::{create_backend_f64, Backend}; // Passe den Cratename ggf. an deine Cargo.toml an
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

#[test]
fn test_cuda_f64_svd_correctness() {
    // 1. Erstelle eine definierte, nicht-quadratische Testmatrix (M = 4, N = 3)
    // Damit wird verifiziert, ob min(M,N) im Jacobi-Pfad korrekt isoliert wird.
    let a = Array2::from_shape_vec(
        (4, 3),
        vec![
            2.0,  4.0,  6.0,
            8.0,  10.0, 12.0,
            14.0, 16.0, 18.0,
            20.0, 22.0, 24.0,
        ],
    ).unwrap();

    // 2. f64 Backend initialisieren
    let backend = create_backend_f64(Backend::CudaF64);

    // 3. SVD via Jacobi (gesvdj) auf der GPU berechnen
    let (u, sigma, vt) = backend.compute_svd(&a).expect("f64 SVD Berechnung fehlgeschlagen");

    // 4. Dimensions-Validierung
    assert_eq!(u.shape(), &[4, 4], "U-Matrix hat die falsche Dimension");
    assert_eq!(sigma.shape(), &[4, 3], "Sigma-Matrix hat die falsche Dimension");
    assert_eq!(vt.shape(), &[3, 3], "V^T-Matrix hat die falsche Dimension");

    // 5. Mathematische Validierung: Rekonstruktion A = U * Sigma * Vt
    let sigma_vt = sigma.dot(&vt);
    let a_reconstructed = u.dot(&sigma_vt);

    // Da wir f64 nutzen, wählen wir eine entsprechend schärfere Toleranz (Epsilon)
    let epsilon = 1e-12f64;
    for r in 0..a.nrows() {
        for c in 0..a.ncols() {
            let diff = (a[[r, c]] - a_reconstructed[[r, c]]).abs();
            assert!(
                diff < epsilon,
                "Rekonstruktionsfehler bei Index [{}, {}]: Erwartet {}, Erhalten {} (Diff: {})",
                r, c, a[[r, c]], a_reconstructed[[r, c]], diff
            );
        }
    }
    println!("✓ Mathematische Korrektheitsprüfung für CudaF64Svd (Jacobi) erfolgreich bestanden!");
}

#[test]
#[ignore] // Mit 'cargo test --test cuda_f64_test benchmark_cuda_f64_large_matrix --release -- --ignored' ausführen
fn benchmark_cuda_f64_large_matrix() {
    println!("Generiere 10000x10000 Zufallsmatrix (f64) auf der CPU...");
    let start_setup = Instant::now();
    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f64>::random((10000, 10000), dist);
    println!("Matrix-Generierung abgeschlossen in: {:?}", start_setup.elapsed());

    println!("Initialisiere CudaF64Svd Backend...");
    let backend = create_backend_f64(Backend::CudaF64);

    println!("Starte Jacobi-SVD-Berechnung auf der GPU...");
    let start_calc = Instant::now();
    let (u, s, vh) = backend.compute_svd(&a).unwrap();
    let duration = start_calc.elapsed();

    println!("f64 SVD erfolgreich beendet!");
    println!("Ausgabe-Dimensionen:");
    println!("  U Shape:  {:?}", u.shape());
    println!("  S Shape:  {:?}", s.shape());
    println!("  Vh Shape: {:?}", vh.shape());
    println!("Reine Berechnungszeit (GPU + Transfers): {:?}", duration);
}

