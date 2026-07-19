// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cuda")]

use svdwrapper::{create_backend_f32, Backend}; // Passe den Cratename ggf. an deine Cargo.toml an
use ndarray::{Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

#[test]
fn test_cuda_f32_svd_correctness() {
    // 1. Erstelle eine definierte, nicht-quadratische Testmatrix (M = 4, N = 3)
    // Das testet gleichzeitig, ob das Min(M,N) Handling fehlerfrei läuft.
    let a = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ],
    ).unwrap();

    // 2. Backend initialisieren
    let backend = create_backend_f32(Backend::CudaF32);

    // 3. SVD auf der GPU berechnen
    let (u, sigma, vt) = backend.compute_svd(&a).expect("SVD Berechnung fehlgeschlagen");

    // 4. Assertions auf die korrekten Matrix-Dimensionen
    assert_eq!(u.shape(), &[4, 4], "U-Matrix hat die falsche Dimension");
    assert_eq!(sigma.shape(), &[4, 3], "Sigma-Matrix hat die falsche Dimension");
    assert_eq!(vt.shape(), &[3, 3], "V^T-Matrix hat die falsche Dimension");

    // 5. Mathematische Validierung: Rekonstruktion A_reconstructed = U * Sigma * Vt
    let sigma_vt = sigma.dot(&vt);
    let a_reconstructed = u.dot(&sigma_vt);

    // Vergleiche alle Elemente mit einer kleinen Toleranz (Epsilon) aufgrund von Floating-Point-Ungenauigkeiten
    let epsilon = 1e-4f32;
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
    println!("✓ Mathematische Korrektheitsprüfung für CudaF32Svd erfolgreich bestanden!");
}

#[test]
#[ignore] // Mit 'cargo test -- --ignored' ausführen, da dieser Test sehr lange dauert
fn benchmark_cuda_f32_large_matrix() {
    println!("Generiere 10000x10000 Zufallsmatrix auf der CPU...");
    let start_setup = Instant::now();
    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f32>::random((10000, 10000), dist);
    println!("Matrix-Generierung abgeschlossen in: {:?}", start_setup.elapsed());

    println!("Initialisiere CudaF32Svd Backend...");
    let backend = create_backend_f32(Backend::CudaF32);

    println!("Starte SVD-Berechnung auf der GPU...");
    let start_calc = Instant::now();
    let (u, s, vh) = backend.compute_svd(&a).unwrap();
    let duration = start_calc.elapsed();

    // Verhindert das Einfrieren deines Terminals: Zeige nur Metadaten
    println!("f32 SVD erfolgreich beendet!");
    println!("Ausgabe-Dimensionen:");
    println!("  U Shape:  {:?}", u.shape());
    println!("  S Shape:  {:?}", s.shape());
    println!("  Vh Shape: {:?}", vh.shape());
    println!("Reine Berechnungszeit (GPU + Transfers): {:?}", duration);
}

