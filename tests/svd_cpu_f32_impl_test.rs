// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

#![cfg(feature = "cpu")]

use svdwrapper::{create_backend_f32, Backend};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

#[test]
fn test_cpu_f32_svd_correctness() {
    // 1. Verwende eine rechteckige Matrix (M = 4, N = 3) um das Sigma-Layout zu prüfen
    let a = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0,  6.0,  9.0,
            12.0, 15.0, 18.0,
            21.0, 24.0, 27.0,
            30.0, 33.0, 36.0,
        ],
    ).unwrap();

    // 2. CPU Backend initialisieren
    let backend = create_backend_f32(Backend::CpuF32);

    // 3. SVD berechnen
    let (u, sigma, vt) = backend.compute_svd(&a).expect("CPU SVD fehlgeschlagen");

    // 4. Dimensionen validieren (Sigma MUSS 4x3 sein, nicht 3x3)
    assert_eq!(u.shape(), &[4, 4], "U-Matrix hat falsche Dimension");
    assert_eq!(sigma.shape(), &[4, 3], "Sigma-Matrix hat falsche Dimension");
    assert_eq!(vt.shape(), &[3, 3], "V^T-Matrix hat falsche Dimension");

    // 5. Mathematische Validierung via Rekonstruktion: A = U * Sigma * Vt
    let sigma_vt = sigma.dot(&vt);
    let a_reconstructed = u.dot(&sigma_vt);

    // Hohe Präzision dank f32 LAPACK-Unterstützung
    let epsilon = 1e-5f32;
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
    println!("✓ Mathematische Korrektheitsprüfung für CpuSvd erfolgreich bestanden!");
}

#[test]
#[ignore] // Ausführen mit 'cargo test --test svd_cpu_impl_test benchmark_cpu_large_matrix --release -- --ignored'
fn benchmark_cpu_large_matrix() {
    println!("Generiere 10000x10000 Zufallsmatrix (f32) für CPU...");
    let start_setup = Instant::now();
    let dist = Uniform::new(1.0, 10.0).unwrap();
    let a = Array2::<f32>::random((10000, 10000), dist);
    println!("Matrix generiert in: {:?}", start_setup.elapsed());

    let backend = create_backend_f32(Backend::CpuF32);

    println!("Starte SVD auf der CPU...");
    let start_calc = Instant::now();
    let (u, s, vh) = backend.compute_svd(&a).unwrap();
    let duration = start_calc.elapsed();

    println!("CPU SVD erfolgreich beendet!");
    println!("Ausgabe-Dimensionen:");
    println!("  U Shape:  {:?}", u.shape());
    println!("  S Shape:  {:?}", s.shape());
    println!("  Vh Shape: {:?}", vh.shape());
    println!("Berechnungszeit (CPU): {:?}", duration);
}

