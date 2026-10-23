#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use e3nn_core::radial::{
    cosine_cutoff_forward, cutoff_gaussian_radial_forward, cutoff_gaussian_radial_vjp,
    distance_forward, distance_vjp, gaussian_radial_basis_forward,
};

const EPS: f64 = 1.0e-12;

fn vectors(count: usize) -> Vec<[f64; 3]> {
    (0..count)
        .map(|index| {
            let x = 0.5 + index as f64 * 0.001;
            [x, 0.25 * x.sin(), 0.5 * x.cos()]
        })
        .collect()
}

fn bench_distance_radial(c: &mut Criterion) {
    let centers = [0.5, 1.0, 1.5, 2.0];
    let gamma = 0.8;
    let cutoff = 3.5;
    let mut group = c.benchmark_group("distance_radial_cutoff");

    for count in [100_usize, 1_000, 10_000, 100_000] {
        let vectors = vectors(count);
        group.bench_with_input(
            BenchmarkId::new("forward", count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let mut total = 0.0;
                    let mut basis = [0.0; 4];
                    for r in vectors {
                        let distance = distance_forward(*r, EPS).unwrap().distance;
                        let cutoff_out = cosine_cutoff_forward(distance, cutoff).unwrap();
                        gaussian_radial_basis_forward(distance, &centers, gamma, &mut basis)
                            .unwrap();
                        total += cutoff_out.value
                            + basis
                                .iter()
                                .map(|value| cutoff_out.value * value)
                                .sum::<f64>();
                    }
                    black_box(total)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("forward_vjp", count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let mut total = 0.0;
                    let mut basis = [0.0; 4];
                    let mut weighted = [0.0; 4];
                    let bar = [0.1, -0.2, 0.3, -0.4];
                    for r in vectors {
                        let distance = distance_forward(*r, EPS).unwrap().distance;
                        cutoff_gaussian_radial_forward(
                            distance,
                            cutoff,
                            &centers,
                            gamma,
                            &mut weighted,
                        )
                        .unwrap();
                        gaussian_radial_basis_forward(distance, &centers, gamma, &mut basis)
                            .unwrap();
                        let bar_d = cutoff_gaussian_radial_vjp(
                            distance, cutoff, &centers, gamma, &basis, &bar,
                        )
                        .unwrap();
                        let bar_r = distance_vjp(*r, bar_d, EPS).unwrap();
                        total += weighted.iter().sum::<f64>() + bar_r.iter().sum::<f64>();
                    }
                    black_box(total)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_distance_radial);
criterion_main!(benches);
