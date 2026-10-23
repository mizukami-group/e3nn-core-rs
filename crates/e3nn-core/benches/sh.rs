#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use e3nn_core::SphericalHarmonicsPlan;

fn vectors(count: usize) -> Vec<[f64; 3]> {
    (0..count)
        .map(|index| {
            let t = 0.2 + index as f64 * 0.0007;
            [t.sin() + 0.3, t.cos() - 0.1, 0.5 + 0.25 * t]
        })
        .collect()
}

fn bench_spherical_harmonics(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical_harmonics");
    for l_max in [1_usize, 2] {
        let plan = SphericalHarmonicsPlan::component(l_max, true).unwrap();
        for count in [100_usize, 1_000, 10_000] {
            let vectors = vectors(count);
            group.bench_with_input(
                BenchmarkId::new(format!("l{l_max}_forward"), count),
                &vectors,
                |b, vectors| {
                    let mut out = vec![0.0; plan.output_dim()];
                    b.iter(|| {
                        let mut total = 0.0;
                        for r in vectors {
                            plan.forward(*r, &mut out).unwrap();
                            total += out.iter().sum::<f64>();
                        }
                        black_box(total)
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("l{l_max}_forward_vjp"), count),
                &vectors,
                |b, vectors| {
                    let mut out = vec![0.0; plan.output_dim()];
                    let grad_y = vec![0.25; plan.output_dim()];
                    b.iter(|| {
                        let mut total = 0.0;
                        for r in vectors {
                            plan.forward(*r, &mut out).unwrap();
                            let mut grad_r = [0.0; 3];
                            plan.vjp_edge_vector(*r, &grad_y, &mut grad_r).unwrap();
                            total += out.iter().sum::<f64>() + grad_r.iter().sum::<f64>();
                        }
                        black_box(total)
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_spherical_harmonics);
criterion_main!(benches);
