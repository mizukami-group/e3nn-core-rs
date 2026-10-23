#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use e3nn_core::{CgEntry, Irreps, TensorProductInstruction, TensorProductPlan};

const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;

fn e3nn_tensor_product_smoke_plan() -> TensorProductPlan {
    let irreps_in1 = "2x0e + 1x1o".parse::<Irreps>().unwrap();
    let irreps_in2 = "1x0e + 1x1o".parse::<Irreps>().unwrap();
    let irreps_out = "2x0e + 1x1o".parse::<Irreps>().unwrap();
    let in1_segment = [0, 0, 1, 1, 0, 1, 2, 2, 2];
    let in2_segment = [0, 0, 0, 0, 1, 1, 0, 1, 1];
    let out_segment = [0, 1, 0, 1, 2, 2, 2, 0, 1];
    let cg_offset = [0, 0, 0, 0, 1, 1, 4, 7, 7];
    let cg_len = [1, 1, 1, 1, 3, 3, 3, 3, 3];
    let path_weight = [
        INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, 1.0, 1.0, 1.0, INV_SQRT_3, INV_SQRT_3,
    ];
    let instructions = (0..9)
        .map(|index| TensorProductInstruction {
            in1_segment: in1_segment[index],
            in2_segment: in2_segment[index],
            out_segment: out_segment[index],
            cg_offset: cg_offset[index],
            cg_len: cg_len[index],
            weight_offset: Some(index),
            path_weight: path_weight[index],
        })
        .collect();
    let cg_entries = vec![
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: 1.0,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 1,
            k_m: 1,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 2,
            k_m: 2,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 1,
            j_m: 0,
            k_m: 1,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 2,
            j_m: 0,
            k_m: 2,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 1,
            j_m: 1,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 2,
            j_m: 2,
            k_m: 0,
            value: INV_SQRT_3,
        },
    ];
    TensorProductPlan::new(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        cg_entries,
        9,
    )
    .unwrap()
}

fn inputs(count: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|row| {
            (0..dim)
                .map(|col| ((row + 1) as f64 * (col + 2) as f64 * 0.013).sin())
                .collect()
        })
        .collect()
}

fn bench_tensor_product(c: &mut Criterion) {
    let plan = e3nn_tensor_product_smoke_plan();
    let mut group = c.benchmark_group("tensor_product");
    for count in [100_usize, 1_000, 10_000] {
        let x = inputs(count, plan.input1_dim());
        let y = inputs(count, plan.input2_dim());
        let w = inputs(count, plan.weight_numel);
        let grad_out = inputs(count, plan.output_dim());

        group.bench_with_input(BenchmarkId::new("forward", count), &count, |b, _| {
            let mut out = vec![0.0; plan.output_dim()];
            b.iter(|| {
                let mut total = 0.0;
                for index in 0..count {
                    plan.forward(&x[index], &y[index], Some(&w[index]), &mut out)
                        .unwrap();
                    total += out.iter().sum::<f64>();
                }
                black_box(total)
            })
        });
        group.bench_with_input(BenchmarkId::new("forward_vjp", count), &count, |b, _| {
            let mut out = vec![0.0; plan.output_dim()];
            let mut grad_x = vec![0.0; plan.input1_dim()];
            let mut grad_y = vec![0.0; plan.input2_dim()];
            let mut grad_w = vec![0.0; plan.weight_numel];
            b.iter(|| {
                let mut total = 0.0;
                for index in 0..count {
                    plan.forward(&x[index], &y[index], Some(&w[index]), &mut out)
                        .unwrap();
                    total += out.iter().sum::<f64>();
                    grad_x.fill(0.0);
                    grad_y.fill(0.0);
                    grad_w.fill(0.0);
                    plan.vjp(
                        &x[index],
                        &y[index],
                        Some(&w[index]),
                        &grad_out[index],
                        &mut grad_x,
                        &mut grad_y,
                        Some(&mut grad_w),
                    )
                    .unwrap();
                    total += grad_x.iter().chain(&grad_y).chain(&grad_w).sum::<f64>();
                }
                black_box(total)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tensor_product);
criterion_main!(benches);
