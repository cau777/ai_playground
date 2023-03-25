use std::time::Duration;
use criterion::*;
use codebase::gpu::buffers::upload_array_to_gpu;
use codebase::gpu::gpu_data::{GpuData};
use codebase::nn::batch_config::BatchConfig;
use codebase::nn::key_assigner::KeyAssigner;
use codebase::nn::layers::filtering::max_pool::*;
use codebase::nn::layers::nn_layers::{ForwardData, LayerOps};
use codebase::nn::layers::stored_array::StoredArray;
use codebase::utils::Array4F;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("main");
    group.measurement_time(Duration::from_secs(15));

    let gpu = GpuData::new_global().unwrap();
    let config = MaxPoolConfig {
        size: 2,
        stride: 2,
        padding: 1,
    };
    const SHAPE: [usize; 4] = [4, 2, 6, 6];
    let inputs = Array4F::ones(SHAPE).into_dyn();
    let inputs = upload_array_to_gpu(&inputs, &gpu).unwrap();
    let inputs = StoredArray::GpuLocal { gpu: gpu.clone(), data: inputs, shape: SHAPE.to_vec() };

    group.bench_function("normal,gpu-buffer", |b| b.iter(|| {
        MaxPoolLayer::forward(ForwardData {
            inputs: inputs.clone(),
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            storage: &Default::default(),
            forward_cache: None,
            prev_iteration_cache: None,
            gpu: Some(gpu.clone()),
        }, &config).unwrap();
    }));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);