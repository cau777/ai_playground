use codebase::nn::batch_config::BatchConfig;
use codebase::nn::key_assigner::KeyAssigner;
use codebase::nn::layers::*;
use codebase::nn::layers::nn_layers::*;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use codebase::utils::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use criterion::*;
use codebase::gpu::shader_runner::GpuData;

fn criterion_benchmark(c: &mut Criterion) {
    let config = dense_layer::DenseConfig {
        in_values: 256,
        out_values: 256,
        init_mode: dense_layer::DenseLayerInit::Random(),
        weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
    };
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut storage = GenericStorage::new();
    dense_layer::DenseLayer::init(InitData {
        storage: &mut storage,
        assigner: &mut KeyAssigner::new(),
    }, &config).unwrap();
    let gpu = GpuData::new_global().unwrap();

    c.bench_function("dense 256x256 forward", |b| b.iter(|| {
        dense_layer::DenseLayer::forward(ForwardData {
            inputs: Array2F::random((64, 256), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut GenericStorage::new()
        }, &config).unwrap();
    }));

    c.bench_function("dense 256x256 backward", |b| b.iter(|| {
        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("dense_256_256_0".to_owned(), vec![Array2F::random((64, 256), &dist).into_dyn()]);

        dense_layer::DenseLayer::backward(BackwardData {
            grad: Array2F::random((64, 256), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut forward_cache,
            backward_cache: &mut GenericStorage::new(),
            gpu: Some(gpu.clone())
        }, &config).unwrap();
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);