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

fn criterion_benchmark(c: &mut Criterion) {
    let config = convolution_layer::ConvolutionConfig {
        init_mode: convolution_layer::ConvolutionInitMode::HeNormal(),
        stride:2,
        kernel_size: 2,
        lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        in_channels: 1,
        out_channels: 16,
        padding: 0,
    };
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut storage = GenericStorage::new();
    convolution_layer::ConvolutionLayer::init(InitData {
        storage: &mut storage,
        assigner: &mut KeyAssigner::new(),
    }, &config).unwrap();

    c.bench_function("conv 24x24~16 forward", |b| b.iter(|| {
        convolution_layer::ConvolutionLayer::forward(ForwardData {
            inputs: Array4F::random((64, 1, 24, 24), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut GenericStorage::new()
        }, &config).unwrap();
    }));

    c.bench_function("conv 24x24~16 backward", |b| b.iter(|| {
        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("convolution_1_16_0".to_owned(), vec![Array4F::random((64, 1, 24, 24), &dist).into_dyn()]);

        convolution_layer::ConvolutionLayer::backward(BackwardData {
            grad: Array4F::random((64, 16, 12, 12), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut forward_cache,
            backward_cache: &mut GenericStorage::new(),
        }, &config).unwrap();
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);