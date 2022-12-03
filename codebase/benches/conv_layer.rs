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
use codebase::nn::layers::convolution;

fn criterion_benchmark(c: &mut Criterion) {
    let config = convolution::ConvolutionConfig {
        init_mode: convolution::ConvolutionInitMode::HeNormal(),
        stride: 1,
        kernel_size: 5,
        lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        in_channels: 32,
        out_channels: 64,
        padding: 2,
    };
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut storage = GenericStorage::new();
    convolution::ConvolutionLayer::init(InitData {
        storage: &mut storage,
        assigner: &mut KeyAssigner::new(),
    }, &config).unwrap();
    let gpu = GpuData::new_global().unwrap();


    c.bench_function("conv 24x24~64 forward", |b| b.iter(|| {
        convolution::ConvolutionLayer::forward(ForwardData {
            inputs: Array4F::random((64, 32, 14, 14), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut GenericStorage::new(),
            gpu: Some(gpu.clone()),
        }, &config).unwrap();
    }));

    // c.bench_function("conv 24x24~64 backward", |b| b.iter(|| {
    //     let mut forward_cache = GenericStorage::new();
    //     forward_cache.insert("convolution_32_64_0".to_owned(), vec![Array4F::random((64, 32, 18, 18), &dist).into_dyn()]);
    //
    //     convolution::ConvolutionLayer::backward(BackwardData {
    //         grad: Array4F::random((64, 64, 14, 14), &dist).into_dyn(),
    //         storage: &storage,
    //         batch_config: &BatchConfig::new_not_train(),
    //         assigner: &mut KeyAssigner::new(),
    //         forward_cache: &mut forward_cache,
    //         backward_cache: &mut GenericStorage::new(),
    //         gpu: Some(gpu.clone())
    //     }, &config).unwrap();
    // }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);