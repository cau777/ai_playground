use std::time::Duration;
use codebase::nn::layers::*;
use codebase::nn::layers::nn_layers::*;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use codebase::chess::decision_tree::{building, DecisionTree};
use criterion::*;
use codebase::chess::board_controller::GameController;
use codebase::chess::decision_tree::building::{BuilderOptions, LimiterFactors};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::nn::controller::NNController;
use codebase::nn::layers::filtering::convolution;
use codebase::nn::loss::loss_func::LossFunc;

fn build_controller(cache: bool) -> NNController {
    NNController::new(Layer::Sequential(sequential_layer::SequentialConfig {
        layers: vec![
            Layer::Convolution(convolution::ConvolutionConfig {
                in_channels: 6,
                stride: 1,
                kernel_size: 3,
                init_mode: convolution::ConvolutionInitMode::HeNormal(),
                out_channels: 2,
                padding: 0,
                lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                cache,
            }),
            Layer::Relu,
            Layer::MaxPool(filtering::max_pool::MaxPoolConfig{
                stride:2,
                padding: 1,
                size: 2,
            }),
            Layer::Flatten,
            Layer::Dense(dense_layer::DenseConfig {
                init_mode: dense_layer::DenseLayerInit::Random(),
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                out_values: 1,
                in_values: 4 * 4 * 2,
            }),
        ],
    }), LossFunc::Mse).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("funcs");
    group.measurement_time(Duration::from_secs(60*5));
    {
        let controller = build_controller(true);

        group.bench_function("optimized2", |b| b.iter(|| {
            let builder = building::DecisionTreesBuilder::new(
                vec![DecisionTree::new(true)],
                vec![TreeCursor::new(GameController::new_start())],
                BuilderOptions {
                    limits: LimiterFactors {
                        max_iterations: Some(10),
                        ..LimiterFactors::default()
                    },
                    batch_size: 32,
                    max_cache_bytes: 5_000,
                    add_random_to_openings: false,
                    random_node_chance: 0.0,
                    ..BuilderOptions::default()
                },
            );
            let (_tree2, _) = builder.build(&controller);
        }));
    }


    {
        let controller = build_controller(false);

        group.bench_function("optimized2-nocache", |b| b.iter(|| {
            let builder = building::DecisionTreesBuilder::new(
                vec![DecisionTree::new(true)],
                vec![TreeCursor::new(GameController::new_start())],
                BuilderOptions {
                    limits: LimiterFactors {
                        max_iterations: Some(10),
                        ..LimiterFactors::default()
                    },
                    batch_size: 32,
                    max_cache_bytes: 5_000,
                    add_random_to_openings: false,
                    ..BuilderOptions::default()
                },
            );
            let (_tree2, _) = builder.build(&controller);
        }));
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);