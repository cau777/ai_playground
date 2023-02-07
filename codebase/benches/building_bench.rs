use std::time::Duration;
use codebase::nn::batch_config::BatchConfig;
use codebase::nn::key_assigner::KeyAssigner;
use codebase::nn::layers::*;
use codebase::nn::layers::nn_layers::*;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use codebase::utils::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use codebase::chess::decision_tree::{building, building_exp, building_exp_2, DecisionTree};

use criterion::*;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::building_exp_2::{BuilderOptions, LimiterFactors};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::gpu::gpu_data::GpuData;
use codebase::nn::controller::NNController;
use codebase::nn::layers::filtering::convolution;
use codebase::nn::loss::loss_func::LossFunc;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("funcs");
    group.measurement_time(Duration::from_secs(30));

    let controller = NNController::new(Layer::Sequential(sequential_layer::SequentialConfig {
        layers: vec![
            Layer::Convolution(convolution::ConvolutionConfig {
                in_channels: 6,
                stride: 1,
                kernel_size: 3,
                init_mode: convolution::ConvolutionInitMode::HeNormal(),
                out_channels: 2,
                padding: 0,
                lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            }),
            Layer::Flatten,
            Layer::Dense(dense_layer::DenseConfig {
                init_mode: dense_layer::DenseLayerInit::Random(),
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                out_values: 1,
                in_values: 6 * 6 * 2,
            }),
        ],
    }), LossFunc::Mse).unwrap();

    // group.bench_function("normal", |b| b.iter(|| {
    //     let builder = building::DecisionTreesBuilder::new(
    //         vec![DecisionTree::new(true)],
    //         vec![TreeCursor::new(BoardController::new_start())],
    //         building::NextNodeStrategy::BestNodeAlways { min_nodes_explored: 10 },
    //         32,
    //     );
    //     let (_tree1, _) = builder.build(&controller, |_| {});
    // }));

    group.bench_function("optimized", |b| b.iter(|| {
        let builder = building_exp::DecisionTreesBuilder::new(
            vec![DecisionTree::new(true)],
            vec![TreeCursor::new(BoardController::new_start())],
            building_exp::NextNodeStrategy::BestNodeAlways { min_nodes_explored: 30 },
            32, 5_000,
        );
        let (_tree2, _) = builder.build(&controller, |_| {});
    }));

    group.bench_function("optimized2", |b| b.iter(|| {
        let builder = building_exp_2::DecisionTreesBuilder::new(
            vec![DecisionTree::new(true)],
            vec![TreeCursor::new(BoardController::new_start())],
            building_exp_2::BuilderOptions {
                limits: building_exp_2::LimiterFactors {
                    // max_iterations: Some(15),
                    max_explored_nodes: Some(30),
                    ..LimiterFactors::default()
                },
                batch_size: 32,
                max_cache: 5_000,
                ..BuilderOptions::default()
            }
        );
        let (_tree2, _) = builder.build(&controller);
    }));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);