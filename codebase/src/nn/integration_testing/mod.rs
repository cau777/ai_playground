use std::iter::zip;
use crate::chess::board_controller::BoardController;
use crate::chess::decision_tree::{building, building_exp};
use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::DecisionTree;
use crate::nn::controller::NNController;
use crate::nn::layers::nn_layers::Layer;
use crate::nn::layers::*;
use crate::nn::layers::filtering::*;
use crate::nn::loss::loss_func::LossFunc;
use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
use crate::nn::lr_calculators::lr_calculator::LrCalc;

#[test]
fn test_tree_building() {
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

    let nodes = 100;
    let builder = building::DecisionTreesBuilder::new(
        vec![DecisionTree::new(true)],
        vec![TreeCursor::new(BoardController::new_start())],
        building::NextNodeStrategy::BestNodeAlways { min_nodes_explored: nodes },
        20,
    );
    let (tree1, _) = builder.build(&controller, |_| {});

    let builder = building_exp::DecisionTreesBuilder::new(
        vec![DecisionTree::new(true)],
        vec![TreeCursor::new(BoardController::new_start())],
        building_exp::NextNodeStrategy::BestNodeAlways { min_nodes_explored: nodes },
        20,
    );
    let (tree2, _) = builder.build(&controller, |_| {});

    assert!(zip(tree1[0].nodes().map(|o| o.eval).collect::<Vec<_>>(),
                tree2[0].nodes().map(|o| o.eval).collect::<Vec<_>>())
        .all(|(a, b)| (a - b).abs() < 0.0001));
}

#[test]
fn test_tree_building_cached() {}