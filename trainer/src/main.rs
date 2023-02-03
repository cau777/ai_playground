extern crate core;

mod env_config;
mod client;
mod digits;
mod files;
mod chess;

use std::sync::Arc;
use codebase::integration::deserialization::{deserialize_storage};
use codebase::integration::layers_loading::{ModelXmlConfig};
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use http::{StatusCode};
use crate::client::ServerClient;
use crate::env_config::EnvConfig;

fn main() {
    let config = &EnvConfig::new();
    if config.profile {
        profile_code();
    } else {
        let client = Arc::new(ServerClient::new(config));

        let name = &config.name;
        let model_config = client.load_model_config(name).unwrap();

        let storage_response = client.get_trainable(name).unwrap();
        let storage = match storage_response.status() {
            StatusCode::NOT_FOUND => init(model_config.clone(), &client, name),
            StatusCode::OK => deserialize_storage(&storage_response.bytes().unwrap()).unwrap(),
            _ => panic!("Invalid response from /trainable"),
        };

        match name.as_str() {
            "digits" => digits::train(storage, model_config, config, &client),
            "chess" => chess::train(storage, model_config, config, client),
            _ => panic!("Invalid model name {}", name),
        }
    }
}

fn init(model_config: ModelXmlConfig, client: &ServerClient, name: &str) -> GenericStorage {
    let controller = NNController::new(model_config.main_layer, model_config.loss_func).unwrap();
    let storage = controller.export();
    client.submit(&storage, 100_000.0, name);
    storage
}

fn profile_code() {
    use codebase::nn::layers::*;
    use codebase::nn::layers::nn_layers::*;
    use codebase::nn::layers::filtering::*;
    use codebase::nn::layers::filtering::convolution::*;
    use codebase::nn::lr_calculators::lr_calculator::*;
    use codebase::nn::lr_calculators::constant_lr::*;
    use codebase::nn::loss::loss_func::*;
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

    let builder = codebase::chess::decision_tree::building_exp::DecisionTreesBuilder::new(
        vec![codebase::chess::decision_tree::DecisionTree::new(true)],
        vec![codebase::chess::decision_tree::cursor::TreeCursor::new(codebase::chess::board_controller::BoardController::new_start())],
        codebase::chess::decision_tree::building_exp::NextNodeStrategy::BestNodeAlways { min_nodes_explored: 5_000 },
        32,
        1_000,
    );
    let (tree, _) = builder.build(&controller, |_| {});
    println!("tree_len = {}", tree[0].len());
}