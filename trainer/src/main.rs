extern crate core;

mod env_config;
mod client;
mod digits;
mod files;
mod chess;

use std::fs::OpenOptions;
use std::io::Read;
use std::sync::Arc;
use codebase::chess::decision_tree::building::{BuilderOptions, DecisionTreesBuilder, LimiterFactors};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::integration::deserialization::{deserialize_storage};
use codebase::integration::layers_loading::{load_model_xml, ModelXmlConfig};
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use http::{StatusCode};
use crate::client::ServerClient;
use crate::env_config::EnvConfig;

// #[cfg(feature = "dhat-heap")]
// #[global_allocator]
// static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    // #[cfg(feature = "dhat-heap")]
    // let _profiler = dhat::Profiler::builder().trim_backtraces(Some(50)).build();
    // profile_code();
    // return;

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
    use codebase::nn::loss::loss_func::*;

    let mut file = OpenOptions::new().read(true).open(r"C:\Users\caua_\Projects\ai_playground\trainer\profile_config.xml").unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();

    let controller = NNController::new(load_model_xml(
        &bytes
    ).unwrap().main_layer, LossFunc::Mse).unwrap();

    let builder = DecisionTreesBuilder::new(
        vec![
            DecisionTree::new(true),
        ],
        vec![
            TreeCursor::new(codebase::chess::board_controller::BoardController::new_start()),
        ],
        BuilderOptions {
            limits: LimiterFactors {
                max_iterations: Some(200),
                ..Default::default()
            },
            next_node_strategy: codebase::chess::decision_tree::building::NextNodeStrategy::Computed {
                eval_delta_exp: 6.0,
                depth_delta_exp: 0.002,
            },
            batch_size: 64,
            max_cache_bytes: 10_000,
            add_random_to_openings: false,
            random_node_chance: 0.0,
            ..Default::default()
        }
    );
    let (tree, _) = builder.build(&controller);
    println!("tree_len = {}", tree[0].len());
//     println!("After building");
//     std::io::stdin().read_line(&mut String::new()).unwrap();
//
//     drop(builder);
//     drop(tree);
//     use codebase::gpu::gpu_data::GpuData;
//     let NNController {cached_gpu, storage,..} = controller;
//     let gpu = Arc::try_unwrap(Arc::try_unwrap(cached_gpu).ok().unwrap().into_inner().unwrap().unwrap().unwrap()).ok().unwrap();
//     // println!("storage len {}", storage.len());
//
//     gpu.reset_fast_mem_alloc();
//     let GpuData {device, queue, descriptor_alloc, cmd_alloc, cache, std_mem_alloc, fast_mem_alloc, pools, contexts}
// =gpu;

    //     = Arc::try_unwrap(Arc::try_unwrap(cached_gpu).ok().unwrap().into_inner().unwrap().unwrap().unwrap()).ok().unwrap();
    //
    //
    // drop(storage);
    // drop(pools);
    // drop(cmd_alloc);
    // drop(descriptor_alloc);
    // drop(std_mem_alloc);
    // drop(queue);
    // drop(device);
    // drop(contexts);
    // drop(cache);
    // println!("first dropped");
    // std::io::stdin().read_line(&mut String::new()).unwrap();
    //
    // drop(fast_mem_alloc);
    // println!("fast_mem_alloc dropped");
    // std::io::stdin().read_line(&mut String::new()).unwrap();
    //
    // println!("second dropped");
    // std::io::stdin().read_line(&mut String::new()).unwrap();

}