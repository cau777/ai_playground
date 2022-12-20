mod env_config;
mod client;
mod digits;
mod files;
mod chess;

use codebase::integration::deserialization::{deserialize_storage};
use codebase::integration::layers_loading::{ModelXmlConfig};
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use http::{StatusCode};
use crate::client::ServerClient;
use crate::env_config::EnvConfig;

fn main() {
    let config = &EnvConfig::new();
    let client = &ServerClient::new(config);

    let name = &config.name;
    let model_config = client.load_model_config(name).unwrap();

    let storage_response = client.get_trainable(name).unwrap();
    let storage = match storage_response.status() {
        StatusCode::NOT_FOUND => init(model_config.clone(), client, name),
        StatusCode::OK => deserialize_storage(&storage_response.bytes().unwrap()).unwrap(),
        _ => panic!("Invalid response from /trainable"),
    };

    match name.as_str() {
        "digits" => digits::train(storage, model_config, config, client),
        "chess" => chess::train(storage, model_config, config, client),
        _ => panic!("Invalid model name {}", name),
    }
}

fn init(model_config: ModelXmlConfig, client: &ServerClient, name: &str) -> GenericStorage {
    let controller = NNController::new(model_config.main_layer, model_config.loss_func).unwrap();
    let storage = controller.export();
    client.submit(&storage, 100_000.0, name);
    storage
}
