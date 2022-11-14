mod env_config;
mod client;

use std::fs::OpenOptions;
use std::io;
use std::io::{Read, Write};
use std::io::ErrorKind::InvalidData;
use codebase::integration::deserialization::{deserialize_pairs, deserialize_storage};
use codebase::integration::layers_loading::{load_model_xml, ModelXmlConfig};
use codebase::integration::serde_utils::Pairs;
use codebase::integration::serialization::{serialize_storage, serialize_version};
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use codebase::nn::train_config::TrainConfig;
use http::{Method, Request, StatusCode};
use rand::thread_rng;
use crate::client::ServerClient;
use crate::env_config::EnvConfig;

const NAME: &str = "digits";

fn main() {
    let config = &EnvConfig::new();
    let client = &ServerClient::new(config);

    let model_config = client.load_model_config().unwrap();

    let storage_response = client.get("trainable").unwrap();
    let storage = match storage_response.status() {
        StatusCode::NOT_FOUND => init(model_config.clone(), client),
        StatusCode::OK => deserialize_storage(&storage_response.bytes().unwrap()).unwrap(),
        _ => panic!("Invalid response from /trainable"),
    };

    train(storage, model_config, config, client);
}

fn init(model_config: ModelXmlConfig, client: &ServerClient) -> GenericStorage {
    let controller = NNController::new(model_config.main_layer, model_config.loss_func).unwrap();
    let storage = controller.export();
    client.submit(&storage, 100_000.0);
    storage
}

fn train(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig, client: &ServerClient) {
    let mut controller = NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap();
    let data = load_data("train", config).unwrap();
    let mut rng = thread_rng();

    for version in 0..10 {
        let mut total_loss=0.0;

        for epoch in 0..10 {
            let data = data.pick_rand(128, &mut rng); // TODO: param
            let loss=controller.train_batch(data.inputs, &data.expected, TrainConfig::default()).unwrap();
            total_loss += loss;
            println!("{} -> loss={}", epoch, loss);
        }

        let avg_loss = total_loss / config.max_epochs as f64;
        client.submit(&controller.export(), avg_loss);
    }
}

fn load_data(filename: &str, config: &EnvConfig) -> io::Result<Pairs> {
    let mut file = OpenOptions::new().read(true)
        .open(format!("{}/{}/{}.dat", config.mounted_path, NAME, filename))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    deserialize_pairs(&buffer).map_err(|e| io::Error::new(InvalidData, e))
}