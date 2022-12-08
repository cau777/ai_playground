mod env_config;
mod client;

use std::fs::OpenOptions;
use std::io;
use std::io::{Read};
use std::io::ErrorKind::InvalidData;
use codebase::integration::deserialization::{deserialize_pairs, deserialize_storage};
use codebase::integration::layers_loading::{ModelXmlConfig};
use codebase::integration::serde_utils::Pairs;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use http::{StatusCode};
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
    let train_data = load_data("train", config).unwrap();
    let validate_data = load_data("validate", config).unwrap();
    let mut rng = thread_rng();

    for version in 0..config.versions {
        let mut total_loss = 0.0;

        println!("Start {}", version + 1);
        for epoch in 0..config.epochs_per_version {
            let data = train_data.pick_rand(128, &mut rng); // TODO: param
            let loss = controller.train_batch(data.inputs, &data.expected).unwrap();
            total_loss += loss;

            if epoch % 16 == 0 {
                println!("    {} -> loss={}", epoch, loss);
            }
        }

        let avg_loss = total_loss / config.epochs_per_version as f64;
        let tested_loss = validate(&validate_data, &controller);

        println!("    Finished with avg_loss={} and tested_loss={}", avg_loss, tested_loss);
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

fn validate(data: &Pairs, controller: &NNController) -> f64 {
    println!("Started testing");

    let mut total = 0.0;
    let mut count = 0;
    for batch in data.chunks_iter(256) {
        total += controller.test_batch(batch.0.into_owned(), &batch.1.into_owned()).unwrap();
        count += 1;
    }
    total / (count as f64)
}