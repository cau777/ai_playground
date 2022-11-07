extern crate core;

use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::model_deltas;
use codebase::integration::serialization::*;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use codebase::nn::train_config::TrainConfig;
use codebase::utils::Array3F;
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["bindings"])]
    fn insertLog(message: String, level: u32);
}

fn log(message: String, level: u32) {
    insertLog(message, level)
}

trait LogUnwrap<T> {
    fn unwrap_log(self) -> T;
}

impl<T, TErr: std::fmt::Display + std::fmt::Debug> LogUnwrap<T> for Result<T, TErr> {
    fn unwrap_log(self) -> T {
        if self.is_err() {
            let e = &self.as_ref().err().unwrap();
            log(format!("{}: {:?}", e, e), 2);
        }
        self.unwrap()
    }
}

struct SharedStorage {
    initial: GenericStorage,
    storage: GenericStorage,
    accum_mean_delta: f64,
    accum_versions: u32,
}

impl SharedStorage {
    fn empty() -> Self {
        Self {
            initial: GenericStorage::new(),
            storage: GenericStorage::new(),
            accum_mean_delta: 0.0,
            accum_versions: 0,
        }
    }
    
    fn set(&mut self, storage: &GenericStorage) {
        self.initial = storage.clone();
        self.storage = storage.clone();
        self.accum_mean_delta = 0.0;
        self.accum_versions = 0;
    }

    fn import(&mut self, deltas: GenericStorage, internal_input: bool) {
        if internal_input {
            let mean_delta = deltas
                .iter()
                .flat_map(|(_, value)| value)
                .map(|o| o.iter().map(|o| o.abs() as f64).sum::<f64>() / o.len() as f64)
                .sum::<f64>()
                / deltas.len() as f64;
            self.accum_mean_delta += mean_delta;
            self.accum_versions += 1;
        }

        model_deltas::import_deltas(&mut self.storage, deltas);
    }

    fn export_deltas(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut current = self.storage.clone();
        model_deltas::export_deltas(&self.initial, &mut current);
        let bytes = serialize_storage(&current);

        self.accum_mean_delta = 0.0;
        self.accum_versions = 0;
        self.initial = self.storage.clone();
        // self.storage.clone_into(&mut self.initial);

        Ok(bytes)
    }
}

static SHARED_STORAGE: once_cell::sync::Lazy<Arc<RwLock<SharedStorage>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(SharedStorage::empty())));

#[wasm_bindgen]
pub fn import_server_storage(model_data: &[u8]) {
    let storage = deserialize_storage(model_data).unwrap_log();
    SHARED_STORAGE.write().unwrap().set(&storage);
}

#[wasm_bindgen]
pub fn import_local_deltas(deltas: &[u8]) {
    let deltas = deserialize_storage(deltas).unwrap_log();
    let mut shared = SHARED_STORAGE.write().unwrap_log();
    shared.import(deltas, true);
}

#[wasm_bindgen]
pub fn should_push() -> bool {
    let shared = SHARED_STORAGE.read().unwrap_log();
    let result = shared.accum_mean_delta > 0.01 || shared.accum_versions > 50;
    result
}

#[wasm_bindgen]
pub fn export_current() -> Vec<u8> {
    let shared = SHARED_STORAGE.read().unwrap_log();
    serialize_storage(&shared.storage)
}

#[wasm_bindgen]
pub fn export_deltas() -> Vec<u8> {
    let mut shared = SHARED_STORAGE.write().unwrap_log();
    shared.export_deltas().unwrap_log()
}

#[wasm_bindgen]
pub fn prepare_digits(pixels: &[f32]) -> Vec<u8> {
    let inputs = Array3F::from_shape_vec(
        (1, 28, 28),
        pixels.iter().copied().map(|o| o / 255.0).collect(),
    ).unwrap();
    serialize_array(&inputs.into_dyn())
}

// ------------------
// Web workers
// ------------------

#[wasm_bindgen]
pub fn train_job(config: &[u8], storage: &[u8], pairs: &[u8], workers: u32) -> Vec<u8> {
    let config = load_model_xml(config).unwrap_log();
    let storage = deserialize_storage(&storage).unwrap_log();
    let mut controller = NNController::load(config.main_layer, config.loss_func, storage).unwrap();
    let pairs = deserialize_pairs(&pairs).unwrap();
    let initial = controller.export();
    let train_result = controller
        .train_batch(pairs.inputs.clone(), &pairs.expected, TrainConfig{workers})
        .unwrap_log();
    log(format!("Train result = {}", scale_error(train_result)), 0);

    let mut result = controller.export();
    model_deltas::export_deltas(&initial, &mut result);
    serialize_storage(&result)
}

#[wasm_bindgen]
pub fn validate_job(config: &[u8], storage: &[u8], pairs: &[u8]) -> f64 {
    let config = load_model_xml(&config).unwrap_log();
    let storage = deserialize_storage(&storage).unwrap_log();
    let controller = NNController::load(config.main_layer, config.loss_func, storage).unwrap();
    let pairs = deserialize_pairs(&pairs).unwrap();
    let result = controller
        .test_batch(pairs.inputs, &pairs.expected)
        .unwrap();
    let result = scale_error(result);
    log(format!("Test result = {}", result), 0);
    result
}

#[wasm_bindgen]
pub fn eval_job(config: &[u8], storage: &[u8], inputs: &[u8]) -> Vec<f32> {
    let config = load_model_xml(&config).unwrap_log();
    let storage = deserialize_storage(&storage).unwrap_log();
    let controller = NNController::load(config.main_layer, config.loss_func, storage).unwrap();
    let inputs = deserialize_array(inputs).unwrap();
    let result = controller.eval_batch(inputs).unwrap();
    result.into_iter().collect()
}

fn scale_error(error: f64) -> f64 {
    100.0 - (error as f64) * 5.0
}
