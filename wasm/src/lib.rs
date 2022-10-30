extern crate core;

use codebase::integration::joining::{join_as_bytes, split_bytes_3};
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::model_deltas::{export_deltas, import_deltas};
use codebase::integration::serialization::*;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use codebase::utils::{Array2F, Array3F};
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
        Self::new(GenericStorage::new())
    }

    fn new(storage: GenericStorage) -> Self {
        Self {
            initial: storage.clone(),
            storage,
            accum_mean_delta: 0.0,
            accum_versions: 0,
        }
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

        import_deltas(&mut self.storage, deltas);
    }

    fn export(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut current = self.storage.clone();
        export_deltas(&self.initial, &mut current);
        let bytes = serialize_storage(&current);

        self.accum_mean_delta = 0.0;
        self.accum_versions = 0;
        self.storage.clone_into(&mut self.initial);

        Ok(bytes)
    }
}

static SHARED_STORAGE: once_cell::sync::Lazy<Arc<RwLock<SharedStorage>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(SharedStorage::empty())));

static SHARED_CONFIG: once_cell::sync::Lazy<Arc<RwLock<Vec<u8>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(Vec::new())));

#[wasm_bindgen]
pub fn load_config(config: &[u8]) {
    let mut shared = SHARED_CONFIG.write().unwrap();
    config.clone_into(&mut shared);
}

#[wasm_bindgen]
pub fn load_initial(model_data: &[u8]) {
    let storage = deserialize_storage(model_data).unwrap_log();
    *SHARED_STORAGE.write().unwrap() = SharedStorage::new(storage);
}

#[wasm_bindgen]
pub fn load_server_deltas(deltas: &[u8]) {
    let deltas = deserialize_storage(deltas).unwrap_log();
    let mut shared = SHARED_STORAGE.write().unwrap_log();
    shared.import(deltas, false)
}

#[wasm_bindgen]
pub fn load_train_deltas(deltas: &[u8]) {
    let deltas = deserialize_storage(deltas).unwrap_log();
    let mut shared = SHARED_STORAGE.write().unwrap_log();
    shared.import(deltas, true);
}

#[wasm_bindgen]
pub fn prepare_train_job(pairs: &[u8]) -> Vec<u8> {
    let shared = SHARED_STORAGE.read().unwrap();
    let config = &SHARED_CONFIG.read().unwrap();
    let storage = &serialize_storage(&shared.storage);
    join_as_bytes(&[config, storage, pairs])
}

#[wasm_bindgen]
pub fn prepare_validate_job(pairs: &[u8], storage: &[u8]) -> Vec<u8> {
    let config = &SHARED_CONFIG.read().unwrap();
    // log(format!("{:?}", deserialize_storage(storage)), 0);
    join_as_bytes(&[config, storage, pairs])
}

#[wasm_bindgen]
pub fn prepare_eval_job(data: &[f32], storage: &[u8]) -> Vec<u8> {
    let inputs = Array3F::from_shape_vec(
        (1, 28, 28),
        data.iter().copied().map(|o| o / 255.0).collect(),
    ).unwrap();
    let inputs = &serialize_array(&inputs.into_dyn());
    let config = &SHARED_CONFIG.read().unwrap();
    join_as_bytes(&[config, storage, inputs])
}

#[wasm_bindgen]
pub fn train_job(data: &[u8]) -> Vec<u8> {
    let [config, storage, pairs] = split_bytes_3(data).unwrap();

    let config = load_model_xml(&config).unwrap_log();
    let storage = deserialize_storage(&storage).unwrap_log();
    let mut controller = NNController::load(config.main_layer, config.loss_func, storage).unwrap();
    let pairs = deserialize_pairs(&pairs).unwrap();
    let initial = controller.export();
    let train_result = controller
        .train_batch(pairs.inputs.clone(), &pairs.expected)
        .unwrap_log();
    log(format!("Train result = {}", scale_error(train_result)), 0);

    let mut result = controller.export();
    export_deltas(&initial, &mut result);
    serialize_storage(&result)
}

#[wasm_bindgen]
pub fn validate_job(data: &[u8]) -> f64 {
    let [config, storage, pairs] = split_bytes_3(data).unwrap();

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
pub fn eval_job(data: &[u8]) -> Vec<f32> {
    let [config, storage, inputs] = split_bytes_3(data).unwrap();
    
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

#[wasm_bindgen]
pub fn should_push() -> bool {
    let shared = SHARED_STORAGE.read().unwrap_log();
    let result = shared.accum_mean_delta > 0.01 || shared.accum_versions > 50;

    log(
        format!(
            "Should push query: delta = {}, versions = {} => {}",
            shared.accum_mean_delta, shared.accum_versions, result
        ),
        0,
    );

    result
}

#[wasm_bindgen]
pub fn export_bytes() -> Vec<u8> {
    let mut shared = SHARED_STORAGE.write().unwrap_log();
    log("Exporting".to_owned(), 0);
    shared.export().unwrap_log()
}
