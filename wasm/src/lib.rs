extern crate core;

use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::model_deltas::{export_deltas, import_deltas};
use codebase::integration::proto_loading::{
    load_model_from_bytes, load_pair_from_bytes, save_model_bytes,
};
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::{GenericStorage, Layer, LayerError};
use codebase::nn::loss::loss_func::LossFunc;
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

fn console_log(message: String) {
    log(&message);
}

trait LogUnwrap<T> {
    fn unwrap_log(self) -> T;
}

impl<T, TErr: std::fmt::Display + std::fmt::Debug> LogUnwrap<T> for Result<T, TErr> {
    fn unwrap_log(self) -> T {
        if self.is_err() {
            let e = &self.as_ref().err().unwrap();
            console_log(format!("{}: {:?}", e, e));
        }
        self.unwrap()
    }
}

/*

#[wasm_bindgen]
pub fn process_train(model_data: Param, model_config: Param, train_data: Param) -> Vec<u8> {
    let model_bytes = model_data.unwrap();
    let model_bytes = decompress_default(&model_bytes).unwrap_log();
    let initial = load_model_from_bytes(&model_bytes).unwrap();

    let model_config = model_config.unwrap();
    let model_config = load_model_xml(&model_config).unwrap_log();

    let mut controller = NNController::load(
        model_config.main_layer,
        model_config.loss_func,
        initial.clone(),
    )
    .unwrap_log();

    let inputs = train_data.unwrap();
    let inputs = decompress_default(&inputs).unwrap_log();
    let inputs = load_pair_from_bytes(&inputs).unwrap();
    /*
    for _ in 0..10 {
        for (key, value) in controller.export() {
            for (index, value) in value.iter().enumerate() {
                console_log(format!("{}->{} {:?}", key, index, value.iter().take(15).collect::<Vec<_>>()));
            }
        }

        let train_result = controller
            .train_batch(inputs.inputs.clone(), &inputs.expected)
            .unwrap_log();
        console_log(format!("Result = {}", train_result));
        console_log("-----------------------".to_owned());
    }
    */
    let train_result = controller
        .train_batch(inputs.inputs.clone(), &inputs.expected)
        .unwrap_log();
    console_log(format!("Result = {}", train_result));

    let mut result = controller.export();
    export_deltas(&initial, &mut result);

    let result = save_task_result_bytes(TaskResult::Train(result)).unwrap_log();
    compress_default(&result).unwrap_log()
}

#[wasm_bindgen]
pub fn process_test(
    model_data: Param,
    model_config: Param,
    test_data: Param,
    version: u32,
    batch: u32,
) -> Vec<u8> {
    let model_bytes = model_data.unwrap();
    let model_bytes = decompress_default(&model_bytes).unwrap_log();
    let initial = load_model_from_bytes(&model_bytes).unwrap();

    let model_config = model_config.unwrap();
    let model_config = load_model_xml(&model_config).unwrap_log();

    let controller = NNController::load(
        model_config.main_layer,
        model_config.loss_func,
        initial.clone(),
    )
    .unwrap_log();

    let inputs = test_data.unwrap();
    let inputs = decompress_default(&inputs).unwrap_log();
    let inputs = load_pair_from_bytes(&inputs).unwrap();

    let result = 100.0
        - controller
            .test_batch(inputs.inputs, &inputs.expected)
            .unwrap_log();
    let result = save_task_result_bytes(TaskResult::Test(version, batch, result as f64)).unwrap();
    compress_default(&result).unwrap_log()
}

*/

struct SharedData {
    initial: GenericStorage,
    storage: GenericStorage,
    layer: Layer,
    loss: LossFunc,
    accum_mean_delta: f64,
    accum_versions: u32,
}

impl SharedData {
    fn empty() -> Self {
        Self::new(Layer::Relu, LossFunc::Mse, GenericStorage::new())
    }

    fn new(layer: Layer, loss: LossFunc, storage: GenericStorage) -> Self {
        Self {
            initial: storage.clone(),
            storage,
            layer,
            loss,
            accum_mean_delta: 0.0,
            accum_versions: 0,
        }
    }

    fn build(&self) -> Result<NNController, LayerError> {
        NNController::load(self.layer.clone(), self.loss.clone(), self.storage.clone())
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
        let bytes = save_model_bytes(&current)?;
        let compressed = compress_default(&bytes)?;
        self.accum_mean_delta = 0.0;
        self.accum_versions = 0;
        self.storage.clone_into(&mut self.initial);
        Ok(compressed)
    }
}

static SHARED_DATA: once_cell::sync::Lazy<Arc<RwLock<SharedData>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(SharedData::empty())));

#[wasm_bindgen]
pub fn load_initial(model_data: &[u8], model_config: &[u8]) {
    let model_data = decompress_default(&model_data).unwrap_log();
    let storage = load_model_from_bytes(&model_data).unwrap();
    let model_config = load_model_xml(&model_config).unwrap_log();
    *SHARED_DATA.write().unwrap() =
        SharedData::new(model_config.main_layer, model_config.loss_func, storage);
}

#[wasm_bindgen]
pub fn load_deltas(deltas: &[u8]) {
    let mut shared = SHARED_DATA.write().unwrap_log();

    let deltas = decompress_default(deltas).unwrap_log();
    let deltas = load_model_from_bytes(&deltas).unwrap();
    shared.import(deltas, false)
}

#[wasm_bindgen]
pub fn train(train_data: &[u8]) {
    let inputs = decompress_default(&train_data).unwrap_log();
    let inputs = load_pair_from_bytes(&inputs).unwrap();

    let mut controller = {
        let shared = SHARED_DATA.read().unwrap_log();
        shared.build().unwrap_log()
    };

    let initial = controller.export();

    let train_result = controller
        .train_batch(inputs.inputs.clone(), &inputs.expected)
        .unwrap_log();
    console_log(format!("Result = {}", scale_error(train_result)));

    let mut result = controller.export();
    export_deltas(&initial, &mut result);

    let mut shared = SHARED_DATA.write().unwrap_log();
    shared.import(result, true);
}

#[wasm_bindgen]
pub fn test(test_data: &[u8]) -> f64 {
    let inputs = decompress_default(&test_data).unwrap_log();
    let inputs = load_pair_from_bytes(&inputs).unwrap();

    let controller = {
        let shared = SHARED_DATA.read().unwrap_log();
        shared.build().unwrap_log()
    };

    let result = controller
        .test_batch(inputs.inputs, &inputs.expected)
        .unwrap();
    scale_error(result)
}

fn scale_error(error: f64) -> f64 {
    100.0 - (error as f64) * 5.0
}

#[wasm_bindgen]
pub fn should_push() -> bool {
    let shared = SHARED_DATA.read().unwrap_log();
    console_log(format!(
        "Delta = {} Version = {}",
        shared.accum_mean_delta, shared.accum_versions
    ));
    shared.accum_mean_delta > 0.01 || shared.accum_versions > 50
}

#[wasm_bindgen]
pub fn export_bytes() -> Vec<u8> {
    let mut shared = SHARED_DATA.write().unwrap_log();
    console_log(format!("Exporting"));
    shared.export().unwrap_log()
}
