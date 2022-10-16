extern crate core;

use std::ops::SubAssign;
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::proto_loading::{load_model_from_bytes, load_pair_from_bytes, save_model_bytes, save_task_result_bytes, TaskResult};
use codebase::nn::controller::NNController;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

type Param = Option<Vec<u8>>;

fn console_log(message: String) {
    log(&message);
}

trait LogUnwrap<T> {
    fn unwrap_log(self) -> T;
}

impl<T, TErr: std::fmt::Display+std::fmt::Debug> LogUnwrap<T> for Result<T, TErr> {
    fn unwrap_log(self) -> T {
        if self.is_err() {
            console_log(format!("{}", &self.as_ref().err().unwrap()));
        }
        self.unwrap()
    }
}

#[wasm_bindgen]
pub fn process_task(name: &str, model_data: Param, model_config: Param, train_data: Param) -> Vec<u8> {
    let model_bytes = model_data.unwrap();
    let model_bytes = decompress_default(&model_bytes).unwrap_log();
    let initial = load_model_from_bytes(&model_bytes).unwrap();
    
    let model_config = model_config.unwrap();
    let model_config = load_model_xml(&model_config).unwrap_log();
    
    let mut controller = NNController::load(model_config.main_layer, model_config.loss_func, initial.clone()).unwrap_log();

    let inputs = train_data.unwrap();
    let inputs = decompress_default(&inputs).unwrap_log();
    let inputs = load_pair_from_bytes(&inputs).unwrap();

    console_log("Start training".to_owned());
    console_log(format!("Inputs = {:?}, Expected = {:?}", inputs.inputs.shape(), inputs.expected.shape()));
    
    let train_result = controller.train_batch(inputs.inputs, &inputs.expected).unwrap_log();
    console_log(format!("Result = {}", train_result));
    let mut result = controller.export();
    console_log("After export".to_owned());

    for (key, value) in result.iter_mut() {
        for (index, arr) in value.iter_mut().enumerate() {
            if initial.get(key).is_none() {
                console_log(format!("Key {} not found", key))
            }
            
            let initial_val = &initial.get(key).and_then(|o| o.get(index));
            match initial_val {
                Some(initial_val) => {
                    if arr.shape()!=initial_val.shape() {
                        console_log(format!("Shapes differ {:?} != {:?}", arr.shape(), initial_val.shape()))
                    }
                    arr.sub_assign(*initial_val);
                }
                _ => console_log(format!("Index {} not found in key {}", index, key))
            }
            
        }
    }
    console_log("After assign".to_owned());

    let result = save_task_result_bytes(TaskResult::Train(result)).unwrap_log();
    console_log("After save".to_owned());
    compress_default(&result).unwrap_log()
}
