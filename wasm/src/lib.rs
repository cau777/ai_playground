extern crate core;

use std::fmt::format;
use std::ops::SubAssign;
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};
use codebase::integration::array_pair::ArrayPair;
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::proto_loading::{load_model_from_bytes, load_pair_from_bytes, save_model_bytes};
use codebase::nn::controller::NNController;
use codebase::nn::layers::dense_layer::{DenseLayerConfig, DenseLayerInit};
use codebase::nn::layers::nn_layers::{GenericStorage, Layer};
use codebase::nn::layers::sequential_layer::SequentialLayerConfig;
use codebase::nn::loss::loss_func::LossFunc;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use codebase::utils::ArrayDynF;
use wasm_bindgen::prelude::*;

// #[wasm_bindgen]
// pub fn test_bytes(b: &[u8]) -> String {
//     format!("{:?}", b)
// }
//
// #[wasm_bindgen]
// pub fn test_proto() -> String {
//     let mut storage = GenericStorage::new();
//     storage.insert("test".to_owned(), vec![ArrayDynF::ones(vec![2, 3, 4])]);
//
//     let bytes = save_model_bytes(&storage).unwrap();
//     format!("{} {:?}", bytes.len(), bytes)
// }
//
// static INITIAL_MODEL: Lazy<Arc<RwLock<Option<GenericStorage>>>> = Lazy::new(|| Arc::new(RwLock::new(None)));
// static CONTROLLER: Lazy<Arc<RwLock<Option<NNController>>>> = Lazy::new(|| Arc::new(RwLock::new(None)));
//
// #[wasm_bindgen]
// pub fn load_model(model_bytes: &[u8]) {
//     // TODO
//     let main_layer = Layer::Sequential(SequentialLayerConfig {
//         layers: vec![
//             Layer::Dense(DenseLayerConfig {
//                 in_values: 10,
//                 out_values: 15,
//                 biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
//                 weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
//                 init_mode: DenseLayerInit::Random(),
//             }),
//             Layer::Relu,
//             Layer::Dense(DenseLayerConfig {
//                 in_values: 15,
//                 out_values: 10,
//                 biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
//                 weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
//                 init_mode: DenseLayerInit::Random(),
//             }),
//         ]
//     });
//
//     let model = load_model_from_bytes(model_bytes);
//     *CONTROLLER.write().unwrap() = Some(NNController::load(main_layer, LossFunc::Mse, model.unwrap()).unwrap());
// }
//
// #[wasm_bindgen]
// pub fn train(batch: &[u8]) -> f32 {
//     let mut controller = CONTROLLER.write().unwrap();
//     let mut controller = controller.as_mut().unwrap();
//     let ArrayPair { expected, inputs } = load_pair_from_bytes(batch).unwrap();
//
//     // TODO: loss
//     controller.train_batch(inputs, &expected).unwrap()
// }
//
// #[wasm_bindgen]
// pub fn test(batch: &[u8]) -> f32 {
//     let controller = CONTROLLER.read().unwrap();
//     let controller = controller.as_ref().unwrap();
//     let ArrayPair { expected, inputs } = load_pair_from_bytes(batch).unwrap();
//
//     controller.test_batch(inputs, &expected).unwrap()
// }

fn main_layer() -> Layer {
    Layer::Sequential(SequentialLayerConfig {
        layers: vec![
            Layer::Dense(DenseLayerConfig {
                in_values: 10,
                out_values: 15,
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                init_mode: DenseLayerInit::Random(),
            }),
            Layer::Relu,
            Layer::Dense(DenseLayerConfig {
                in_values: 15,
                out_values: 10,
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                init_mode: DenseLayerInit::Random(),
            }),
        ]
    })
}

// #[wasm_bindgen(getter_with_clone)]
pub struct Deps {
    pub model: Option<Vec<u8>>,
    pub train_data: Option<Vec<u8>>,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn process_task(name: &str, model: Option<Vec<u8>>, train_data: Option<Vec<u8>>) -> Vec<u8> {
    let model_bytes = model.unwrap();
    let model_bytes = decompress_default(&model_bytes).unwrap();
    let initial = load_model_from_bytes(&model_bytes).unwrap();
    let mut controller = NNController::load(main_layer(), LossFunc::Mse, initial.clone()).unwrap();

    let inputs = train_data.unwrap();
    let inputs = decompress_default(&inputs).unwrap();
    let inputs = load_pair_from_bytes(&inputs).unwrap();

    controller.train_batch(inputs.inputs, &inputs.expected).map_err(|o| {
        log(&format!("{:?}", &o));
        o
    }).unwrap();
    let mut result = controller.export();

    for (key, value) in result.iter_mut() {
        for (index, arr) in value.iter_mut().enumerate() {
            let initial_val = &initial[key][index];
            arr.sub_assign(initial_val);
        }
    }

    let result = save_model_bytes(&result).unwrap();
    compress_default(&result).unwrap()
}
//
// #[wasm_bindgen]
// pub fn export_delta() -> Vec<u8> {
//     let controller = CONTROLLER.read().unwrap();
//     let controller = controller.as_ref().unwrap();
//
//     let initial = INITIAL_MODEL.read().unwrap();
//     let initial = initial.as_ref().unwrap();
//
//     let mut result = controller.export();
//     for (key, value) in result.iter_mut() {
//         for (index, arr) in value.iter_mut().enumerate() {
//             let initial_val = &initial[key][index];
//             arr.sub_assign(initial_val);
//         }
//     }
//
//     save_model_bytes(&result).unwrap()
// }