extern crate core;

use wasm_bindgen::prelude::*;
use crate::integration::proto_loading::save_model_bytes;
use crate::nn::layers::nn_layers::GenericStorage;
use crate::utils::{Array4F, ArrayDynF};

pub mod nn;
pub mod utils;
pub mod integration;
pub mod compiled_protos;

#[wasm_bindgen]
pub fn test_bytes(b: &[u8]) -> String {
    format!("{:?}", b)
}

#[wasm_bindgen]
pub fn test_proto() -> String {
    let mut storage = GenericStorage::new();
    storage.insert("test".to_owned(), vec![ArrayDynF::ones(vec![2, 3, 4])]);

    let bytes = save_model_bytes(&storage).unwrap();
    format!("{} {:?}", bytes.len(), bytes)
}

