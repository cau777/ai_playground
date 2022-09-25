extern crate core;

use ndarray::prelude::*;
use wasm_bindgen::prelude::*;
use crate::utils::Array4F;

mod nn;
mod utils;

#[wasm_bindgen]
pub fn ones() -> String {
    let mut arr = Array4F::ones((5, 5, 5, 5).f());
    arr *= &array![5.0];
    (arr).to_string()
}

#[wasm_bindgen]
pub fn zeros() -> String {
    let arr = Array4F::zeros((5, 5, 5, 5).f());
    arr.to_string()
}

