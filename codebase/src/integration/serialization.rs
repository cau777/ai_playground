use crate::integration::serde_utils::*;
use crate::{nn::layers::nn_layers::GenericStorage, utils::ArrayDynF};

pub fn write_num_vec(result: &mut Vec<u8>, array: &ArrayDynF) {
    array
        .iter()
        .flat_map(|o| o.to_be_bytes())
        .for_each(|o| result.push(o))
}

fn write_array(result: &mut Vec<u8>, array: &ArrayDynF) {
    write_u32(result, array.shape().len() as u32);
    for shape_item in array.shape().iter() {
        write_u32(result, *shape_item as u32);
    }
    write_num_vec(result, array);
}

fn write_storage(result: &mut Vec<u8>, storage: &GenericStorage) {
    for (key, value) in storage.iter() {
        write_u32(result, key.len() as u32);
        result.extend(key.as_bytes());

        write_u32(result, value.len() as u32);
        for item in value.iter() {
            write_array(result, item);
        }
    }
}

pub fn serialize_array(array: &ArrayDynF) -> Vec<u8> {
    let mut result = Vec::new();
    write_array(&mut result, array);
    result
}

pub fn serialize_storage(storage: &GenericStorage) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(0); // No compression
    write_storage(&mut result, storage);
    result
}

pub fn serialize_version(storage: &GenericStorage, loss: f64) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(0); // No compression
    write_f64(&mut result, loss);
    write_storage(&mut result, storage);
    result
}

pub fn serialize_pairs(pairs: &Pairs) -> Vec<u8> {
    let mut result = Vec::new();

    result.push(0); // No compression
    write_array(&mut result, &pairs.inputs);
    write_array(&mut result, &pairs.expected);

    result
}
