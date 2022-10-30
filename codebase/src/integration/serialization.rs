use std::io::{self, Read};
use crate::integration::byte_utils::*;
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

pub fn serialize_array(array: &ArrayDynF) -> Vec<u8> {
    let mut result = Vec::new();
    write_array(&mut result, array);
    result
}

pub fn serialize_storage(storage: &GenericStorage) -> Vec<u8> {
    let mut result = Vec::new();

    result.push(0); // No compression
    for (key, value) in storage.iter() {
        write_u32(&mut result, key.len() as u32);
        result.extend(key.as_bytes());

        write_u32(&mut result, value.len() as u32);
        for item in value.iter() {
            write_array(&mut result, item);
        }
    }

    result
}

pub struct Pairs {
    pub inputs: ArrayDynF,
    pub expected: ArrayDynF,
}

pub fn serialize_pairs(pairs: &Pairs) -> Vec<u8> {
    let mut result = Vec::new();

    result.push(0); // No compression
    write_array(&mut result, &pairs.inputs);
    write_array(&mut result, &pairs.expected);

    result
}

fn read_num_vec(source: &mut &[u8], shape: &[usize]) -> io::Result<ArrayDynF> {
    let length = shape.iter().copied().reduce(|a, b| a * b).unwrap_or(1);
    let mut buffer = vec![0; length * 4];
    source.read_exact(&mut buffer)?;

    let nums = buffer
        .chunks_exact(4)
        .map(|arr| f32::from_be_bytes([arr[0], arr[1], arr[2], arr[3]]))
        .collect();
    Ok(ArrayDynF::from_shape_vec(shape, nums).unwrap())
}

fn read_array(source: &mut &[u8]) -> io::Result<ArrayDynF> {
    let shape_len = read_u32(source)? as usize;
    let mut shape = vec![0; shape_len];
    for i in 0..shape_len {
        shape[i] = read_u32(source)? as usize;
    }

    read_num_vec(source, &shape)
}

pub fn deserialize_array(mut bytes: &[u8]) -> DeserResult<ArrayDynF> {
    Ok(read_array(&mut bytes)?)
}

pub fn deserialize_storage(mut bytes: &[u8]) -> DeserResult<GenericStorage> {
    let mut result = GenericStorage::new();
    let _compression = read_u8(&mut bytes)?;

    while bytes.len() != 0 {
        let key_len = read_u32(&mut bytes)?;
        let mut key_bytes = vec![0; key_len as usize];
        bytes.read_exact(&mut key_bytes)?;
        let key = String::from_utf8(key_bytes)?;

        let vec_len = read_u32(&mut bytes)? as usize;
        let mut arrays = Vec::with_capacity(vec_len);
        for _ in 0..vec_len {
            arrays.push(read_array(&mut bytes)?);
        }

        result.insert(key, arrays);
    }

    Ok(result)
}

pub fn deserialize_pairs(mut bytes: &[u8]) -> DeserResult<Pairs> {
    let _compression = read_u8(&mut bytes)?;
    
    Ok(Pairs {
        inputs: read_array(&mut bytes)?,
        expected: read_array(&mut bytes)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand;
    use rand::prelude::*;

    #[test]
    fn test_integrity() {
        let mut inputs = GenericStorage::new();
        let mut rng = ndarray_rand::rand::thread_rng();
        let dist = ndarray_rand::rand::distributions::Uniform::new(-10.0, 10.0);

        for _ in 0..20 {
            let key_len = rng.gen_range(1..=8);
            let key: String = rng
                .clone()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(key_len)
                .map(char::from)
                .collect();

            let value_len = rng.gen_range(1..5);
            let mut value = Vec::with_capacity(value_len);
            for _ in 0..value_len {
                let shape: Vec<_> = (0..rng.gen_range(1..5))
                    .map(|_| rng.gen_range(1..20))
                    .collect();
                let length = shape.iter().copied().reduce(|a, b| a * b).unwrap();
                let v = rng.clone().sample_iter(dist).take(length).collect();

                value.push(ArrayDynF::from_shape_vec(shape, v).unwrap())
            }

            inputs.insert(key, value);
        }

        let serialized = serialize_storage(&inputs);
        let result = deserialize_storage(&serialized).unwrap();
        println!("{}", serialized.len());
        println!("{}", result.len());

        for key in inputs.keys() {
            assert_eq!(inputs[key], result[key]);
        }
        for key in result.keys() {
            assert_eq!(inputs[key], result[key]);
        }
    }
}
