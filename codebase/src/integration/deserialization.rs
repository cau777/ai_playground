use std::io;
use std::io::Read;
use crate::ArrayDynF;
use crate::integration::serde_utils::*;
use crate::nn::layers::nn_layers::GenericStorage;

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

fn read_storage(source: &mut &[u8]) -> DeserResult<GenericStorage> {
    let mut result = GenericStorage::new();

    while source.len() != 0 {
        let key_len = read_u32(source)?;
        let mut key_bytes = vec![0; key_len as usize];
        source.read_exact(&mut key_bytes)?;
        let key = String::from_utf8(key_bytes)?;

        let vec_len = read_u32(source)? as usize;
        let mut arrays = Vec::with_capacity(vec_len);
        for _ in 0..vec_len {
            arrays.push(read_array(source)?);
        }

        result.insert(key, arrays);
    }

    Ok(result)
}

pub fn deserialize_array(mut bytes: &[u8]) -> DeserResult<ArrayDynF> {
    Ok(read_array(&mut bytes)?)
}

pub fn deserialize_storage(mut bytes: &[u8]) -> DeserResult<GenericStorage> {
    let _compression = read_u8(&mut bytes)?;
    let result = read_storage(&mut bytes)?;
    Ok(result)
}

pub fn deserialize_version(mut bytes: &[u8])-> DeserResult<(GenericStorage, f64)> {
    let _compression = read_u8(&mut bytes)?;
    let loss = read_f64(&mut bytes)?;
    let storage = read_storage(&mut bytes)?;
    Ok((storage, loss))
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
    use crate::integration::serialization::serialize_storage;

    #[test]
    fn test_integrity() {
        let mut inputs = GenericStorage::new();
        let mut rng = thread_rng();
        let dist = rand::distributions::Uniform::new(-10.0, 10.0);

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
