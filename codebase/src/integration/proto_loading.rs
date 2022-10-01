use protobuf::Message;
use crate::compiled_protos::model_storage::{ModelStorageData, NdArrayData, PairData};
use crate::nn::layers::nn_layers::GenericStorage;
use crate::utils::ArrayDynF;

pub fn load_model_from_bytes(bytes: &[u8]) -> Option<GenericStorage> {
    let data = ModelStorageData::parse_from_bytes(bytes).ok()?;
    let mut result = GenericStorage::new();

    for pair in data.pairs.into_iter() {
        let mut arrays = Vec::new();

        for arr in pair.arrays.into_iter() {
            let shape: Vec<usize> = arr.shape.iter().map(|o| *o as usize).collect();
            let array = ArrayDynF::from_shape_vec(shape, arr.numbers).ok()?;
            arrays.push(array);
        }

        result.insert(pair.key, arrays);
    }

    Some(result)
}

pub fn save_model_bytes(model: &GenericStorage) -> protobuf::Result<Vec<u8>> {
    let mut data = ModelStorageData::new();
    
    for (key, value) in model.iter() {
        let arrays = value.iter()
            .map(|o| {
                let mut data = NdArrayData::new();
                data.shape = o.shape().iter().map(|o| *o as u32).collect();
                data.numbers = o.iter().copied().collect();
                data
            }).collect();
        
        let mut pair = PairData::new();
        pair.key = key.clone();
        pair.arrays = arrays;
        data.pairs.push(pair);
    }
    
    data.write_to_bytes()
}