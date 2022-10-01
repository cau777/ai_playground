use crate::nn::layers::nn_layers::GenericStorage;
use crate::utils::ArrayDynF;

pub fn remove_from_storage1(storage: &mut GenericStorage, key: &str) -> Option<[ArrayDynF; 1]> {
    let mut data = storage.remove(key)?;
    Some([data.pop()?])
}

