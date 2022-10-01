use crate::nn::layers::nn_layers::GenericStorage;
use crate::utils::ArrayDynF;

pub fn clone_from_storage1(storage: &GenericStorage, key: &str) -> [ArrayDynF; 1] {
    let data = storage.get(key).unwrap();
    [data[0].clone()]
}

pub fn clone_from_storage2(storage: &GenericStorage, key: &str) -> [ArrayDynF; 2] {
    let data = storage.get(key).unwrap();
    [data[0].clone(), data[1].clone()]
}

pub fn remove_from_storage1(storage: &mut GenericStorage, key: &str) -> [ArrayDynF; 1] {
    let mut data = storage.remove(key).unwrap();
    [data.remove(0)]
}

pub fn remove_from_storage2(storage: &mut GenericStorage, key: &str) -> [ArrayDynF; 2] {
    let mut data = storage.remove(key).unwrap();
    [data.remove(0), data.remove(0)]
}

pub fn get_mut_from_storage<'a>(storage: &'a mut GenericStorage, key: &str, index: usize) -> &'a mut ArrayDynF {
    let data = storage.get_mut(key).unwrap();
    data.get_mut(index).unwrap()
}
