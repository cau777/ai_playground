use std::collections::{HashMap, HashSet};
use std::iter::zip;
use ndarray::{Axis, stack};
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

pub fn remove_from_storage3(storage: &mut GenericStorage, key: &str) -> [ArrayDynF; 3] {
    let mut data = storage.remove(key).unwrap();
    [data.remove(0), data.remove(0), data.remove(0)]
}

pub fn get_mut_from_storage<'a>(storage: &'a mut GenericStorage, key: &str, index: usize) -> &'a mut ArrayDynF {
    let data = storage.get_mut(key).unwrap();
    data.get_mut(index).unwrap()
}

fn assert_all_same_keys<'a>(mut items: impl Iterator<Item=&'a GenericStorage>) -> bool {
    let mut expected = HashSet::new();
    let first = items.next().unwrap();

    for key in first.keys() {
        expected.insert(key.to_owned());
    }

    for item in items {
        if item.len() != expected.len() {
            return false;
        }
        for key in item.keys() {
            if !expected.contains(key) {
                return false;
            }
        }
    }

    true
}

fn assert_all_same_shapes<'a>(mut items: impl Iterator<Item=&'a GenericStorage>) -> bool {
    let mut expected: HashMap<String, Vec<Vec<usize>>> = HashMap::new();
    let first = items.next().unwrap();

    for (key, value) in first {
        expected.insert(key.to_owned(), value.iter().map(|o| o.shape().to_vec()).collect());
    }

    for item in items {
        for (key, value) in item {
            for (expected, actual) in zip(expected[key].iter(), value.iter().map(|o| o.shape())) {
                if expected.as_slice() != actual {
                    return false;
                }
            }
        }
    }

    true
}

pub fn combine_storages<'a>(items: &'a[&'a GenericStorage]) -> Option<GenericStorage> {
    if !assert_all_same_keys(items.iter().copied()) || !assert_all_same_shapes(items.iter().copied()){
        None
    } else {
        let mut result = GenericStorage::new();

        for key in items[0].keys() {
            let size = items[0][key].len();
            let mut inner = Vec::with_capacity(size);
            for i in 0..size {
                let views: Vec<_> = items.iter().map(|o| o[key][i].view()).collect();
                inner.push(stack(Axis(0), &views).ok()?)
            }
            result.insert(key.to_owned(), inner);
        }

        Some(result)
    }
}

pub fn split_storages(item: GenericStorage, parts: usize) -> Option<Vec<GenericStorage>> {
    let mut result = vec![GenericStorage::default(); parts];
    for (key, value) in item {
        for r in &mut result {
            r.insert(key.clone(), vec![]);
        }

        for arr in value {
            let arr_parts: Vec<_> = arr.outer_iter().map(|o| o.to_owned()).collect();
            if arr_parts.len() != parts {
                return None;
            }

            for (p, r) in zip(arr_parts, &mut result) {
                r.get_mut(&key).as_mut().unwrap().push(p);
            }
        }
    }

    Some(result)
}