use std::ops::{SubAssign, AddAssign};

use crate::nn::layers::nn_layers::GenericStorage;

pub fn export_deltas(initial: &GenericStorage, result: &mut GenericStorage) {
    for (key, value) in result.iter_mut() {
        for (index, arr) in value.iter_mut().enumerate() {
            let initial_val = &initial.get(key).and_then(|o| o.get(index));
            match initial_val {
                Some(initial_val) => {
                    arr.sub_assign(*initial_val);
                }
                _ => {},
            }
        }
    }
}

pub fn import_deltas(current: &mut GenericStorage, deltas: GenericStorage) {
    for (key, value) in deltas.into_iter() {
        if !current.contains_key(&key) {
            current.insert(key, value);
        } else {
            let current_item = current.get_mut(&key).unwrap();
            for (index, delta_arr) in value.into_iter().enumerate() {
                if current_item.len() <= index {
                    current_item.insert(index, delta_arr);
                } else {
                    current_item.get_mut(index).unwrap().add_assign(&delta_arr);
                }
            }
        }
    }
}