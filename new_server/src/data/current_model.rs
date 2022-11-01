use crate::data::model_source::ModelSource;
use codebase::integration::serialization::{deserialize_storage, serialize_storage};
use codebase::nn::layers::nn_layers::GenericStorage;
use std::io;
use std::ops::AddAssign;

const SAVE_AFTER: u32 = 50;

pub struct CurrentModel {
    storage: GenericStorage,
    version: u32,
}

impl CurrentModel {
    pub fn new(source: &mut ModelSource) -> Self {
        let best = source.latest();
        let model_bytes = source.load_model_bytes(best).unwrap();
        let storage = deserialize_storage(&model_bytes).unwrap();
        Self {
            storage,
            version: best,
        }
    }

    pub fn reload(&mut self, source: &mut ModelSource) {
        *self = Self::new(source);
    }

    pub fn increment(&mut self, versions: u32, deltas: GenericStorage) {
        for (key, value) in deltas.into_iter() {
            if !self.storage.contains_key(&key) {
                self.storage.insert(key, value);
            } else {
                let current_item = self.storage.get_mut(&key).unwrap();
                for (index, delta_arr) in value.into_iter().enumerate() {
                    if current_item.len() <= index {
                        current_item.insert(index, delta_arr);
                    } else {
                        current_item.get_mut(index).unwrap().add_assign(&delta_arr);
                    }
                }
            }
        }

        self.version += versions;
    }

    pub fn should_save(&self) -> bool {
        self.version % SAVE_AFTER == 0
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    
    pub fn save_to(&self, source: &mut ModelSource) -> Result<(), io::Error> {
        source.save_model(self.version, &self.storage)
    }

    pub fn export(&self) -> Vec<u8> {
        serialize_storage(&self.storage)
    }
}
