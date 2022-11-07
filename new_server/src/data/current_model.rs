use crate::data::model_source::ModelSource;
use codebase::integration::serialization::{deserialize_storage, serialize_storage};
use codebase::nn::layers::nn_layers::GenericStorage;
use std::io;
use codebase::integration::model_deltas;

const SAVE_AFTER: u32 = 20;

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
        model_deltas::import_deltas(&mut self.storage, deltas);
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
