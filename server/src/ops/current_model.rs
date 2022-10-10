use std::ops::AddAssign;
use std::sync::RwLock;
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::proto_loading::{load_model_from_bytes, save_model_bytes};
use codebase::nn::layers::nn_layers::GenericStorage;
use rocket::State;
use crate::ops::model_source::ModelSource;

pub type CurrentModelState = State<RwLock<CurrentModel>>;

pub struct CurrentModel {
    model: GenericStorage,
    version: u32,
}

impl CurrentModel {
    pub fn new(source: &ModelSource) -> Self {
    //     TODO: most recent
        let best = source.most_recent();
        let model_bytes = source.load_model_bytes(best).unwrap();
        let model = load_model_from_bytes(&model_bytes).unwrap();
        Self {
            model,
            version: best
        }
    }

    pub fn increment(&mut self, deltas: GenericStorage) {
        //  TODO: error handling
        for (key, value) in self.model.iter_mut() {
            for (index, arr) in value.iter_mut().enumerate() {
                let initial_val = &deltas[key][index];
                arr.add_assign(initial_val);
            }
        }

        self.version += 1;
    }

    pub fn should_save(&self) -> bool {
        self.version % 10 == 0 // TODO
    }

    pub fn save_to(&self, source: &mut ModelSource) {
    //      TODO
        unimplemented!()
    }

    pub fn export(&self) -> Vec<u8> {
        //   TODO
        let bytes = save_model_bytes(&self.model).unwrap();
        compress_default(&bytes).unwrap()
    }
}