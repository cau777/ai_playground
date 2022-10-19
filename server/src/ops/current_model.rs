use crate::ops::model_source::ModelSource;
use codebase::integration::compression::compress_default;
use codebase::integration::proto_loading::{load_model_from_bytes, save_model_bytes};
use codebase::nn::layers::nn_layers::GenericStorage;
use rocket::State;
use std::io;
use std::ops::AddAssign;
use std::sync::RwLock;

pub type CurrentModelState = State<RwLock<CurrentModel>>;
const SAVE_AFTER: u32 = 50; //TODO: test after

pub struct CurrentModel {
    model: GenericStorage,
    version: u32,
}

impl CurrentModel {
    pub fn new(source: &mut ModelSource) -> Self {
        let best = source.latest();
        let model_bytes = source.load_model_bytes(best).unwrap();
        let model = load_model_from_bytes(&model_bytes).unwrap();
        Self {
            model,
            version: best,
        }
    }

    pub fn reload(&mut self, source: &mut ModelSource) {
        *self = Self::new(source);
    }

    pub fn increment(&mut self, deltas: GenericStorage) {
        for (key, value) in deltas.into_iter() {
            if !self.model.contains_key(&key) {
                self.model.insert(key, value);
            } else {
                let current_item = self.model.get_mut(&key).unwrap();
                for (index, delta_arr) in value.into_iter().enumerate() {
                    if current_item.len() <= index {
                        current_item.insert(index, delta_arr);
                    } else {
                        println!("{}->{}", key, index);
                        println!("Self {:?}", current_item[index].iter().take(10).collect::<Vec<_>>());
                        println!("Deltas {:?}", delta_arr.iter().take(10).collect::<Vec<_>>());
                        
                        current_item.get_mut(index).unwrap().add_assign(&delta_arr);
                        println!("Result {:?}", current_item[index].iter().take(10).collect::<Vec<_>>());
                    }
                }
            }
        }

        self.version += 1;
    }

    pub fn should_save(&self) -> bool {
        self.version % SAVE_AFTER == 0
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    
    pub fn save_to(&self, source: &mut ModelSource) -> Result<(), io::Error> {
        source.save_model(self.version, &self.model)
    }

    pub fn export(&self) -> Vec<u8> {
        //   TODO
        let bytes = save_model_bytes(&self.model).unwrap();
        compress_default(&bytes).unwrap()
    }
}
