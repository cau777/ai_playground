use std::io::ErrorKind::InvalidData;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::LayerError;
use tokio::io;
use crate::FileManager;

#[derive(Default)]
pub struct LoadedModel {
    version: u32,
    controller: Option<NNController>,
}

impl LoadedModel {
    pub fn new() -> Self {
        Self {
            version: 0,
            controller: None,
        }
    }

    pub fn assert_loaded(&mut self, target: u32, file_manager: &FileManager) -> Result<(), LayerError> {
        if self.version != target {
            let storage = file_manager.get_storage(target)?;
            let config = file_manager.get_config()?;
            self.controller = Some(NNController::load(config.main_layer, config.loss_func, storage)?);
            self.version = target;
        }
        Ok(())
    }
    pub fn get_loaded(&self) -> Option<&NNController> {
        self.controller.as_ref()
    }
}