use codebase::nn::controller::NNController;
use codebase::utils::GenericResult;
use crate::{FileManager, LoadedModelDep};

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

    pub fn get_loaded(&self) -> Option<&NNController> {
        self.controller.as_ref()
    }
}

pub async fn assert_model_loaded(loaded_model_dep: &LoadedModelDep, target: u32, file_manager: &FileManager) -> GenericResult<()> {
    let curr_version = {
        loaded_model_dep.read().await.version
    };

    if curr_version != target {
        let storage = file_manager.get_storage(target)?;
        let config = file_manager.get_config()?;

        let mut loaded = loaded_model_dep.write().await;
        loaded.controller = Some(NNController::load(config.main_layer, config.loss_func, storage)?);
        loaded.version = target;
    }

    Ok(())
}