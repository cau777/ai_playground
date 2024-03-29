use std::sync::Arc;
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::{EnvConfig, ServerClient};
use crate::chess::train_scheduler::TrainerScheduler;

mod train_scheduler;
// mod endgames_trainer;
// mod games_trainer;
mod game_metrics;
mod subtrees_trainer;
mod utils;

const NAME: &str = "chess";
const BATCH_SIZE: usize = 64;

// trait Trainer {
//     fn new(config: &EnvConfig) -> Self;
//     fn train_version(&mut self, config: &EnvConfig);
// }

pub fn train(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig, client: Arc<ServerClient>) {
    let mut trainer = TrainerScheduler::new(initial, model_config, config);
    trainer.train_versions(config.versions, config, client);
}
