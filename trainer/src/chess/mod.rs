use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::{EnvConfig, ServerClient};
use crate::chess::train_scheduler::TrainerScheduler;

mod train_scheduler;
mod endgames_trainer;
mod games_trainer;
mod results_aggregator;

const NAME: &str = "chess";
const BATCH_SIZE: usize = 128;

// trait Trainer {
//     fn new(config: &EnvConfig) -> Self;
//     fn train_version(&mut self, config: &EnvConfig);
// }

pub fn train(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig, client: &ServerClient) {
    let mut trainer = TrainerScheduler::new(initial, model_config, config);
    trainer.train_versions(config.versions, config, client);
}