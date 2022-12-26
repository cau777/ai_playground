use std::time::{SystemTime, UNIX_EPOCH};
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::chess::endgames_trainer::EndgamesTrainer;
use crate::chess::NAME;
use crate::{EnvConfig, ServerClient};
use crate::chess::games_trainer::GamesTrainer;

pub struct TrainerScheduler{
    endgames_trainer: EndgamesTrainer,
    games_trainer: GamesTrainer,
    controller: NNController,
}

impl TrainerScheduler {
    pub fn new(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig) -> Self {
        Self {
            endgames_trainer: EndgamesTrainer::new(config),
            games_trainer: GamesTrainer::new(config),
            controller: NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap(),
        }
    }

    pub fn train_versions(&mut self, count: u32, config: &EnvConfig, client: &ServerClient) {
        for version in 0..count {
            if true {
                self.games_trainer.train_version(&mut self.controller, config);
            } else {
                println!("Start {}", version + 1);
                let metrics = self.endgames_trainer.train_version(&mut self.controller, config);
                println!("    Finished with {:?}", metrics);

                // For now, the only criteria to evaluate the model's performance is the time spent training
                let start = SystemTime::now();
                let since_the_epoch = start
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards");

                let export = &self.controller.export();
                client.submit(export, since_the_epoch.as_millis() as f64, NAME);
            }
        }
    }
}