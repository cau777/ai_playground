use std::time::{SystemTime, UNIX_EPOCH};
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::chess::endgames_trainer::EndgamesTrainer;
use crate::chess::NAME;
use crate::{EnvConfig, ServerClient};
use crate::chess::games_trainer::GamesTrainer;

pub struct TrainerScheduler {
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
        let mut prev_aborted_rate = 1.0;
        let mut to_play = 1;

        for version in 0..count {
            println!("Start {}", version + 1);

            if to_play > 0 || prev_aborted_rate < 0.3 {
                println!("Decided to play against itself");
                let metrics = self.games_trainer.train_version(&mut self.controller, config);
                prev_aborted_rate = metrics.aborted_rate;
                println!("    Finished with {:?}", metrics);
                to_play = (to_play - 1).max(0);
            } else {
                println!("Decided to train endgames");
                let metrics = self.endgames_trainer.train_version(&mut self.controller, config);
                println!("    Finished with {:?}", metrics);
                to_play = 3;
            }

            let export = &self.controller.export();
            client.submit(export, 0.0, NAME);
        }
    }
}