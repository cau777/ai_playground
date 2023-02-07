use std::fmt::Debug;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use codebase::chess::decision_tree::building_exp::NextNodeStrategy;
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::chess::endgames_trainer::EndgamesTrainer;
use crate::chess::NAME;
use crate::{EnvConfig, ServerClient};
use crate::chess::game_metrics::GameMetrics;
use crate::chess::games_trainer::{GamesTrainer};

pub struct TrainerScheduler {
    endgames_trainer: EndgamesTrainer,
    games_trainer: GamesTrainer,
    controller: NNController,
}

enum TrainingStrategy {
    Endgames,
    FullGames,
    OpponentsResponses,
}

impl TrainerScheduler {
    pub fn new(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig) -> Self {
        Self {
            endgames_trainer: EndgamesTrainer::new(config),
            games_trainer: GamesTrainer::new(config),
            controller: NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap(),
        }
    }

    pub fn train_versions(&mut self, count: u32, config: &EnvConfig, client: Arc<ServerClient>) {
        let mut queue = Vec::new();
        let mut all_metrics = Vec::new();
        queue.push(TrainingStrategy::FullGames);
        let mut completed: Option<(_, _)> = None;
        let client = Arc::new(client);

        for version in 0..count {
            println!("Start {}", version + 1);

            let prev_completed = completed;
            completed = None;

            let client = client.clone();
            let upload_thread = thread::spawn(move || {
                if let Some(prev_completed) = prev_completed {
                    client.submit(&prev_completed.0, prev_completed.1, NAME);
                }
            });

            self.train_version(config, &mut queue, &mut all_metrics,
                               &mut completed, version);

            upload_thread.join().unwrap();
        }
    }

    fn train_version(&mut self, config: &EnvConfig, queue: &mut Vec<TrainingStrategy>,
                     all_metrics: &mut Vec<GameMetrics>, completed: &mut Option<(GenericStorage, f64)>, version: u32) {
        if queue.is_empty() {
            queue.push(Self::choose_next(all_metrics, version));
        }
        let next = queue.remove(0);

        match next {
            TrainingStrategy::Endgames => println!("Decided to train endgames"),
            TrainingStrategy::FullGames => println!("Decided to play full games against itself"),
            TrainingStrategy::OpponentsResponses => println!("Decided to train guessing the opponent's responses"),
        };

        match next {
            TrainingStrategy::Endgames => {
                let metrics = self.endgames_trainer.train_version(&mut self.controller, config);
                Self::print_metrics(&metrics);
                queue.push(TrainingStrategy::FullGames);
                queue.push(TrainingStrategy::FullGames);
            }
            TrainingStrategy::FullGames => {
                let result = self.games_trainer.train_version(&mut self.controller, NextNodeStrategy::ContinueLineThenBestVariantOrRandom {
                    min_full_paths: 20,
                    random_node_chance: 0.2,
                }, 8);
                Self::print_metrics(&result);
                all_metrics.push(result);
                if all_metrics.len() > 10 {
                    all_metrics.remove(0);
                }
            }
            TrainingStrategy::OpponentsResponses => {
                let result = self.games_trainer.train_version(&mut self.controller, NextNodeStrategy::BestOrRandomNode {
                    min_nodes_explored: 5_000,
                    random_node_chance: 0.92,
                }, 1);
                Self::print_metrics(&result);
                queue.push(TrainingStrategy::FullGames);
            }
        }

        if version != 0 && version % 2 == 0 {
            // For now, the only criteria to evaluate the model's performance is the time spent training
            let start = SystemTime::now();
            let since_the_epoch = start
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");

            let export = self.controller.export();
            *completed = Some((export, -(since_the_epoch.as_millis() as f64)));
        }
    }

    fn print_metrics(metrics: &impl Debug) {
        println!("   Finished with {:?}", metrics)
    }

    fn choose_next(metrics: &[GameMetrics], version: u32) -> TrainingStrategy {
        if metrics.len() < 10 {
            return TrainingStrategy::FullGames;
        }

        let mut avg_metrics = GameMetrics::default();
        for m in metrics {
            avg_metrics.add(m);
        }
        avg_metrics.rescale(1.0 / metrics.len() as f64);

        let win_rate = avg_metrics.white_win_rate + avg_metrics.black_win_rate;
        let unbalanced_win_rate = (avg_metrics.white_win_rate - avg_metrics.black_win_rate).abs() / win_rate > 0.3;
        let low_confidence = avg_metrics.average_confidence < 0.9;

        if version % 10 == 0 && (unbalanced_win_rate || low_confidence) {
            TrainingStrategy::FullGames // TODO: OpponentsResponses
        } else if avg_metrics.aborted_rate > 0.1 || avg_metrics.stalemate_rate > 0.2 {
            TrainingStrategy::Endgames
        } else {
            TrainingStrategy::FullGames
        }
    }
}