use std::fmt::Debug;
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use codebase::chess::decision_tree::building::{BuilderOptions, LimiterFactors, NextNodeStrategy};
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::chess::NAME;
use crate::{EnvConfig, ServerClient};
use crate::chess::game_metrics::GameMetrics;
use crate::chess::subtrees_trainer::SubtreesTrainer;

pub struct TrainerScheduler {
    subtrees_trainer: SubtreesTrainer,
    controller: NNController,
}

impl TrainerScheduler {
    pub fn new(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig) -> Self {
        Self {
            subtrees_trainer: SubtreesTrainer::new(config),
            controller: NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap(),
        }
    }

    pub fn train_versions(&mut self, count: u32, config: &EnvConfig, client: Arc<ServerClient>) {
        let mut all_metrics = Vec::new();
        let mut completed: Option<(_, _)> = None;
        let client = Arc::new(client);

        for version in 0..count {
            println!("Start {}", version + 1);

            let start = Instant::now();
            let prev_completed = completed;
            completed = None;

            let client = client.clone();
            let upload_thread = thread::spawn(move || {
                if let Some(prev_completed) = prev_completed {
                    client.submit(&prev_completed.0, prev_completed.1, NAME);
                }
            });

            all_metrics.push(self.train_cycle(config, &mut completed, version));

            let elapsed = start.elapsed().as_millis();
            println!("   in {}ms ({} cycles/s)", elapsed, 1000.0 / elapsed as f64);
            upload_thread.join().unwrap();
        }
    }

    fn train_cycle(&mut self, config: &EnvConfig, completed: &mut Option<(GenericStorage, f64)>, version: u32) -> GameMetrics {
        println!("Training cycle {} using Subtrees strategy", version);

        let result = self.subtrees_trainer.train_version(&mut self.controller, BuilderOptions {
            limits: LimiterFactors {
                max_full_paths_explored: Some(40),
                ..LimiterFactors::default()
            },
            random_node_chance: 0.15,
            next_node_strategy: NextNodeStrategy::Computed {
                eval_delta_exp: 5.0,
                depth_delta_exp: 0.1,
            },
            ..BuilderOptions::default()
        }, 1);
        Self::print_metrics(&result);

        if version != 0 && version % 5 == 0 {
            // For now, the only criteria to evaluate the model's performance is the time spent training
            let start = SystemTime::now();
            let since_the_epoch = start
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards");

            let export = self.controller.export();
            *completed = Some((export, -(since_the_epoch.as_millis() as f64)));
        }

        result
    }

    fn print_metrics(metrics: &impl Debug) {
        println!("   Finished with {:?}", metrics)
    }
}