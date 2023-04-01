use std::fs::OpenOptions;
use std::sync::{Arc, Mutex};
use codebase::chess::board::Board;
use codebase::chess::board_controller::GameController;
use codebase::chess::decision_tree::building::{BuilderOptions, DecisionTreesBuilder};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::game_result::{DrawReason, GameResult};
use codebase::chess::openings::openings_tree::OpeningsTree;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F};
use codebase::utils::ndarray::{Axis, stack};
use rand::{Rng, thread_rng};
use crate::chess::BATCH_SIZE;
use crate::chess::game_metrics::GameMetrics;
use crate::chess::utils::{calc_nodes_confidence_2, dedupe};
use crate::EnvConfig;

pub struct GamesTrainer {
    opening_tree: Arc<OpeningsTree>,
    max_cache_size_kb: u64,
}

impl GamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        let file = OpenOptions::new().read(true).open(format!("{}/chess/openings.dat", config.mounted_path)).unwrap();
        Self {
            opening_tree: Arc::new(OpeningsTree::load_from_file(file).unwrap()),
            max_cache_size_kb: config.max_cache_size_kb,
        }
    }

    pub fn train_version(&self, controller: &mut NNController, options: BuilderOptions, sim_games: usize) -> GameMetrics {
        let (games, metrics) = self.analyze_games(controller, sim_games, options);

        for chunk in games.chunks(BATCH_SIZE) {
            let inputs: Vec<_> = chunk.iter().map(|(b, _)| b.to_array()).collect();
            let views: Vec<_> = inputs.iter().map(|o| o.view()).collect();
            let inputs = stack(Axis(0), &views).unwrap();

            let expected: Vec<_> = chunk.iter().map(|(_, v)| *v).collect();
            let expected = Array2F::from_shape_vec((expected.len(), 1), expected).unwrap();
            controller.train_batch(inputs.into_dyn(), &expected.into_dyn()).unwrap();
        }
        metrics
    }

    fn analyze_games(&self, controller: &NNController, count: usize, options: BuilderOptions)
                     -> (Vec<(Board, f32)>, GameMetrics) {
        let mut trees = Vec::with_capacity(count);
        let mut cursors = Vec::with_capacity(count);
        for _ in 0..count {
            trees.push(DecisionTree::new(true));
            cursors.push(self.create_cursor());
        }

        let metrics = Arc::new(Mutex::new(GameMetrics::default()));

        let (trees, mut cursors) = {
            let metrics = metrics.clone();

            let options = BuilderOptions {
                on_game_result: Box::new(move |(result, _)| {
                    let mut metrics = metrics.lock().unwrap();

                    match result {
                        GameResult::Undefined => {}
                        GameResult::Draw(reason) => {
                            metrics.branches.draw_rate += 1.0;
                            match reason {
                                DrawReason::Aborted => metrics.branches.aborted_rate += 1.0,
                                DrawReason::Stalemate => metrics.branches.stalemate_rate += 1.0,
                                DrawReason::InsufficientMaterial => metrics.branches.insuff_material_rate += 1.0,
                                DrawReason::FiftyMoveRule => metrics.branches.draw_50mr_rate += 1.0,
                                DrawReason::Repetition => metrics.branches.repetition_rate += 1.0,
                            }
                        }
                        GameResult::Win(side, _) => {
                            if side {
                                metrics.branches.white_win_rate += 1.0;
                            } else {
                                metrics.branches.black_win_rate += 1.0;
                            }
                        }
                    }
                }),
                batch_size: BATCH_SIZE,
                max_cache_bytes: self.max_cache_size_kb * 1_000,
                ..options
            };

            let builder = DecisionTreesBuilder::new(trees, cursors, options);
            builder.build(controller)
        };

        let mut metrics = metrics.lock().unwrap();
        for ending in trees.iter().flat_map(|o| &o.nodes).filter(|o| o.info.is_ending) {
            metrics.total_branches += 1;
            metrics.branches.average_branch_depth += ending.depth as f64;
        }

        let factor = 1.0 / metrics.total_branches as f64;
        metrics.branches.scale(factor);

        for tree in &trees {
            for node in tree.nodes.iter().filter(|o| o.children.is_some()) {
                metrics.total_explored_nodes += 1;
                let children = node.children.as_ref().unwrap();
                let children_len = children.len() as f64;

                metrics.explored_nodes.avg_children += children_len;
                let evals: Vec<_> = children.iter()
                    .map(|&o| tree.nodes[o].pre_eval as f64)
                    .collect();

                let mean = evals.iter().sum::<f64>() / children_len;

                metrics.explored_nodes.avg_mean += mean;
                metrics.explored_nodes.avg_children_std_dev += f64::sqrt(
                    evals.iter().map(|&o| f64::powi(o - mean, 2)).sum::<f64>()
                        /
                        children_len
                );
            }
        }

        let factor = 1.0 / metrics.total_explored_nodes as f64;
        metrics.explored_nodes.scale(factor);

        metrics.total_nodes = trees.iter().map(|o| o.len() as u64).sum();

        let mut result_to_dedup = Vec::new();
        let mut rng = thread_rng();

        for i in 0..count {
            const SHIFT: f32 = 0.000_05;
            let tree = &trees[i];

            let c = &mut cursors[i];
            let confidence = calc_nodes_confidence_2(tree);
            let total_confidence = confidence.iter()
                .flatten()
                .map(|&o| o as f64)
                .sum::<f64>();
            let total_valid = confidence.iter().flatten().count() as f64;
            metrics.explored_nodes.avg_confidence += total_confidence / total_valid / count as f64;

            let nodes = tree.nodes.iter()
                .enumerate()
                // Ignore openings, because there isn't much to learn from them
                .filter(|(_, o)| !o.info.is_opening)
                // Ignore ending positions, like checkmate
                .filter(|(_, o)| !o.info.is_ending)
                .filter(|(_, o)| o.children.is_some())
                .map(|(i, o)| {
                    c.go_to(i, &tree.nodes);
                    let board = c.get_controller().current().clone();
                    let eval = o.eval() * confidence[i].unwrap_or(1.0);
                    (board, eval)
                })
                // Avoid values too close to 1 or 0
                .map(|(b, v)| (b, v.max(-0.9995)))
                .map(|(b, v)| (b, v.min(0.9995)))
                // Shift the value by a small random to avoid loops in training
                .map(|(b, v)| (b, v * (1.0 + rng.gen_range((-SHIFT)..SHIFT))));

            result_to_dedup.extend(nodes);
        }

        // let slice = &result_to_dedup[..10_000];
        // assert_eq!({
        //     let mut from = slice.to_vec();
        //     let mut result = Vec::new();
        //     Self::dedupe_hashmap(&mut from, &mut result);
        //     result
        // }.len(), {
        //     Self::dedupe(slice.to_vec())
        // }.len());

        let result = dedupe(result_to_dedup);
        (result, metrics.clone())
    }
    
    fn create_cursor(&self) -> TreeCursor {
        let mut controller = GameController::new_start();
        controller.set_openings_book(self.opening_tree.clone());
        TreeCursor::new(controller)
    }
}
