use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex};
use bloomfilter::Bloom;
use codebase::chess::board::Board;
use codebase::chess::board_controller::board_hashable::BoardHashable;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::building::{BuilderOptions, DecisionTreesBuilder, LimiterFactors, NextNodeStrategy};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::game_result::{DrawReason, GameResult};
use codebase::chess::openings::openings_tree::OpeningsTree;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F};
use codebase::utils::ndarray::{Axis, stack};
use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::chess::BATCH_SIZE;
use crate::chess::game_metrics::GameMetrics;
use crate::chess::utils::{calc_mean_and_std_dev, calc_nodes_confidence_2, dedupe};
use crate::EnvConfig;

pub struct SubtreesTrainer {
    opening_tree: Arc<OpeningsTree>,
    max_cache_size_kb: u64,
}

impl SubtreesTrainer {
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

    fn get_subtrees(&self, controller: &NNController, limits: LimiterFactors, random_node_chance: f64)
        -> Vec<(DecisionTree, TreeCursor)> {
        const SUBTREE_COUNT: usize = 10; // TODO: analyze
        let tree = vec![DecisionTree::new(true)];
        let cursor = vec![self.create_cursor()];
        let options = BuilderOptions {
            next_node_strategy: NextNodeStrategy::Computed {
                eval_delta_exp: 5.0,
                depth_delta_exp: 0.1,
            },
            limits,
            batch_size: BATCH_SIZE,
            max_cache_bytes: self.max_cache_size_kb * 1_000,
            on_game_result: Box::new(|_| {}),
            random_node_chance: random_node_chance * 2.0,
            ..Default::default()
        };

        let (mut tree, mut cursor) = DecisionTreesBuilder::new(tree, cursor, options)
            .build(controller);
        let tree = tree.remove(0);
        let mut cursor = cursor.remove(0);

        tree.nodes.iter()
            .enumerate()
            .filter(|(_, o)| o.children.is_some())
            .filter(|(_, o)| !o.info.is_opening)
            .map(|(i, o)| {
                // Negate to reverse sorting
                (i, -f32::abs(o.pre_eval - o.eval()))
            })
            .sorted_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .take(SUBTREE_COUNT)
            .map(|i| {
                println!("{}", i);
                tree.create_subtree(&mut cursor, i)
            })
            .collect()
    }

    fn analyze_games(&self, controller: &NNController, count: usize, options: BuilderOptions)
                     -> (Vec<(Board, f32)>, GameMetrics) {
        let zipped = self.get_subtrees(controller, options.limits.clone(), options.random_node_chance);
        let (trees, cursors): (Vec<_>, Vec<_>) = zipped.into_iter().unzip();

        // println!("\n{}\n", trees[0].to_svg());

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

        // println!("\n{}\n", trees[0].to_svg());

        let mut metrics = metrics.lock().unwrap();
        for ending in trees.iter().flat_map(|o| &o.nodes).filter(|o| o.info.is_ending) {
            metrics.total_branches += 1;
            metrics.branches.average_branch_depth += ending.depth as f64;
        }

        let factor = 1.0 / metrics.total_branches as f64;
        metrics.branches.scale(factor);

        for tree in &trees {
            for (index, node) in tree.nodes.iter()
                .enumerate()
                .filter(|(_, o)| o.children.is_some()) {
                let children = node.children.as_ref().unwrap();

                metrics.total_explored_nodes += 1;
                metrics.explored_nodes.avg_children += children.len() as f64;

                let (mean, std_dev) = calc_mean_and_std_dev(tree, index);
                metrics.explored_nodes.avg_mean += mean;
                metrics.explored_nodes.avg_children_std_dev += std_dev;
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

        let result = dedupe(result_to_dedup);
        (result, metrics.clone())
    }

    fn create_cursor(&self) -> TreeCursor {
        let mut controller = BoardController::new_start();
        controller.add_openings_tree(self.opening_tree.clone());
        TreeCursor::new(controller)
    }
}
