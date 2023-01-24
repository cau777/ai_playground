use std::fs::OpenOptions;
use std::sync::Arc;
use codebase::chess::board::Board;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::building_exp::{DecisionTreesBuilder, NextNodeStrategy};
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
use crate::EnvConfig;


pub struct GamesTrainer {
    opening_tree: Arc<OpeningsTree>,
}

impl GamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        let file = OpenOptions::new().read(true).open(format!("{}/chess/openings.dat", config.mounted_path)).unwrap();
        Self {
            opening_tree: Arc::new(OpeningsTree::load_from_file(file).unwrap()),
        }
    }

    pub fn train_version(&self, controller: &mut NNController, strategy: NextNodeStrategy) -> GameMetrics {
        let (games, metrics) = self.analyze_games(controller, 8, strategy);

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

    fn analyze_games(&self, controller: &NNController, count: usize, strategy: NextNodeStrategy)
                     -> (Vec<(Board, f32)>, GameMetrics) {
        let mut trees = Vec::with_capacity(count);
        let mut cursors = Vec::with_capacity(count);
        for _ in 0..count {
            trees.push(DecisionTree::new(true));
            cursors.push(self.create_cursor());
        }

        let mut metrics = GameMetrics::default();
        let builder = DecisionTreesBuilder::new(trees, cursors, strategy, BATCH_SIZE);

        let (trees, mut cursors) = builder.build(controller, |(result, tree, node)| {
            match result {
                GameResult::Undefined => {}
                GameResult::Draw(reason) => {
                    metrics.draw_rate += 1.0;
                    metrics.average_len += tree.get_depth_at(node) as f64;
                    metrics.total_branches += 1;
                    match reason {
                        DrawReason::Aborted => metrics.aborted_rate += 1.0,
                        DrawReason::Stalemate => metrics.stalemate_rate += 1.0,
                        DrawReason::InsufficientMaterial => metrics.insuff_material_rate += 1.0,
                        DrawReason::FiftyMoveRule => metrics.draw_50mr_rate += 1.0,
                        DrawReason::Repetition => metrics.repetition_rate += 1.0,
                    }
                }
                GameResult::Win(side, _) => {
                    metrics.average_len += tree.get_depth_at(node) as f64;
                    metrics.total_branches += 1;
                    if side {
                        metrics.white_win_rate += 1.0;
                    } else {
                        metrics.black_win_rate += 1.0;
                    }
                }
            }
        });
        metrics.total_nodes = trees.iter().map(|o| o.len() as u64).sum();
        metrics.rescale_by_branches();

        // std::fs::OpenOptions::new().write(true).create(true).open(
        //     format!("../out/{}_1.svg",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("Time went backwards").as_secs()
        //     )).unwrap().write_all(trees[0].to_svg().as_bytes()).unwrap();

        // std::fs::OpenOptions::new().write(true).create(true).open(
        //     format!("../out/{}_2.svg",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("Time went backwards").as_secs()
        //     )).unwrap().write_all(trees[1].to_svg().as_bytes()).unwrap();

        let mut result = Vec::new();
        let mut rng = thread_rng();

        for i in 0..count {
            const SHIFT: f32 = 0.0005;

            let nodes = trees[i]
                .trainable_nodes(&mut cursors[i])
                .map(|(b, v)| (b, v.max(0.005)))
                .map(|(b, v)| (b, v.min(0.995)))
                // Shift the value by a small random to avoid loops in training
                .map(|(b, v)| (b, v * (1.0 + rng.gen_range((-SHIFT)..SHIFT))));
            result.extend(nodes);
        }
        (result, metrics)
    }

    fn create_cursor(&self) -> TreeCursor {
        let mut controller = BoardController::new_start();
        controller.add_openings_tree(self.opening_tree.clone());
        TreeCursor::new(controller)
    }
}
