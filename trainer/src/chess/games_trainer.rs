use std::fs::OpenOptions;
use std::sync::Arc;
use codebase::chess::board::Board;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::game_result::{DrawReason, GameResult};
use codebase::chess::openings::openings_tree::OpeningsTree;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F, ArrayDynF};
use codebase::utils::ndarray::{Axis, stack};
use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::chess::{BATCH_SIZE};
use crate::chess::results_aggregator::ResultsAggregator;
use crate::EnvConfig;

pub struct GamesTrainer {
    opening_tree: Arc<OpeningsTree>,
}

#[derive(Debug, Default)]
pub struct GameMetrics {
    pub total_branches: u64,
    pub total_nodes: u64,

    pub aborted_rate: f64,
    pub repetition_rate: f64,
    pub draw_50mr_rate: f64,
    pub stalemate_rate: f64,
    pub insuff_material_rate: f64,

    pub white_win_rate: f64,
    pub black_win_rate: f64,
    pub draw_rate: f64,
    pub average_len: f64,
}

impl GameMetrics {
    pub fn rescale(&mut self) {
        let factor = 1.0 / self.total_branches as f64;
        self.aborted_rate *= factor;
        self.repetition_rate *= factor;
        self.draw_50mr_rate *= factor;
        self.stalemate_rate *= factor;
        self.insuff_material_rate *= factor;
        self.white_win_rate *= factor;
        self.black_win_rate *= factor;
        self.draw_rate *= factor;
        self.average_len *= factor;
    }
}

impl GamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        let file = OpenOptions::new().read(true).open(format!("{}/chess/openings.dat", config.mounted_path)).unwrap();
        Self {
            opening_tree: Arc::new(OpeningsTree::load_from_file(file).unwrap()),
        }
    }

    pub fn train_version(&self, controller: &mut NNController, _: &EnvConfig) -> GameMetrics {
        let (games, metrics) = self.analyze_games(controller, 8, 24);

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

    fn analyze_games(&self, controller: &NNController, count: usize, total_paths: usize) -> (Vec<(Board, f32)>, GameMetrics) {
        let mut trees = Vec::with_capacity(count);
        let mut cursors = Vec::with_capacity(count);
        for _ in 0..count {
            trees.push(DecisionTree::new(true));
            cursors.push(self.create_cursor());
        }

        let mut nodes = vec![0; count];
        let mut queue = Vec::with_capacity(count);
        let mut completed = (0..count).collect_vec();
        let mut metrics = GameMetrics::default();

        let mut explored_paths = 0;
        while explored_paths < total_paths {
            for c in completed {
                Self::create_requests(&mut queue, &trees, &mut cursors, &nodes, c, &mut metrics);
            }

            let to_eval = Self::get_requests(&queue, count);
            if !to_eval.is_empty() {
                let inputs: Vec<_> = to_eval.iter().map(|(_, _, arr)| arr.view()).collect();
                let inputs = stack(Axis(0), &inputs).unwrap();
                let output = controller.eval_for_train(inputs).unwrap();

                for (input_index, out) in output.output.outer_iter().enumerate() {
                    let (queue_index, local_index, _) = &to_eval[input_index];
                    let value = *out.first().unwrap() - 0.5 * 2.0;

                    queue[*queue_index].submit(*local_index, value, false, false);
                }
            }

            completed = self.clear_completed(&mut queue, &mut trees, &mut cursors, &mut nodes,
                                             &mut explored_paths);
            // println!("------ {} ------", completed.len());
        }

        // OpenOptions::new().create(true).write(true).open("./tree.svg").unwrap()
        //     .write_all(trees[0].borrow().to_svg().as_bytes()).unwrap();

        metrics.total_nodes = trees.iter().map(|o| o.len() as u64).sum();
        metrics.rescale();

        let mut result = Vec::new();
        for i in 0..count {
            result.extend(trees[i].trainable_nodes(&mut cursors[i]));
        }
        (result, metrics)
    }

    fn create_requests(queue: &mut Vec<ResultsAggregator>, trees: &[DecisionTree],
                       cursors: &mut [TreeCursor], nodes: &[usize], x: usize, metrics: &mut GameMetrics) {
        let node = nodes[x];
        let tree = &trees[x];
        let c = &mut cursors[x];
        let side = tree.get_side_at(node);
        let mut rng = thread_rng();

        tree.move_cursor(c, node);
        let mut controller = c.get_controller().clone();
        let continuations = controller.get_opening_continuations();
        let mut agg;

        if continuations.is_empty() {
            let moves = controller.get_possible_moves(side);
            agg = ResultsAggregator::new(x, moves.len());

            for m in moves {
                controller.apply_move(m);

                let moves = controller.get_possible_moves(!side);
                match controller.get_game_result(&moves) {
                    GameResult::Undefined => {
                        agg.push(m, Some(controller.current().to_array().into_dyn()));
                    }
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
                        let index = agg.push(m, None);
                        agg.submit(index, 0.0, true, false);
                    }
                    GameResult::Win(side, _) => {
                        metrics.average_len += tree.get_depth_at(node) as f64;
                        metrics.total_branches += 1;
                        if side {
                            metrics.white_win_rate += 1.0;
                        } else {
                            metrics.black_win_rate += 1.0;
                        }
                        let index = agg.push(m, None);
                        agg.submit(index, if side { 1.0 } else { 0.0 }, true, false);
                    }
                }

                controller.revert();
            };
        } else {
            agg = ResultsAggregator::new(x, continuations.len());
            for m in continuations {
                let index = agg.push(m, None);
                // Openings are usually good for both sides
                // Add a small random value so the AI can choose from different openings
                agg.submit(index, rng.gen_range((-1.0)..1.0) * 0.00001, false, true);
            }
        }

        queue.push(agg);
    }

    fn get_requests(queue: &[ResultsAggregator], count: usize) -> Vec<(usize, usize, ArrayDynF)> {
        let mut result = Vec::with_capacity(BATCH_SIZE);
        for i in 0..count {
            let requests = queue[i].requests_to_eval()
                .take(BATCH_SIZE - result.len())
                .map(|(local_i, o)| (i, local_i, o.clone()));
            result.extend(requests);
        }
        result
    }

    fn clear_completed(&self, queue: &mut Vec<ResultsAggregator>, trees: &mut [DecisionTree], cursors: &mut [TreeCursor],
                       nodes: &mut [usize], explored_nodes: &mut usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(queue.len());
        while let Some(agg) = queue.get(0) {
            if !agg.is_ready() { break; }
            result.push(agg.owner);
            self.apply(agg, trees, nodes, explored_nodes, cursors);
            queue.remove(0);
        }
        result
    }

    fn apply(&self, agg: &ResultsAggregator, trees: &mut [DecisionTree], nodes: &mut [usize],
             explored_nodes: &mut usize, cursors: &mut [TreeCursor]) {
        let tree = &mut trees[agg.owner];
        let node = nodes[agg.owner];
        let new = tree.submit_node_children(node, &agg.arrange());

        if tree.is_ending_at(new) {
            nodes[agg.owner] = match tree.get_unexplored_node() {
                Some(v) => v,
                None => {
                    trees[agg.owner] = DecisionTree::new(true);
                    cursors[agg.owner] = self.create_cursor();
                    0
                },
            };
            *explored_nodes += 1;
        } else {
            nodes[agg.owner] = new;
        }
    }

    fn create_cursor(&self) -> TreeCursor {
        let mut controller = BoardController::new_start();
        controller.add_openings_tree(self.opening_tree.clone());
        TreeCursor::new(controller)
    }
}
