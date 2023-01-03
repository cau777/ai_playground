use std::fs::OpenOptions;
use std::sync::Arc;
use codebase::chess::board::Board;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::game_result::{DrawReason, GameResult};
use codebase::chess::movement::Movement;
use codebase::chess::openings::openings_tree::OpeningsTree;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F, ArrayDynF};
use codebase::utils::ndarray::{Axis, stack};
use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::chess::BATCH_SIZE;
use crate::chess::game_metrics::GameMetrics;
use crate::chess::results_aggregator::ResultsAggregator;
use crate::EnvConfig;

#[derive(Copy, Clone)]
pub enum NextNodeStrategy {
    DepthFirst { total_full_paths: usize },
    BreadthFirst { total_iterations: usize },
}

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

            let expected: Vec<_> = chunk.iter().map(|(_, v)| v.max(-0.995).min(0.995)).collect();
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

        let mut nodes = vec![0; count];
        let mut queue = Vec::with_capacity(count);
        let mut completed = (0..count).collect_vec();
        let mut metrics = GameMetrics::default();

        let mut explored_paths = 0;
        let mut iterations = 0;
        while Self::should_continue(explored_paths, iterations, strategy) {
            let mut failed = Vec::new();
            for c in completed {
                let new = Self::create_request(&trees, &mut cursors, &nodes, c, &mut metrics, strategy);
                match new {
                    Some(new) => queue.push(new),
                    None => failed.push(c),
                }
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
                                             
            iterations += completed.len();
            completed.extend(failed);
            // println!("------ {} ------", completed.len());
        }

        // OpenOptions::new().create(true).write(true).open("./tree.svg").unwrap()
        //     .write_all(trees[0].borrow().to_svg().as_bytes()).unwrap();

        metrics.total_nodes = trees.iter().map(|o| o.len() as u64).sum();
        metrics.rescale_by_branches();

        let mut result = Vec::new();
        for i in 0..count {
            result.extend(trees[i].trainable_nodes(&mut cursors[i]));
        }
        (result, metrics)
    }

    fn should_continue(explored_paths: usize, iterations: usize, strategy: NextNodeStrategy) -> bool {
        match strategy {
            NextNodeStrategy::DepthFirst { total_full_paths } => explored_paths < total_full_paths,
            NextNodeStrategy::BreadthFirst { total_iterations } => iterations < total_iterations,
        }
    }

    fn create_request(trees: &[DecisionTree], cursors: &mut [TreeCursor], nodes: &[usize], x: usize, metrics: &mut GameMetrics,
                      strategy: NextNodeStrategy) -> Option<ResultsAggregator> {
        let prev_node = nodes[x];
        let tree = &trees[x];
        let c = &mut cursors[x];

        let side = tree.get_side_at(prev_node);
        let mut rng = thread_rng();
        let next = match Self::decide_next_node(prev_node, tree, strategy) {
            Some(v) => v,
            None => return None,
        };

        tree.move_cursor(c, next);
        let mut controller = c.get_controller().clone();
        let continuations = controller.get_opening_continuations();
        let mut agg;

        if continuations.is_empty() {
            let moves = controller.get_possible_moves(side);
            agg = ResultsAggregator::new(x, moves.len());

            for m in moves {
                controller.apply_move(m);

                let moves = controller.get_possible_moves(!side);
                Self::handle_game_result(metrics, prev_node, tree, &mut controller, &mut agg, m, &moves);

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

        Some(agg)
    }

    fn decide_next_node(prev_node: usize, tree: &DecisionTree, strategy: NextNodeStrategy) -> Option<usize> {
        // println!("{:?}", tree.get_unexplored_node());
        match strategy {
            NextNodeStrategy::DepthFirst { .. } => Some(prev_node),
            NextNodeStrategy::BreadthFirst { .. } => if prev_node == 0 {
                Some(0)
            } else{
                tree.get_unexplored_node()
            }
        }
    }


    fn handle_game_result(metrics: &mut GameMetrics, node: usize, tree: &DecisionTree,
                          controller: &mut BoardController, agg: &mut ResultsAggregator,
                          m: Movement, moves: &Vec<Movement>) {
        match controller.get_game_result(moves) {
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
                agg.submit(index, if side { 1.0 } else { -1.0 }, true, false);
            }
        }
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
                       nodes: &mut [usize], explored_paths: &mut usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(queue.len());
        while let Some(agg) = queue.get(0) {
            if !agg.is_ready() { break; }
            result.push(agg.owner);
            self.apply(agg, trees, nodes, explored_paths, cursors);
            queue.remove(0);
        }
        result
    }

    fn apply(&self, agg: &ResultsAggregator, trees: &mut [DecisionTree], nodes: &mut [usize],
             explored_paths: &mut usize, cursors: &mut [TreeCursor]) {
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
                }
            };
            *explored_paths += 1;
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
