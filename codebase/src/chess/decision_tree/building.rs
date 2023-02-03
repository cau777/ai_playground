use ndarray::{Axis, stack};
use ndarray_rand::rand::{Rng, thread_rng};
use crate::ArrayDynF;
use crate::chess::board_controller::BoardController;
use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::DecisionTree;
use crate::chess::decision_tree::results_aggregator::ResultsAggregator;
use crate::chess::game_result::{GameResult};
use crate::chess::movement::Movement;
use crate::nn::controller::NNController;

type OnGameResultParams<'a> = (GameResult, &'a DecisionTree, usize);

#[derive(Copy, Clone)]
pub enum NextNodeStrategy {
    ContinueLineThenBestVariant { min_full_paths: usize },
    ContinueLineThenBestVariantOrRandom { min_full_paths: usize, random_node_chance: f64 },
    BestNodeAlways { min_nodes_explored: usize },
    BestOrRandomNode { min_nodes_explored: usize, random_node_chance: f64 },
}

pub struct DecisionTreesBuilder {
    initial_trees: Vec<DecisionTree>,
    initial_cursors: Vec<TreeCursor>,
    strategy: NextNodeStrategy,
    batch_size: usize,
}

#[derive(Default)]
struct LimitingFactors {
    explored_nodes: usize,
    explored_paths: usize,
    iterations: usize,
}

fn scale_output(x: f32) -> f32 {
    if x < 0.95 && x > 0.05 {
        x
    } else {
        0.998 / (1.0 + f32::exp(-6.54 * (x - 0.5)))
    }
}

impl DecisionTreesBuilder {
    pub fn new(initial_trees: Vec<DecisionTree>, initial_cursors: Vec<TreeCursor>, strategy: NextNodeStrategy, batch_size: usize) -> Self {
        if initial_trees.len() != initial_cursors.len() {
            panic!("initial_trees and initial_cursors should have the same length");
        }

        Self {
            strategy,
            initial_trees,
            initial_cursors,
            batch_size,
        }
    }

    pub fn build(&self, controller: &NNController, mut on_game_result: impl FnMut(OnGameResultParams)) -> (Vec<DecisionTree>, Vec<TreeCursor>) {
        let count = self.initial_trees.len();
        let mut trees = self.initial_trees.clone();
        let mut cursors = self.initial_cursors.clone();
        let mut curr_nodes = vec![0; count];
        let mut queue = Vec::with_capacity(count);
        let mut completed: Vec<_> = (0..count).collect();

        let mut limiting_factors = LimitingFactors::default();
        while self.should_continue(&limiting_factors) {
            for c in completed {
                let new = self.create_request(&mut trees, &mut cursors,
                                              &mut curr_nodes, c, &mut on_game_result, &mut limiting_factors);
                queue.push(new);
            }

            let to_eval = self.get_requests(&queue, count);
            if !to_eval.is_empty() {
                let inputs: Vec<_> = to_eval.iter().map(|(_, _, arr)| arr.view()).collect();
                let inputs = stack(Axis(0), &inputs).unwrap();
                let output = controller.eval_batch(inputs).unwrap();

                for (input_index, out) in output.outer_iter().enumerate() {
                    let (queue_index, local_index, _) = &to_eval[input_index];
                    let value = *out.first().unwrap();
                    let value = scale_output(value);

                    queue[*queue_index].submit(*local_index, value, false, false);
                }
            }

            completed = self.clear_completed(&mut trees, &mut queue, &mut curr_nodes);

            limiting_factors.explored_nodes += completed.len();
            limiting_factors.iterations += 1;
        }

        (trees, cursors)
    }
    
    fn should_continue(&self, limiting: &LimitingFactors) -> bool {
        match self.strategy {
            NextNodeStrategy::ContinueLineThenBestVariant { min_full_paths } => limiting.explored_paths < min_full_paths,
            NextNodeStrategy::ContinueLineThenBestVariantOrRandom { min_full_paths, .. } => limiting.explored_paths < min_full_paths,
            NextNodeStrategy::BestNodeAlways { min_nodes_explored } => limiting.explored_nodes < min_nodes_explored,
            NextNodeStrategy::BestOrRandomNode { min_nodes_explored, .. } => limiting.explored_nodes < min_nodes_explored,
        }
    }

    fn decide_next_node(&self, prev_node: usize, tree: &DecisionTree, limiting: &mut LimitingFactors, rng: &mut impl Rng) -> Option<usize> {
        if tree.len() == 1 {
            return Some(0);
        }

        match self.strategy {
            NextNodeStrategy::ContinueLineThenBestVariant { .. } => {
                let next = tree.get_continuation_at(prev_node);
                next.and_then(|o| {
                    if tree.is_ending_at(o) {
                        limiting.explored_paths += 1;
                        tree.get_best_path_variant()
                    } else {
                        Some(o)
                    }
                })
            }
            NextNodeStrategy::ContinueLineThenBestVariantOrRandom { random_node_chance, .. } => {
                let next = tree.get_continuation_at(prev_node);
                next.and_then(|o| {
                    if tree.is_ending_at(o) {
                        limiting.explored_paths += 1;
                        if rng.gen_range(0.0..1.0) < random_node_chance {
                            tree.get_best_path_random_node(rng)
                        } else {
                            tree.get_best_path_variant()
                        }
                    } else {
                        Some(o)
                    }
                })
            }
            NextNodeStrategy::BestNodeAlways { .. } => {
                tree.get_best_path_variant()
            }
            NextNodeStrategy::BestOrRandomNode { random_node_chance, .. } => {
                if rng.gen_range(0.0..1.0) < random_node_chance {
                    tree.get_random_unexplored_node(rng)
                } else {
                    tree.get_best_path_variant()
                }
            }
        }
    }

    fn handle_game_result(&self, controller: &BoardController, agg: &mut ResultsAggregator,
                          m: Movement, moves: &Vec<Movement>, tree: &DecisionTree, node: usize,
                          on_game_result: &mut impl FnMut(OnGameResultParams)) {
        let result = controller.get_game_result(moves);
        (on_game_result)((result.clone(), tree, node));

        match result {
            GameResult::Undefined => {
                agg.push(m, Some(controller.current().to_array().into_dyn()));
            }
            GameResult::Draw(_) => {
                let index = agg.push(m, None);
                agg.submit(index, 0.5, true, false);
            }
            GameResult::Win(side, _) => {
                let index = agg.push(m, None);
                agg.submit(index, if side { 1.0 } else { 0.0 }, true, false);
            }
        }
    }

    fn get_requests(&self, queue: &[ResultsAggregator], count: usize) -> Vec<(usize, usize, ArrayDynF)> {
        let mut result = Vec::with_capacity(self.batch_size);
        for i in 0..count {
            let requests = queue[i].requests_to_eval()
                .take(self.batch_size - result.len())
                .map(|(local_i, o)| (i, local_i, o.clone()));
            result.extend(requests);
        }
        result
    }

    fn clear_completed(&self, trees: &mut [DecisionTree], queue: &mut Vec<ResultsAggregator>,
                       curr_nodes: &mut [usize]) -> Vec<usize> {
        let mut result = Vec::with_capacity(queue.len());
        while let Some(agg) = queue.get(0) {
            if !agg.is_ready() { break; }
            result.push(agg.owner);
            self.apply(trees, agg, curr_nodes);
            queue.remove(0);
        }
        result
    }

    fn apply(&self, trees: &mut [DecisionTree], agg: &ResultsAggregator, curr_nodes: &mut [usize]) {
        let owner = agg.owner;
        let tree = &mut trees[owner];
        let node = curr_nodes[owner];
        let arrange = agg.arrange();
        if arrange.len() == 0 {
            eprintln!("Unexpected zero-length children in node {} in tree {}", node, tree.to_svg());
        } else {
            tree.submit_node_children(node, &arrange);
        }
    }

    fn create_request(&self, trees: &mut [DecisionTree], cursors: &mut [TreeCursor], prev_nodes: &mut [usize],
                      x: usize, on_game_result: &mut impl FnMut(OnGameResultParams), limiting: &mut LimitingFactors) -> ResultsAggregator {
        let prev_node = prev_nodes[x];

        let mut rng = thread_rng();
        let node = match self.decide_next_node(prev_node, &trees[x], limiting, &mut rng) {
            Some(v) => v,
            None => {
                trees[x] = self.initial_trees[x].clone();
                cursors[x] = self.initial_cursors[x].clone();
                0
            }
        };

        let c = &mut cursors[x];
        let tree = &trees[x];
        prev_nodes[x] = node;

        let side = tree.get_side_at(node);
        tree.move_cursor(c, node);
        let mut controller = c.get_controller().clone();
        let continuations = controller.get_opening_continuations();
        // println!("Continuations from {} = {:?}", controller.current(), continuations);
        let mut agg;

        if continuations.is_empty() {
            let moves = controller.get_possible_moves(side);
            agg = ResultsAggregator::new(x, moves.len());

            for m in moves {
                controller.apply_move(m);

                let moves = controller.get_possible_moves(!side);
                self.handle_game_result(&controller, &mut agg, m, &moves, tree, node, on_game_result);

                controller.revert();
            };
        } else {
            agg = ResultsAggregator::new(x, continuations.len());
            for m in continuations {
                let index = agg.push(m, None);
                // Openings are usually good for both sides
                // Add a small random value so the AI can choose from different openings
                agg.submit(index, rng.gen_range(0.5 + (-1.0)..1.0) * 0.001, false, true);
            }
        }

        agg
    }
}
