use crate::chess::decision_tree::building::limiting_factors::LimiterFactors;
use crate::chess::decision_tree::building::OnGameResultFn;
use crate::chess::decision_tree::DecisionTree;
use crate::chess::decision_tree::node::Node;

pub struct BuilderOptions {
    pub next_node_strategy: NextNodeStrategy,
    pub batch_size: usize,
    pub max_cache: usize,
    pub random_node_chance: f64,
    pub limits: LimiterFactors,
    pub on_game_result: OnGameResultFn,
}

pub enum NextNodeStrategy {
    BestNode,
    Deepest,
    Computed {depth_factor: f64, best_path_delta_exp: f64},
}

pub fn compute_next_node_score(tree: &DecisionTree, node: &Node, depth_factor: f64, best_path_delta_exp: f64) -> f64 {
    let best_path_delta  = f32::abs(tree.nodes[0].eval() - node.eval()) as f64;
    node.depth as f64 * depth_factor * f64::exp(-best_path_delta * best_path_delta_exp)
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            next_node_strategy: NextNodeStrategy::BestNode,
            batch_size: 64,
            max_cache: 10_000,
            random_node_chance: 0.0,
            limits: Default::default(),
            on_game_result: Box::new(|_| {}),
        }
    }
}