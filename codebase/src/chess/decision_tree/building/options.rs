use crate::chess::decision_tree::building::limiting_factors::LimiterFactors;
use crate::chess::decision_tree::building::OnGameResultFn;
use crate::chess::decision_tree::DecisionTree;
use crate::chess::decision_tree::node::Node;

pub struct BuilderOptions {
    pub next_node_strategy: NextNodeStrategy,
    pub batch_size: usize,
    pub max_cache_bytes: u64,
    pub random_node_chance: f64,
    pub limits: LimiterFactors,
    pub on_game_result: OnGameResultFn,
    pub add_random_to_openings: bool,
}

/// Determines how the next node to explore will be chosen
pub enum NextNodeStrategy {
    /// Try to follow the best moves path
    BestNode,
    /// The deepest node in the tree
    Deepest,
    /// Computes a value for each node based on those properties
    Computed { depth_delta_exp: f64, eval_delta_exp: f64},
}

pub fn compute_next_node_score(tree: &DecisionTree, node: &Node, deepest: usize, depth_delta_exp: f64, eval_delta_exp: f64) -> f64 {
    let deepest = deepest as f64;
    let delta_depth = deepest - node.depth as f64;
    let delta_eval = f64::abs(tree.nodes[0].eval() as f64 - node.eval() as f64);
    f64::exp(-depth_delta_exp * delta_depth - eval_delta_exp * delta_eval)
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            next_node_strategy: NextNodeStrategy::BestNode,
            batch_size: 64,
            max_cache_bytes: 20_000_000,
            random_node_chance: 0.0,
            limits: Default::default(),
            on_game_result: Box::new(|_| {}),
            add_random_to_openings: true,
        }
    }
}