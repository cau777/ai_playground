pub struct BuilderOptions {
    pub next_node_strategy: NextNodeStrategy,
    pub batch_size: usize,
    pub max_cache: usize,
    pub random_node_chance: f64,
}

pub enum NextNodeStrategy {
    BestNode,
    Deepest,
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            next_node_strategy: NextNodeStrategy::BestNode,
            batch_size: 64,
            max_cache: 10_000,
            random_node_chance: 0.0,
        }
    }
}