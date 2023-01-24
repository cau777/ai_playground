#[derive(Copy, Clone)]
pub enum NextNodeStrategy {
    ContinueLineThenBestVariant { min_full_paths: usize },
    ContinueLineThenBestVariantOrRandom { min_full_paths: usize, random_node_chance: f64 },
    BestNodeAlways { min_nodes_explored: usize },
    BestOrRandomNode { min_nodes_explored: usize, random_node_chance: f64 },
}