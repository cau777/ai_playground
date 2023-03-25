use std::collections::HashMap;
use std::mem;
use crate::chess::decision_tree::building::{compute_next_node_score, NextNodeStrategy};
use crate::chess::decision_tree::DecisionTree;
use crate::nn::layers::nn_layers::GenericStorage;

#[derive(Clone)]
pub struct Cache {
    pub count: usize,
    buffer: HashMap<usize, GenericStorage>,
    pub current_bytes: u64,
    max_bytes: u64,
    pub last_searched: usize,
    current: usize,
}

impl Cache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            count: 0,
            buffer: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            last_searched: 0,
            current: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get(&self, index: usize) -> Option<&GenericStorage> {
        self.buffer.get(&index)
    }

    pub fn insert_next(&mut self, value: Option<GenericStorage>) {
        if let Some(value) = value {
            self.count += 1;
            self.current_bytes += Self::count_bytes(&value);
            self.buffer.insert(self.current, value);
        }

        self.current += 1;
    }

    pub fn remove(&mut self, index: usize) {
        if let Some(value) = self.buffer.get(&index) {
            self.count -= 1;
            self.current_bytes -= Self::count_bytes(value);
        }
        self.buffer.remove(&index);
    }

    fn count_bytes(storage: &GenericStorage) -> u64 {
        const ITEM_SIZE: u64 = mem::size_of::<f32>() as u64;

        storage.values()
            .flatten()
            .map(|o| o.len() as u64 * ITEM_SIZE + 8)
            .sum()
    }

    pub fn remove_excess(&mut self, tree: &DecisionTree, strategy: &NextNodeStrategy) {
        match strategy {
            NextNodeStrategy::BestNode => self.remove_worst_nodes(tree),
            NextNodeStrategy::Deepest => self.remove_shallow_nodes(tree),
            NextNodeStrategy::Computed { eval_delta_exp: best_path_delta_exp, depth_delta_exp: depth_factor } => self.remove_worst_computed(tree, *best_path_delta_exp, *depth_factor)
        };

        // Because of some edge-cases (where one of the worst paths becomes the best for example),
        // It's still necessary to have a simpler algorithm
        while self.should_remove() && self.last_searched < self.buffer.len() {
            self.remove(self.last_searched);
            self.last_searched += 1;
        }
    }

    fn remove_worst_nodes(&mut self, tree: &DecisionTree) {
        let mut stack = Vec::new();
        stack.push(0);

        while !stack.is_empty() && self.should_remove() {
            let node_index = stack.pop().unwrap();
            let node = &tree.nodes[node_index];

            if node.children.is_some() {
                let children: Vec<_> = node.get_ordered_children(tree.start_side)
                    .unwrap()
                    .copied()
                    // Get only the nodes whose cache is stored
                    .filter(|o| self.buffer.contains_key(o))
                    // Get the worst nodes first
                    .rev()
                    .collect();
                if children.is_empty() {
                    // It can be removed from the cache because all of its children are also removed
                    self.remove(node_index);
                } else {
                    stack.extend(children)
                }
            } else {
                // If it is a leaf, just remove if
                self.remove(node_index);
            }
        }
    }

    fn remove_worst_computed(&mut self, tree: &DecisionTree, best_path_delta_exp: f64, depth_factor: f64) {
        let deepest = tree.nodes.iter().map(|o| o.depth).max().unwrap_or_default();
        let mut nodes: Vec<_> = tree.nodes.iter()
            .map(|o| compute_next_node_score(tree, o, deepest, depth_factor, best_path_delta_exp))
            .enumerate()
            .collect();

        nodes.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        for (node_index, _) in nodes {
            if !self.should_remove() { break; }
            self.remove(node_index);
        }
    }

    fn remove_shallow_nodes(&mut self, tree: &DecisionTree) {
        let max_parent_depth = tree.nodes.iter()
            .filter(|o| o.is_visited())
            .map(|o| o.depth)
            .max()
            .unwrap_or_default();

        // First remove the nodes that are less than the maximum depth
        for (i, n) in tree.nodes.iter().enumerate() {
            if !self.should_remove() { break; }
            if n.depth < max_parent_depth {
                self.remove(i)
            }
        }

        // After, select them in the reverse order of the algorithm in games_producer.rs
        let parents = tree.nodes.iter()
            .enumerate()
            .filter(|(_, o)| o.is_visited())
            .filter(|(_, o)| o.depth == max_parent_depth);

        let children = parents
            .filter_map(|(_, o)| o.get_ordered_children(tree.start_side))
            .flat_map(|o| o.rev())
            .copied();

        for i in children {
            if !self.should_remove() { break; }
            self.remove(i);
        }
    }

    fn should_remove(&self) -> bool {
        self.current_bytes >= self.max_bytes
    }
}