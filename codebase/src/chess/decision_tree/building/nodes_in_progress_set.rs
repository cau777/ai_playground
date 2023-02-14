use std::collections::HashSet;
use crate::chess::decision_tree::node::Node;

pub struct NodesInProgressSet {
    set: HashSet<usize>,
}

impl NodesInProgressSet {
    pub fn new() -> Self {
        Self {
            set: HashSet::new()
        }
    }

    pub fn validate_all(&mut self, nodes: &[Node]) {
        self.set.retain(|&o| !nodes[o].is_visited())
    }

    pub fn contains(&self, node: usize) -> bool {
        self.set.contains(&node)
    }

    pub fn insert(&mut self, node: usize) {
        self.set.insert(node);
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}