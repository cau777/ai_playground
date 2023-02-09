use crate::chess::decision_tree::building_exp_2::NextNodeStrategy;
use crate::chess::decision_tree::DecisionTree;
use crate::nn::layers::nn_layers::GenericStorage;

#[derive(Clone)]
pub struct Cache {
    pub count: usize,
    buffer: Vec<Option<GenericStorage>>,
    max: usize,
    pub last_searched: usize,
}

impl Cache {
    pub fn new(max: usize) -> Self {
        Self {
            count: 0,
            buffer: vec![None],
            max,
            last_searched: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get(&self, index: usize) -> &Option<GenericStorage> {
        match self.buffer.get(index) {
            Some(v) => v,
            None => &None,
        }
    }

    pub fn push(&mut self, value: Option<GenericStorage>) {
        if value.is_some() {
            self.count += 1;
        }
        self.buffer.push(value);
    }

    pub fn remove(&mut self, index: usize) {
        if self.buffer[index].is_some() {
            self.count -= 1;
        }
        self.buffer[index] = None;
    }

    pub fn remove_excess(&mut self, tree: &DecisionTree, strategy: &NextNodeStrategy) {
        match strategy {
            NextNodeStrategy::BestNode => self.remove_worst_nodes(tree),
            NextNodeStrategy::Deepest => self.remove_shallow_nodes(tree),
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
                    .filter(|&o| self.buffer[o].is_some())
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
        self.count >= self.max
    }
}