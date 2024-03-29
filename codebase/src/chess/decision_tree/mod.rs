pub use node::NodeExtraInfo;

use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::best_path_iterator::BestPathIterator;
use crate::chess::decision_tree::node::Node;
use crate::chess::movement::Movement;

pub(crate) mod node;
mod best_path_iterator;
pub mod cursor;
mod svg_export;
pub mod building;


#[derive(Debug, Clone)]
pub struct DecisionTree {
    pub start_side: bool,
    pub nodes: Vec<Node>,
}

impl DecisionTree {
    pub fn new(start_side: bool) -> Self {
        Self {
            start_side,
            // Empty first node
            nodes: vec![Node::new(0, Movement::default(), 0.0, 0, None, NodeExtraInfo::default())],
        }
    }

    pub fn create_subtree(&self, cursor: &mut TreeCursor, index: usize) -> (DecisionTree, TreeCursor) {
        cursor.go_to(index, &self.nodes);
        let controller = cursor.get_controller().clone();
        let new_cursor = TreeCursor::new(controller);
        let mut new_tree = Self {
            start_side: self.get_side_at(index),
            // Empty first node
            nodes: vec![Node::new(0, Movement::default(), 0.0, 0, None, NodeExtraInfo::default())],
        };

        let mut stack: Vec<(usize, usize)> = Vec::new();
        stack.push((index, 0));

        // Add every node from the old_tree to the new_tree
        while !stack.is_empty() {
            let (old_index, new_index) = stack.pop().unwrap();

            if let Some(children) = self.nodes[old_index].children.as_ref() {
                let len = new_tree.len();

                let children_info: Vec<_> = children.iter()
                    .map(|&i| {
                        let child = &self.nodes[i];
                        (child.movement, child.pre_eval, child.info)
                    })
                    .collect();

                new_tree.submit_node_children(new_index, &children_info);

                // New nodes are stored in the same order
                for (i, &c) in children.iter().enumerate() {
                    stack.push((c, len + i));
                }
            }
        }

        (new_tree, new_cursor)
    }

    pub fn best_path_moves(&self) -> impl Iterator<Item=&Movement> {
        BestPathIterator::new(&self.nodes, self.start_side, false)
            .map(|o| &self.nodes[o].movement)
    }

    pub fn submit_node_children(&mut self, parent: usize, positions_and_evals: &[(Movement, f32, NodeExtraInfo)]) -> usize {
        let depth = self.nodes[parent].depth + 1;
        let nodes: Vec<_> = positions_and_evals.iter()
            .copied()
            .map(|(m, e, info)| {
                Node::new(parent, m, e, depth, None, info)
            })
            .map(|node| {
                self.nodes.push(node);
                self.nodes.len() - 1 // Index of the added node
            })
            .collect();

        self.nodes[parent].children = Some(nodes);

        for node_index in self.path_to_root(parent) {
            // This clone is probably optimized at compilation
            let mut node = self.nodes[node_index].clone();
            node.refresh_children(&self.nodes, self.start_side);
            self.nodes[node_index] = node;
        }

        *self.nodes[parent].get_ordered_children(self.start_side)
            .and_then(|mut o| o.next()).unwrap()
    }

    pub fn get_side_at(&self, index: usize) -> bool {
        self.nodes[index].get_current_side(self.start_side)
    }

    pub fn is_ending_at(&self, index: usize) -> bool {
        self.nodes[index].info.is_ending
    }

    pub fn get_best_path_variant(&self) -> Option<usize> {
        BestPathIterator::new(&self.nodes, self.start_side, true)
            .filter_map(|o| self.nodes[o].get_ordered_children(self.start_side))
            // Get the best and second best unexplored options for a node
            .filter_map(|mut o| {
                let first = o.next();
                let second = o.find(|o| {
                    let node = &self.nodes[**o];
                    node.children.is_none() && !node.info.is_ending
                });
                match (first, second) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None
                }
            })
            // Compute the difference between the eval of the first continuation option and the second best
            // The logic is that the AI may misevaluate a position by a little and get a wrong result
            .map(|(a, b)| {
                let node_a = &self.nodes[*a];
                let node_b = &self.nodes[*b];
                ((node_a.eval() - node_b.eval()).abs(), b)
            })
            .min_by(|(v1, _), (v2, _)| v1.total_cmp(v2))
            .map(|(_, n)| *n)
    }

    fn path_to_root(&self, mut index: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.nodes[index].depth + 1);

        while index != 0 {
            result.push(index);
            index = self.nodes[index].parent;
        }
        result.push(0);

        result
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_depth_at(&self, index: usize) -> usize {
        self.nodes[index].depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_node_children() {
        fn build(tree: &mut DecisionTree) {
            tree.submit_node_children(0, &[
                (Movement::from_notations("A1", "A7"), 1.0, Default::default()),
                (Movement::from_notations("A2", "A7"), -1.0, Default::default())]);

            tree.submit_node_children(1, &[
                (Movement::from_notations("A3", "A7"), 1.0, Default::default()),
                (Movement::from_notations("A4", "A7"), -1.0, Default::default())]);

            tree.submit_node_children(2, &[
                (Movement::from_notations("A5", "A7"), 1.0, Default::default()),
                (Movement::from_notations("A6", "A7"), -1.0, Default::default())]);
        }

        let mut tree = DecisionTree::new(true);
        build(&mut tree);

        let path: Vec<_> = tree.best_path_moves().copied().collect();
        assert_eq!(path, vec![Movement::from_notations("A1", "A7"), Movement::from_notations("A4", "A7")]);

        let mut tree = DecisionTree::new(false);
        build(&mut tree);

        let path: Vec<_> = tree.best_path_moves().copied().collect();
        assert_eq!(path, vec![Movement::from_notations("A2", "A7"), Movement::from_notations("A5", "A7")]);
    }

    #[test]
    fn test_get_unexplored_node() {
        let mut tree = DecisionTree::new(true);
        tree.submit_node_children(0, &[
            (Movement::from_notations("A1", "A7"), 0.1, Default::default()),
            (Movement::from_notations("A2", "A7"), 0.0, Default::default()),
        ]);

        tree.submit_node_children(1, &[
            (Movement::from_notations("A3", "A7"), 0.1, Default::default()),
            (Movement::from_notations("A4", "A7"), 0.3, Default::default()),
        ]);

        assert_eq!(tree.get_best_path_variant(), Some(2));

        tree.submit_node_children(2, &[
            (Movement::from_notations("A5", "A7"), -5.0, Default::default()),
        ]);

        assert_eq!(tree.get_best_path_variant(), Some(4));
    }
}

