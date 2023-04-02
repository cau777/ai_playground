use crate::chess::decision_tree::node::Node;

/// Iterator that follows the best moves
pub struct BestPathIterator<'a> {
    current: Option<usize>,
    nodes: &'a Vec<Node>,
    start_side: bool,
    yield_root: bool,
}

impl<'a> BestPathIterator<'a> {
    pub fn new(nodes: &'a Vec<Node>, start_side: bool, yield_root: bool) -> Self {
        Self {
            current: Some(0),
            nodes,
            start_side,
            yield_root,
        }
    }
}

impl<'a> Iterator for BestPathIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yield_root {
            self.yield_root = false;
            return Some(0);
        }
        match self.current {
            Some(current) => {
                self.current = self.nodes[current].get_ordered_children(self.start_side)
                    .and_then(|mut o| o.next())
                    .copied();
                self.current
            }
            None => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::movement::Movement;
    use super::*;

    #[test]
    fn test_tree() {
        let nodes = vec![
            Node::new_empty(Movement::from_notations("A1", "A2"), 0, Some(vec![2, 1])),
            Node::new_empty(Movement::from_notations("A2", "A2"), 1, Some(vec![4, 3])),
            Node::new_empty(Movement::from_notations("A3", "A2"), 1, None),
            Node::new_empty(Movement::from_notations("A4", "A2"), 2, None),
            Node::new_empty(Movement::from_notations("A5", "A2"), 2, None),
        ];

        let moves: Vec<_> = BestPathIterator::new(&nodes, true, false).collect();
        assert_eq!(moves, vec![1, 4]);

        let moves: Vec<_> = BestPathIterator::new(&nodes, false, true).collect();
        assert_eq!(moves, vec![0, 2]);
    }
}
