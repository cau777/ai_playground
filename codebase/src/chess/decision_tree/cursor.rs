use std::cmp::Ordering;
use crate::chess::board_controller::BoardController;
use crate::chess::decision_tree::node::Node;

#[derive(Clone)]
pub struct TreeCursor {
    current: usize,
    controller: BoardController,
}

impl TreeCursor {
    pub fn new(controller: BoardController) -> Self {
        Self {
            controller,
            current: 0,
        }
    }

    pub fn get_controller(&self) -> &BoardController {
        &self.controller
    }

    pub fn go_to(&mut self, mut target: usize, nodes: &[Node]) {
        if target == self.current {
            return;
        }
        self.go_to_common_point(target, nodes);

        let current = self.current;
        let mut path = Vec::new();
        let mut i = 0;
        while target != current {
            path.push(target);
            target = nodes[target].parent;

            i += 1;
            if i > 1_000_000 {
                panic!("Too many loops in go_to");
            }
        }

        for step in path.iter().rev().copied() {
            self.controller.apply_move(nodes[step].movement);
            self.current = step;
        }
    }

    fn go_to_common_point(&mut self, mut target: usize, nodes: &[Node]) {
        let mut i = 0;
        while self.current != target {
            let current_node = &nodes[self.current];
            let target_node = &nodes[target];

            match current_node.depth.cmp(&target_node.depth) {
                Ordering::Greater => {
                    self.go_back(nodes);
                }
                Ordering::Less => {
                    target = target_node.parent;
                }
                Ordering::Equal => {
                    self.go_back(nodes);
                    target = target_node.parent;
                }
            }

            i += 1;
            if i > 1_000_000 {
                panic!("Too many loops in go_to_common_point");
            }
        }
    }

    fn go_back(&mut self, nodes: &[Node]) {
        self.controller.revert();
        self.current = nodes[self.current].parent;
    }
}


#[cfg(test)]
mod tests {
    use crate::chess::board::Board;
    use crate::chess::decision_tree::DecisionTree;
    use crate::chess::movement::Movement;
    use super::*;

    fn build_tree() -> Vec<Node> {
        let mut tree = DecisionTree::new(true);
        tree.submit_node_children(0, &[
            (Movement::from_notations("A2", "A4"), 2.0, Default::default()),
            (Movement::from_notations("B2", "B4"), 1.0, Default::default()),
        ]);

        tree.submit_node_children(1, &[
            (Movement::from_notations("A7", "A5"), -1.0, Default::default()),
            (Movement::from_notations("B7", "B5"), 0.0, Default::default()),
        ]);

        tree.submit_node_children(3, &[
            (Movement::from_notations("C2", "C4"), 2.0, Default::default()),
            (Movement::from_notations("D2", "D4"), 1.0, Default::default()),
        ]);

        tree.submit_node_children(4, &[
            (Movement::from_notations("E2", "E4"), 2.0, Default::default()),
            (Movement::from_notations("F2", "F4"), 1.0, Default::default()),
        ]);

        tree.nodes
    }

    #[test]
    fn test_go_to_common_point() {
        let nodes = &build_tree();
        let mut cursor = TreeCursor::new(BoardController::new_start());
        cursor.go_to(7, nodes);
        cursor.go_to_common_point(8, nodes);
        assert_eq!(cursor.current, 4);

        cursor.go_to(7, nodes);
        cursor.go_to_common_point(2, nodes);
        assert_eq!(cursor.current, 0);

        cursor.go_to(5, nodes);
        cursor.go_to_common_point(7, nodes);
        assert_eq!(cursor.current, 1);
    }

    #[test]
    fn test_go_to() {
        let nodes = &build_tree();

        let mut cursor = TreeCursor::new(BoardController::new_start());
        cursor.go_to(7, nodes);
        assert_eq!(cursor.controller.current(), &Board::from_literal("\
        r n b q k b n r
        p _ p p p p p p
        _ _ _ _ _ _ _ _
        _ p _ _ _ _ _ _
        P _ _ _ P _ _ _
        _ _ _ _ _ _ _ _
        _ P P P _ P P P
        R N B Q K B N R"));

        cursor.go_to(5, nodes);
        assert_eq!(cursor.controller.current(), &Board::from_literal("\
        r n b q k b n r
        _ p p p p p p p
        _ _ _ _ _ _ _ _
        p _ _ _ _ _ _ _
        P _ P _ _ _ _ _
        _ _ _ _ _ _ _ _
        _ P _ P P P P P
        R N B Q K B N R"));

        cursor.go_to(6, nodes);
        assert_eq!(cursor.controller.current(), &Board::from_literal("\
        r n b q k b n r
        _ p p p p p p p
        _ _ _ _ _ _ _ _
        p _ _ _ _ _ _ _
        P _ _ P _ _ _ _
        _ _ _ _ _ _ _ _
        _ P P _ P P P P
        R N B Q K B N R "));
    }
}