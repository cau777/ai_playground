use crate::chess::movement::Movement;

#[derive(Debug)]
pub struct Node {
    pub parent: usize,
    pub movement: Movement,
    pub eval: f32,
    pub depth: usize,
    pub info: NodeExtraInfo,

    // This array will always be sorted in ascending order based on the node's evals
    pub(crate) children: Option<Vec<usize>>,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct NodeExtraInfo {
    pub is_ending: bool,
    pub is_opening: bool,
}

impl Node {
    pub fn new_empty(movement: Movement, depth: usize, children: Option<Vec<usize>>) -> Self {
        Self {
            parent: usize::MAX, // This is just to throw errors when the parent is used
            movement,
            eval: 0.0,
            depth,
            children,
            info: NodeExtraInfo::default(),
        }
    }

    pub fn new(parent: usize, movement: Movement, eval: f32, depth: usize, children: Option<Vec<usize>>, info: NodeExtraInfo) -> Self {
        Self {
            parent,
            movement,
            eval,
            info,
            depth,
            children,
        }
    }

    pub fn get_ordered_children(&self, start_side: bool) -> Option<impl Iterator<Item=&usize>> {
        let side = self.get_current_side(start_side);
        self.children.as_ref().map(|children| {
            (0..children.len())
                .into_iter()
                // Reverse the iterator if white is playing
                .map(move |o| if side { children.len() - 1 - o } else { o })
                .map(|o| &children[o])
        })
    }

    pub fn refresh_children(&self, mut children: Vec<usize>, all_nodes: &[Node], start_side: bool) -> (Vec<usize>, f32) {
        children.sort_unstable_by(|a, b| all_nodes[*a].eval.total_cmp(&all_nodes[*b].eval));

        let best_node = if self.get_current_side(start_side) {
            *children.last().unwrap()
        } else {
            children[0]
        };
        (children, all_nodes[best_node].eval)
    }

    pub fn apply_children(&mut self, children: Vec<usize>, eval: f32) {
        self.children = Some(children);
        self.eval = eval;
    }

    pub fn clone_children(&self) -> Option<Vec<usize>> {
        self.children.clone()
    }

    /// Get the current side based on the node depth in the tree and the start side
    pub fn get_current_side(&self, start_side: bool) -> bool {
        // == is used for the following truth table:
        // | T T = T |
        // | T F = F |
        // | F T = F |
        // | F F = T |
        (self.depth % 2 == 0) == start_side
    }
}
