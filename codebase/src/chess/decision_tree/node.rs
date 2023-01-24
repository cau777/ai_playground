use crate::chess::movement::Movement;

#[derive(Debug, Clone)]
pub struct Node {
    pub parent: usize,
    pub movement: Movement,
    pub pre_eval: f32,
    pub children_eval: Option<f32>,
    // pub eval: f32,
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
            pre_eval: 0.0,
            depth,
            children,
            info: NodeExtraInfo::default(),
            children_eval: None,
        }
    }

    pub fn new(parent: usize, movement: Movement, eval: f32, depth: usize, children: Option<Vec<usize>>, info: NodeExtraInfo) -> Self {
        Self {
            parent,
            movement,
            pre_eval: eval,
            info,
            depth,
            children,
            children_eval: None,
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

    pub fn refresh_children(&mut self, all_nodes: &[Node], start_side: bool) {
        let side = self.get_current_side(start_side);
        if let Some(children) = self.children.as_mut() {
            children.sort_unstable_by(|&a, &b| all_nodes[a].eval().total_cmp(&all_nodes[b].eval()));
            let best_node = if side {
                *children.last().unwrap()
            } else {
                children[0]
            };

            self.children_eval = Some(all_nodes[best_node].eval());
        }
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

    #[inline]
    pub fn eval(&self) -> f32 {
        self.children_eval.unwrap_or(self.pre_eval)
    }
}
