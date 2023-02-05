mod options;
mod games_producer;
mod request;
mod nodes_in_progress_set;

use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::DecisionTree;

pub use options::*;

pub struct DecisionTreeBuilder {

}

impl DecisionTreeBuilder {
    pub fn new(initial_trees: Vec<DecisionTree>, initial_cursors: Vec<TreeCursor>, options: BuilderOptions) -> Self {
        unimplemented!()
    }
}