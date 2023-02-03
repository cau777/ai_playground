use std::sync::{Arc, RwLock};
use crate::chess::decision_tree::building_exp::NextNodeStrategy;

pub type SyncLimitingFactors = Arc<RwLock<LimitingFactors>>;

#[derive(Default, Clone)]
pub struct LimitingFactors {
    pub explored_nodes: usize,
    pub explored_paths: usize,
    pub iterations: usize,
}

impl LimitingFactors {
    pub fn should_continue(&self, strategy: NextNodeStrategy) -> bool {
        match strategy {
            NextNodeStrategy::ContinueLineThenBestVariant { min_full_paths } => self.explored_paths < min_full_paths,
            NextNodeStrategy::ContinueLineThenBestVariantOrRandom { min_full_paths, .. } => self.explored_paths < min_full_paths,
            NextNodeStrategy::BestNodeAlways { min_nodes_explored } => self.explored_nodes < min_nodes_explored,
            NextNodeStrategy::BestOrRandomNode { min_nodes_explored, .. } => self.explored_nodes < min_nodes_explored,
        }
    }

    pub fn into_sync(self) -> SyncLimitingFactors {
        Arc::new(RwLock::new(self))
    }

    pub fn from_sync(v: SyncLimitingFactors) -> Self {
        v.read().unwrap().clone()
    }
}