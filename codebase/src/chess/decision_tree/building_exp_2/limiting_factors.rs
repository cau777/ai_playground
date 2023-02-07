use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct LimiterFactors {
    pub max_explored_nodes: Option<usize>,
    pub max_full_paths_explored: Option<usize>,
    pub max_iterations: Option<usize>,
}

impl Default for LimiterFactors {
    fn default() -> Self {
        Self {
            max_explored_nodes: None,
            max_iterations: None,
            max_full_paths_explored: None,
        }
    }
}

impl LimiterFactors {
    pub fn should_continue(&self, current: CurrentLimitingFactors) -> bool {
        (self.max_explored_nodes.is_none() || self.max_explored_nodes.unwrap() > current.curr_explored_nodes) &&
            (self.max_iterations.is_none() || self.max_iterations.unwrap() > current.curr_iterations) &&
            (self.max_full_paths_explored.is_none() || self.max_full_paths_explored.unwrap() > current.curr_full_paths_explored) &&

            // If none of the values are set, it will only run once to avoid infinite loops
            (self.max_explored_nodes.is_some() || self.max_iterations.is_some() || self.max_full_paths_explored.is_some() || current.curr_iterations == 0)
    }
}

pub type SyncLimitingFactors = Arc<RwLock<CurrentLimitingFactors>>;

#[derive(Default, Clone)]
pub struct CurrentLimitingFactors {
    pub curr_explored_nodes: usize,
    pub curr_full_paths_explored: usize,
    pub curr_iterations: usize,
}

impl CurrentLimitingFactors {
    pub fn into_sync(self) -> SyncLimitingFactors {
        Arc::new(RwLock::new(self))
    }

    pub fn from_sync(v: SyncLimitingFactors) -> Self {
        v.read().unwrap().clone()
    }
}