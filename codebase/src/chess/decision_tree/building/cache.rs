use std::collections::HashMap;
use std::mem;
use crate::chess::decision_tree::DecisionTree;
use crate::nn::layers::nn_layers::GenericStorage;

/// Stores the values inserted in prev_iteration_cache. It includes functionality to remove
/// items when it exceeds a number of bytes. Items are removed in the order they were added
#[derive(Clone)]
pub struct Cache {
    buffer: HashMap<usize, GenericStorage>,
    current_bytes: u64,
    max_bytes: u64,
    last_removed: usize,
    // The node index that the next inserted cache will refer to.
    current_index: usize,
}

impl Cache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            buffer: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            last_removed: 0,
            current_index: 1,
        }
    }

    pub fn get(&self, index: usize) -> Option<&GenericStorage> {
        self.buffer.get(&index)
    }

    pub fn insert_next(&mut self, value: Option<GenericStorage>) {
        if let Some(value) = value {
            self.current_bytes += Self::count_bytes(&value);
            self.buffer.insert(self.current_index, value);
        }

        self.current_index += 1;
    }

    pub fn remove(&mut self, index: usize) {
        if let Some(value) = self.buffer.get(&index) {
            self.current_bytes -= Self::count_bytes(value);
        }
        self.buffer.remove(&index);
    }

    fn count_bytes(storage: &GenericStorage) -> u64 {
        const ITEM_SIZE: u64 = mem::size_of::<f32>() as u64;

        storage.values()
            .flatten()
            // Add 16 to account for some pointer sizes
            .map(|o| o.len() as u64 * ITEM_SIZE + 16)
            .sum()
    }

    pub fn remove_excess(&mut self) {
        while self.should_remove() && self.last_removed < self.buffer.len() {
            self.remove(self.last_removed);
            self.last_removed += 1;
        }
    }

    fn should_remove(&self) -> bool {
        self.current_bytes >= self.max_bytes
    }
}