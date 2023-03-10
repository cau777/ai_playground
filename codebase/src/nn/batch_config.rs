/// Simple struct that contains some information about the current batch
pub struct BatchConfig {
    // pub max_batch_size: usize,
    pub is_training: bool,
}

impl BatchConfig {
    pub fn new_not_train() -> Self {
        Self { is_training: false }
    }

    pub fn new_train() -> Self {
        Self { is_training: true }
    }
}
