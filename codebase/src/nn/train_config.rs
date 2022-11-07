pub struct TrainConfig {
    pub workers: u32
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            workers: 1,
        }
    }
}
