use super::train_config::TrainConfig;

pub struct BatchConfig {
    pub train_config: Option<TrainConfig>,
}

impl BatchConfig {
    pub fn new_not_train() -> Self {
        Self { train_config: None }
    }

    pub fn new_train(train_config: TrainConfig) -> Self {
        Self { train_config: Some(train_config) }
    }
}
