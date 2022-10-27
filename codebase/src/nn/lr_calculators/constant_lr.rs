use crate::nn::layers::nn_layers::LayerResult;
use crate::nn::lr_calculators::lr_calculator::{LrCalcData, LrCalcOps};
use crate::utils::ArrayDynF;

#[derive(Clone, Debug)]
pub struct ConstantLrConfig {
    pub lr: f32,
}

impl Default for ConstantLrConfig {
    fn default() -> Self {
        Self {
            lr: 0.005
        }
    }
}

pub struct ConstantLr {}

// TODO: exploding numbers in first epochs for giant networks
impl LrCalcOps<ConstantLrConfig> for ConstantLr {
    fn apply(target: ArrayDynF, _: LrCalcData, config: &ConstantLrConfig) -> LayerResult {
        Ok(target * config.lr)
    }
}