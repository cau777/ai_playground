use crate::nn::lr_calculators::lr_calculator::{LrCalcData, LrCalcOps};
use crate::utils::ArrayDynF;

#[derive(Clone)]
pub struct ConstantLrConfig {
    lr: f32,
}

impl Default for ConstantLrConfig {
    fn default() -> Self {
        Self {
            lr: 0.05
        }
    }
}

pub struct ConstantLr {}

impl LrCalcOps<ConstantLrConfig> for ConstantLr {
    fn apply(target: ArrayDynF, _: LrCalcData, config: &ConstantLrConfig) -> ArrayDynF {
        target * config.lr
    }
}