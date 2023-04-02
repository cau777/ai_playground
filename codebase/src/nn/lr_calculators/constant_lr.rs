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
            lr: 0.05
        }
    }
}

/// The simples possible "optimizer". It just multiplies the gradients by a small constant.
pub struct ConstantLr;

impl LrCalcOps<ConstantLrConfig> for ConstantLr {
    fn apply(target: ArrayDynF, _: LrCalcData, config: &ConstantLrConfig) -> LayerResult {
        Ok((target * config.lr).into())
    }
}