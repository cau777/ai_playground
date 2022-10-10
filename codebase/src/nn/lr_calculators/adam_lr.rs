use crate::nn::layers::nn_layers::LayerResult;
use crate::nn::lr_calculators::lr_calculator::{LrCalcData, LrCalcOps};
use crate::utils::{ArrayDynF, lerp_arrays};

#[derive(Clone)]
pub struct AdamConfig {
    pub alpha: f32,
    pub decay1: f32,
    pub decay2: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            alpha: 0.02,
            decay1: 0.9,
            decay2: 0.999,
        }
    }
}

pub struct AdamLrCalc {}

impl LrCalcOps<AdamConfig> for AdamLrCalc {
    fn apply(target: ArrayDynF, data: LrCalcData, config: &AdamConfig) -> LayerResult {
        let LrCalcData {storage, assigner,..} = data;
        let key = assigner.get_key("adam".to_owned());

        let mut moment1: ArrayDynF;
        let mut moment2: ArrayDynF;
        match storage.remove(&key) {
            Some(mut v) => {
                moment1 = v.remove(0);
                moment2 = v.remove(0);
            }
            None => {
                moment1 = ArrayDynF::zeros(target.shape());
                moment2 = ArrayDynF::zeros(target.shape());
            }
        }

        moment1 = lerp_arrays(&target, &moment1, config.decay1);
        moment2 = lerp_arrays(&(&target * &target), &moment2, config.decay2);

        let epoch = data.batch_config.epoch as i32;
        let moment1b = &moment1 / (1.0 - (config.decay1.powi(epoch)));
        let moment2b = &moment2 / (1.0 - (config.decay2.powi(epoch)));

        storage.insert(key, vec![moment1, moment2]);

        Ok(config.alpha * moment1b / (&moment2b * &moment2b + f32::EPSILON))
    }
}