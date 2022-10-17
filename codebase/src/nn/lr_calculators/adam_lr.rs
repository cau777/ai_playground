use crate::nn::layers::nn_layers::LayerResult;
use crate::nn::lr_calculators::lr_calculator::{LrCalcData, LrCalcOps};
use crate::utils::{ArrayDynF, lerp_arrays, Array0F};
use crate::nn::generic_storage::remove_from_storage3;

#[derive(Clone, Debug)]
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
    // TODO: updating epoch twice
    fn apply(target: ArrayDynF, data: LrCalcData, config: &AdamConfig) -> LayerResult {
        let LrCalcData {storage, assigner,..} = data;
        let key = assigner.get_key("adam".to_owned());

        let mut moment1: ArrayDynF;
        let mut moment2: ArrayDynF;
        let epoch: f32;
        match storage.remove(&key) {
            Some(mut v) => {
                moment1 = v.remove(0);
                moment2 = v.remove(0);
                epoch = *v.remove(0).first().unwrap();
            }
            None => {
                moment1 = ArrayDynF::zeros(target.shape());
                moment2 = ArrayDynF::zeros(target.shape());
                epoch = 1.0;
            }
        }

        moment1 = lerp_arrays(&target, &moment1, config.decay1);
        moment2 = lerp_arrays(&(&target * &target), &moment2, config.decay2);
        
        let moment1b = &moment1 / (1.0 - (config.decay1.powf(epoch)));
        let moment2b = &moment2 / (1.0 - (config.decay2.powf(epoch)));
        
        storage.insert(key, vec![moment1, moment2, Array0F::from_elem((), epoch + 1.0).into_dyn()]);

        Ok(config.alpha * moment1b / (&moment2b * &moment2b + f32::EPSILON))
    }
}