mod dense_forward;
mod dense_backward;

use crate::nn::generic_storage::*;
use crate::nn::layers::nn_layers::*;
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};
use crate::utils::{Array1F, Array2F, GetBatchSize};
use ndarray::{Axis, ShapeBuilder, stack};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::ops::{AddAssign};
use ndarray::parallel::prelude::*;

#[derive(Clone, Debug)]
pub struct DenseConfig {
    pub out_values: usize,
    pub in_values: usize,
    pub init_mode: DenseLayerInit,
    pub weights_lr_calc: LrCalc,
    pub biases_lr_calc: LrCalc,
}

#[derive(Clone, Debug)]
pub enum DenseLayerInit {
    WeightsAndBiases(Array2F, Array1F),
    Random(),
}

pub struct DenseLayer {}

fn gen_name(config: &DenseConfig) -> String {
    format!("dense_{}_{}", config.in_values, config.out_values)
}

impl LayerOps<DenseConfig> for DenseLayer {
    fn init(data: InitData, layer_config: &DenseConfig) -> EmptyLayerResult {
        let InitData { assigner, storage } = data;
        let key = assigner.get_key(gen_name(layer_config));

        if let std::collections::hash_map::Entry::Vacant(e) = storage.entry(key) {
            let weights: Array2F;
            let biases: Array1F;

            match &layer_config.init_mode {
                DenseLayerInit::WeightsAndBiases(w, b) => {
                    weights = w.clone();
                    biases = b.clone();
                }
                DenseLayerInit::Random() => {
                    let std_dev = (layer_config.out_values as f32).powf(-0.5);
                    let dist = Normal::new(0.0, std_dev)?;
                    weights = Array2F::random(
                        (layer_config.out_values, layer_config.in_values).f(),
                        dist,
                    );
                    biases = Array1F::zeros((layer_config.out_values).f());
                }
            }

            e.insert(vec![weights.into_dyn(), biases.into_dyn()]);
        }

        Ok(())
    }

    #[inline(never)]
    fn forward(data: ForwardData, layer_config: &DenseConfig) -> LayerResult {
        dense_forward::forward(data, layer_config)
    }

    fn backward(data: BackwardData, layer_config: &DenseConfig) -> LayerResult {
        dense_backward::backward(data, layer_config)
    }
}

impl TrainableLayerOps<DenseConfig> for DenseLayer {
    fn train(data: TrainData, layer_config: &DenseConfig) -> EmptyLayerResult {
        let TrainData {
            backward_cache,
            assigner,
            storage,
            batch_config,
            ..
        } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let [weights_grad, biases_grad] = remove_from_storage2(backward_cache, &key);

        let weights_grad = apply_lr_calc(
            &layer_config.weights_lr_calc,
            weights_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?.into_memory().unwrap();

        let biases_grad = apply_lr_calc(
            &layer_config.biases_lr_calc,
            biases_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?.into_memory().unwrap();

        get_mut_from_storage(storage, &key, 0).add_assign(&weights_grad);
        get_mut_from_storage(storage, &key, 1).add_assign(&biases_grad);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::controller::NNController;
    use crate::nn::layers::nn_layers::*;
    use crate::nn::loss::loss_func::LossFunc;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{Array2F};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::nn::layers::dense_layer::{DenseConfig, DenseLayerInit};

    #[test]
    fn test_train() {
        let mut controller = NNController::new(
            Layer::Dense(DenseConfig {
                in_values: 12,
                out_values: 10,
                init_mode: DenseLayerInit::Random(),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            }),
            LossFunc::Mse,
        )
            .unwrap();
        let inputs = Array2F::random((2, 12), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let expected = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let mut last_loss = 0.0;
        let mut first_loss = None;

        for _ in 0..100 {
            let inputs = inputs.clone();
            last_loss = controller.train_batch(inputs, &expected).unwrap();
            if first_loss.is_none() {
                first_loss = Some(last_loss);
            }
            println!("{}", last_loss);
        }

        assert!(last_loss < first_loss.unwrap());
    }

    pub(crate) fn get_config(init_mode: DenseLayerInit) -> DenseConfig {
        DenseConfig {
            init_mode,
            in_values: 2,
            out_values: 3,
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        }
    }
}
