use ndarray::{Axis, stack};
use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{backward_layer, BackwardData, forward_layer, ForwardData, GenericStorage, init_layer, InitData, Layer, LayerError, train_layer, TrainData};
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad, LossFunc};
use crate::utils::ArrayDynF;

pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    loss: LossFunc,
    epoch: u32,
}


// TODO: Controller assigner backward
impl NNController {
    pub fn new(main_layer: Layer, loss: LossFunc) -> Result<Self, LayerError> {
        let mut storage = GenericStorage::new();
        let mut assigner = KeyAssigner::new();
        init_layer(&main_layer, InitData {
            assigner: &mut assigner,
            storage: &mut storage,
        })?;

        Ok(Self {
            main_layer,
            storage,
            epoch: 1,
            loss
        })
    }

    pub fn load(main_layer: Layer, loss: LossFunc, mut storage: GenericStorage) -> Result<Self, LayerError> {
        let mut assigner = KeyAssigner::new();
        init_layer(&main_layer, InitData {
            assigner: &mut assigner,
            storage: &mut storage,
        })?;

        Ok(Self {
            main_layer,
            storage,
            epoch: 1,
            loss
        })
    }

    pub fn eval_one(&mut self, inputs: ArrayDynF) -> Result<ArrayDynF, LayerError> {
        self.eval_batch(stack![Axis(0), inputs]).map(|o| o.remove_axis(Axis(0)))
    }

    pub fn eval_batch(&mut self, inputs: ArrayDynF) -> Result<ArrayDynF, LayerError> {
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let config = BatchConfig {
            epoch: self.epoch
        };

        forward_layer(&self.main_layer, ForwardData {
            inputs,
            assigner: &mut assigner,
            storage: &mut self.storage,
            forward_cache: &mut forward_cache,
            batch_config: &config,
        })
    }

    pub fn train_one(&mut self, inputs: ArrayDynF, expected: ArrayDynF) -> Result<f32, LayerError> {
        self.train_batch(stack![Axis(0), inputs], &stack![Axis(0), expected])
    }

    pub fn train_batch(&mut self, inputs: ArrayDynF, expected: &ArrayDynF) -> Result<f32, LayerError> {
        let config = BatchConfig {
            epoch: self.epoch
        };
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();

        let output = forward_layer(&self.main_layer, ForwardData {
            inputs,
            assigner: &mut assigner,
            storage: &mut self.storage,
            forward_cache: &mut forward_cache,
            batch_config: &config,
        })?;

        assigner.revert();

        let mut backward_cache = GenericStorage::new();
        let grad = calc_loss_grad(&self.loss, expected, &output);
        let loss_mean = calc_loss(&self.loss, expected, &output).mean().unwrap();
        
        backward_layer(&self.main_layer, BackwardData {
            grad,
            batch_config: &config,
            backward_cache: &mut backward_cache,
            forward_cache: &mut forward_cache,
            storage: &mut self.storage,
            assigner: &mut assigner
        })?;

        assigner.revert();
        
        train_layer(&self.main_layer, TrainData {
            storage: &mut self.storage,
            batch_config: &config,
            assigner: &mut assigner,
            backward_cache: &mut backward_cache
        })?;

        self.epoch += 1;
        Ok(loss_mean)
    }

    pub fn test_one(&self, inputs: ArrayDynF, expected: ArrayDynF)-> Result<f32, LayerError> {
        self.test_batch(stack![Axis(0), inputs], &stack![Axis(0), expected])
    }

    pub fn test_batch(&self, inputs: ArrayDynF, expected: &ArrayDynF)-> Result<f32, LayerError> {
        let config = BatchConfig {
            epoch: self.epoch
        };
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();

        let output = forward_layer(&self.main_layer, ForwardData {
            inputs,
            assigner: &mut assigner,
            storage: &self.storage,
            forward_cache: &mut forward_cache,
            batch_config: &config,
        })?;

        let loss_mean = calc_loss(&self.loss, expected, &output).mean().unwrap();

        Ok(loss_mean)
    }

    pub fn export(&self) -> GenericStorage {
        self.storage.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Read};

    use crate::{integration::layers_loading::load_model_xml, utils::{Array2F, Array3F}};

    use super::*;
    
    #[test]
    fn test_complete() {
        let mut file = OpenOptions::new().read(true).open("../config/digits/model.xml").unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let config = load_model_xml(&bytes).unwrap();
        let mut controller = NNController::new(config.main_layer, config.loss_func).unwrap();
        let result = controller.train_batch(Array3F::ones((256, 28, 28)).into_dyn(), &Array2F::ones((256, 10)).into_dyn()).unwrap();
        println!("{}", result);
    }
}


