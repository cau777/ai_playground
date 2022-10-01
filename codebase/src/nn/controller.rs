use ndarray::{Axis, stack};
use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{backward_layer, BackwardData, forward_layer, ForwardData, GenericStorage, init_layer, InitData, Layer, LayerError, train_layer, TrainData};
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad, LossFunc};
use crate::nn::loss::mse_loss::MseLossFunc;
use crate::utils::ArrayDynF;

pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    epoch: u32,
}

impl NNController {
    pub fn new(main_layer: Layer) -> Result<Self, LayerError> {
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

    pub fn train_batch(&mut self, inputs: ArrayDynF, expected: &ArrayDynF, loss: &LossFunc) -> Result<f32, LayerError> {
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

        assigner.reset_keys();

        let mut backward_cache = GenericStorage::new();
        let grad = calc_loss_grad(loss, expected, &output);
        let loss_mean = calc_loss(loss, expected, &output).mean().unwrap();
        backward_layer(&self.main_layer, BackwardData {
            grad,
            batch_config: &config,
            backward_cache: &mut backward_cache,
            forward_cache: &mut forward_cache,
            storage: &mut self.storage,
            assigner: &mut assigner
        })?;

        assigner.reset_keys();

        train_layer(&self.main_layer, TrainData {
            storage: &mut self.storage,
            batch_config: &config,
            assigner: &mut assigner,
            backward_cache: &mut backward_cache
        })?;

        self.epoch += 1;
        Ok(loss_mean)
    }
}




