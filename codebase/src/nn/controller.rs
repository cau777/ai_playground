use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{
    backward_layer, forward_layer, init_layer, train_layer, BackwardData, ForwardData,
    GenericStorage, InitData, Layer, LayerError, TrainData,
};
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad, LossFunc};
use crate::utils::ArrayDynF;
use ndarray::{stack, Axis};
use crate::gpu::shader_runner::{GlobalGpu, GpuData};

use super::train_config::TrainConfig;

pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    loss: LossFunc,
}

impl NNController {
    pub fn new(main_layer: Layer, loss: LossFunc) -> Result<Self, LayerError> {
        let mut storage = GenericStorage::new();
        let mut assigner = KeyAssigner::new();
        init_layer(
            &main_layer,
            InitData {
                assigner: &mut assigner,
                storage: &mut storage,
            },
        )?;

        Ok(Self {
            main_layer,
            storage,
            loss,
        })
    }

    pub fn load(main_layer: Layer, loss: LossFunc, mut storage: GenericStorage) -> Result<Self, LayerError> {
        let mut assigner = KeyAssigner::new();
        init_layer(
            &main_layer,
            InitData {
                assigner: &mut assigner,
                storage: &mut storage,
            },
        )?;

        Ok(Self {
            main_layer,
            storage,
            loss,
        })
    }

    pub fn eval_one(&self, inputs: ArrayDynF) -> Result<ArrayDynF, LayerError> {
        self.eval_batch(stack![Axis(0), inputs])
            .map(|o| o.remove_axis(Axis(0)))
    }

    pub fn eval_batch(&self, inputs: ArrayDynF) -> Result<ArrayDynF, LayerError> {
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let config = BatchConfig::new_not_train();
        let gpu=   Self::get_gpu();

        forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
                gpu: gpu.clone(),
            },
        )
    }

    pub fn train_one(&mut self, inputs: ArrayDynF, expected: ArrayDynF, config: TrainConfig) -> Result<f64, LayerError> {
        self.train_batch(
            stack![Axis(0), inputs],
            &stack![Axis(0), expected],
            config,
        )
    }

    pub fn train_batch(&mut self, inputs: ArrayDynF, expected: &ArrayDynF, config: TrainConfig) -> Result<f64, LayerError> {
        let config = BatchConfig::new_train(config);
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let gpu = Self::get_gpu();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &mut self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
                gpu: gpu.clone(),
            },
        )?;

        assigner.revert();

        let mut backward_cache = GenericStorage::new();
        let grad = calc_loss_grad(&self.loss, expected, &output);
        let loss_mean = calc_loss(&self.loss, expected, &output)
            .mapv(|o| o as f64)
            .mean()
            .unwrap();

        backward_layer(
            &self.main_layer,
            BackwardData {
                grad,
                batch_config: &config,
                backward_cache: &mut backward_cache,
                forward_cache: &mut forward_cache,
                storage: &mut self.storage,
                assigner: &mut assigner,
                gpu,
            },
        )?;

        assigner.revert();

        train_layer(
            &self.main_layer,
            TrainData {
                storage: &mut self.storage,
                batch_config: &config,
                assigner: &mut assigner,
                backward_cache: &mut backward_cache,
            },
        )?;

        Ok(loss_mean)
    }

    pub fn test_one(&self, inputs: ArrayDynF, expected: ArrayDynF) -> Result<f64, LayerError> {
        self.test_batch(stack![Axis(0), inputs], &stack![Axis(0), expected])
    }

    pub fn test_batch(&self, inputs: ArrayDynF, expected: &ArrayDynF) -> Result<f64, LayerError> {
        let config = BatchConfig::new_not_train();
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let gpu = Self::get_gpu();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
                gpu
            },
        )?;

        let loss_mean = calc_loss(&self.loss, expected, &output)
            .mapv(|o| o as f64)
            .mean()
            .unwrap();
        Ok(loss_mean)
    }

    pub fn export(&self) -> GenericStorage {
        self.storage.clone()
    }

    fn get_gpu() -> Option<GlobalGpu> {
        match GpuData::new_global() {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{:?}", e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Read};

    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use crate::{
        integration::layers_loading::load_model_xml,
        nn::lr_calculators::{constant_lr::ConstantLrConfig, lr_calculator::LrCalc},
        utils::{Array2F, Array3F},
    };

    use super::*;

    #[test]
    #[ignore]
    fn test_digits_model() {
        let mut file = OpenOptions::new()
            .read(true)
            .open("../temp/digits/config.xml")
            .unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let config = load_model_xml(&bytes).unwrap();
        let mut controller = NNController::new(config.main_layer, config.loss_func).unwrap();
        let result = controller
            .train_batch(
                Array3F::ones((64, 28, 28)).into_dyn(),
                &Array2F::ones((64, 10)).into_dyn(),
                TrainConfig::default(),
            )
            .unwrap();
        println!("{}", result);
    }
}
