use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{
    backward_layer, forward_layer, init_layer, train_layer, BackwardData, ForwardData,
    GenericStorage, InitData, Layer, LayerError, TrainData,
};
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad, LossFunc};
use crate::utils::ArrayDynF;
use ndarray::{stack, Axis};

use super::train_config::TrainConfig;

pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    loss: LossFunc,
    epoch: u32,
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
            epoch: 1,
            loss,
        })
    }

    pub fn load(
        main_layer: Layer,
        loss: LossFunc,
        mut storage: GenericStorage,
    ) -> Result<Self, LayerError> {
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
            epoch: 1,
            loss,
        })
    }

    pub fn eval_batch(&self, inputs: ArrayDynF) -> Result<ArrayDynF, LayerError> {
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let config = BatchConfig::new_not_train();

        forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
            },
        )
    }

    pub fn train_one(
        &mut self,
        inputs: ArrayDynF,
        expected: ArrayDynF,
        config: TrainConfig,
    ) -> Result<f64, LayerError> {
        self.train_batch(
            stack![Axis(0), inputs],
            &stack![Axis(0), expected],
            config,
        )
    }

    pub fn train_batch(
        &mut self,
        inputs: ArrayDynF,
        expected: &ArrayDynF,
        config: TrainConfig,
    ) -> Result<f64, LayerError> {
        let config = BatchConfig::new_train(config);
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &mut self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
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

        self.epoch += 1;
        Ok(loss_mean)
    }

    pub fn test_one(&self, inputs: ArrayDynF, expected: ArrayDynF) -> Result<f64, LayerError> {
        self.test_batch(stack![Axis(0), inputs], &stack![Axis(0), expected])
    }

    pub fn test_batch(&self, inputs: ArrayDynF, expected: &ArrayDynF) -> Result<f64, LayerError> {
        let config = BatchConfig::new_not_train();
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
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
    fn test_complete() {
        let mut file = OpenOptions::new()
            .read(true)
            .open("../config/digits/model.xml")
            .unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let config = load_model_xml(&bytes).unwrap();
        let mut controller = NNController::new(config.main_layer, config.loss_func).unwrap();
        let result = controller
            .train_batch(
                Array3F::ones((256, 28, 28)).into_dyn(),
                &Array2F::ones((256, 10)).into_dyn(),
                TrainConfig::default(),
            )
            .unwrap();
        println!("{}", result);
    }

    #[test]
    fn test_001() {
        use crate::nn::layers::*;
        let d = 40;
        let mut controller = NNController::new(
            Layer::Sequential(sequential_layer::SequentialConfig {
                layers: vec![
                    // Layer::ExpandDim(expand_dim_layer::ExpandDimConfig { dim: 0 }),
                    // Layer::Convolution(convolution_layer::ConvolutionConfig {
                    //     in_channels: 1,
                    //     out_channels: 4,
                    //     kernel_size: 3,
                    //     stride: 1,
                    //     padding: 0,
                    //     init_mode: convolution_layer::ConvolutionInitMode::HeNormal(),
                    //     lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    // }),
                    //Layer::Relu,
                    //Layer::MaxPool(max_pool_layer::MaxPoolConfig { size: 2, stride: 2 }),
                    //Layer::Flatten,
                    //Layer::Dense(dense_layer::DenseConfig {
                    //    in_values: 676,
                    //    out_values: 512,
                    //    init_mode: dense_layer::DenseLayerInit::Random(),
                    //    weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    //    biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    //}),
                    //Layer::Relu,
                    Layer::Dense(dense_layer::DenseConfig {
                        in_values: 512 * d,
                        out_values: 256 * d,
                        init_mode: dense_layer::DenseLayerInit::Random(),
                        weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                        biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    }),
                    Layer::Relu,
                    Layer::Dense(dense_layer::DenseConfig {
                        in_values: 256 * d,
                        out_values: 10 * d,
                        init_mode: dense_layer::DenseLayerInit::Random(),
                        weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                        biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    }),
                ],
            }),
            LossFunc::Mse,
        )
        .unwrap();

        let dist = Normal::new(0.0, 1.0).unwrap();

        let inputs = Array2F::random((1, 512 * d), dist).into_dyn();
        let expected = Array2F::random((1, 10 * d), dist).into_dyn();
        for _ in 0..100 {
            let result = controller.train_batch(inputs.clone(), &expected, TrainConfig::default()).unwrap();
            println!("{}", result);
        }
    }
}
