use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{
    backward_layer, forward_layer, init_layer, train_layer, BackwardData, ForwardData,
    GenericStorage, InitData, Layer, TrainData,
};
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad, LossFunc};
use crate::utils::{ArrayDynF, GenericResult};
use ndarray::{stack, Axis};
use crate::gpu::shader_runner::{GlobalGpu, GpuData};

/// Main struct to train and use the AI model
/// ```
/// use codebase::nn::controller::NNController;
/// use codebase::nn::train_config::TrainConfig;
///
/// let mut controller = NNController::new(main_layer, loss_func).unwrap();
///
/// println!("Started training");
/// for epoch in 0..10 {
///     // It's better to split the data into random batches
///     let loss = controller.train_batch(data_inputs.clone(), &data_expected ).unwrap();
///     println!("Epoch {} finished with avg loss {}", epoch, loss);
/// }
///
/// println!("Finished training. Started validating");
/// let final_loss = controller.test_batch(data_inputs, &data_expected).unwrap();
/// println!("Validated loss = {}", final_loss);
///
/// ```
pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    loss: LossFunc,
}

impl NNController {
    /// Create a controller with an empty storage and init its layers
    pub fn new(main_layer: Layer, loss: LossFunc) -> GenericResult<Self> {
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

    /// Create a controller with the provided storage and init its layers
    pub fn load(main_layer: Layer, loss: LossFunc, mut storage: GenericStorage) -> GenericResult<Self> {
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

    /// The same as eval_batch except without the "batch" dimension in the input and the output
    pub fn eval_one(&self, inputs: ArrayDynF) -> GenericResult<ArrayDynF> {
        self.eval_batch(stack![Axis(0), inputs])
            .map(|o| o.remove_axis(Axis(0)))
    }

    /// Forward the input through the layers and return the result.
    /// Uses GPU if available
    pub fn eval_batch(&self, inputs: ArrayDynF) -> GenericResult<ArrayDynF> {
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

    /// The same as train_batch except without the "batch" dimension in the input and the output
    pub fn train_one(&mut self, inputs: ArrayDynF, expected: ArrayDynF) -> GenericResult<f64> {
        self.train_batch(
            stack![Axis(0), inputs],
            &stack![Axis(0), expected],
        )
    }

    /// Execute the following steps to train the model based on **inputs** and the corresponding labels
    /// 1) Evaluate the model output for the given inputs (forward propagation)
    /// 2) Calculate the loss between the output and **expected**
    /// 3) Calculate the gradient of that loss
    /// 4) Use gradient descent to find the gradients of all parameters in all layers (backwards propagation)
    /// 5) Update all parameters with those gradients
    /// #####
    /// Uses GPU if available.
    /// Returns the average loss in the batch
    pub fn train_batch(&mut self, inputs: ArrayDynF, expected: &ArrayDynF) -> GenericResult<f64> {
        let config = BatchConfig::new_train();
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

    /// The same as test_batch except without the "batch" dimension in the input and the output
    pub fn test_one(&self, inputs: ArrayDynF, expected: ArrayDynF) -> GenericResult<f64> {
        self.test_batch(stack![Axis(0), inputs], &stack![Axis(0), expected])
    }

    /// Calculate the loss between **expected** and the result of the forward propagation of **inputs**.
    /// Uses GPU if available
    pub fn test_batch(&self, inputs: ArrayDynF, expected: &ArrayDynF) -> GenericResult<f64> {
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

    /// Return a copy of the inner storage
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

    use crate::{
        integration::layers_loading::load_model_xml,
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
            )
            .unwrap();
        println!("{}", result);
    }
}
