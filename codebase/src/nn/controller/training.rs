use ndarray::{stack, Axis};
use crate::ArrayDynF;
use crate::nn::batch_config::BatchConfig;
use crate::nn::controller::NNController;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::*;
use crate::nn::loss::loss_func::{calc_loss, calc_loss_grad};
use crate::utils::GenericResult;

impl NNController {
    /// The same as train_batch except without the "batch" dimension in the input
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
        let gpu = self.get_gpu();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs: inputs.into(),
                assigner: &mut assigner,
                storage: &mut self.storage,
                forward_cache: Some(&mut forward_cache),
                batch_config: &config,
                gpu: gpu.clone(),
                prev_iteration_cache: None,
            },
        )?.into_memory()?;

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
}