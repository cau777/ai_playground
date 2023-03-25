use ndarray::{stack, Axis};
use crate::ArrayDynF;
use crate::nn::batch_config::BatchConfig;
use crate::nn::controller::NNController;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{forward_layer, ForwardData, GenericStorage};
use crate::nn::loss::loss_func::calc_loss;
use crate::utils::GenericResult;

impl NNController {
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
        let gpu = self.get_gpu();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs: inputs.into(),
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: Some(&mut forward_cache),
                batch_config: &config,
                prev_iteration_cache: None,
                gpu
            },
        )?.into_memory()?;

        let loss_mean = calc_loss(&self.loss, expected, &output)
            .mapv(|o| o as f64)
            .mean()
            .unwrap();
        self.finish_method()?;
        Ok(loss_mean)
    }
}