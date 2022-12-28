use ndarray::{Axis, stack};
use crate::ArrayDynF;
use crate::nn::batch_config::BatchConfig;
use crate::nn::controller::NNController;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{forward_layer, ForwardData, GenericStorage};
use crate::utils::GenericResult;

pub struct FullEvalOutput {
    pub output: ArrayDynF,
    pub forward_cache: GenericStorage,
    pub assigner: KeyAssigner,
}

impl NNController {
    pub fn eval_for_train(&self, inputs: ArrayDynF) -> GenericResult<FullEvalOutput> {
        let config = BatchConfig::new_train();
        let mut assigner = KeyAssigner::new();
        let mut forward_cache = GenericStorage::new();
        let gpu = self.get_gpu();

        let output = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
                gpu: gpu.clone(),
            },
        )?;

        assigner.revert();

        Ok(FullEvalOutput {
            assigner,
            forward_cache,
            output,
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
        let gpu = self.get_gpu();

        forward_layer(
            &self.main_layer,
            ForwardData {
                inputs,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: &mut forward_cache,
                batch_config: &config,
                gpu,
            },
        )
    }
}