use std::sync::Arc;
use ndarray::{Axis, stack};
use crate::ArrayDynF;
use crate::gpu::buffers::upload_array_to_gpu;
use crate::gpu::gpu_data::GlobalGpu;
use crate::nn::batch_config::BatchConfig;
use crate::nn::controller::NNController;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{forward_layer, ForwardData, GenericStorage};
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::GenericResult;

fn prepare_inputs(inputs: ArrayDynF, gpu: Option<GlobalGpu>) -> GenericResult<StoredArray> {
    Ok(match gpu {
        Some(gpu) => {
            StoredArray::GpuLocal {
                shape: inputs.shape().to_vec(),
                data: upload_array_to_gpu(&inputs, &gpu)?,
                gpu,
            }
        }
        None => inputs.into()
    })
}

impl NNController {
    /// The same as eval_batch except without the "batch" dimension in the input and the output
    pub fn eval_one(&self, inputs: ArrayDynF) -> GenericResult<ArrayDynF> {
        self.eval_batch(stack![Axis(0), inputs])
            .map(|o| o.remove_axis(Axis(0)))
    }

    /// Forward the input through the layers and return the result.
    /// Uses GPU if available
    pub fn eval_batch(&self, inputs: ArrayDynF) -> GenericResult<ArrayDynF> {
        let mut assigner = KeyAssigner::new();
        let config = BatchConfig::new_not_train();
        let gpu = self.get_gpu();

        let result = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs: prepare_inputs(inputs, gpu.clone())?,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: None,
                batch_config: &config,
                prev_iteration_cache: None,
                gpu,
            },
        )?.into_memory()?;
        self.finish_method()?;
        Ok(result)
    }

    pub fn eval_with_cache(&self, inputs: ArrayDynF, prev_iteration_cache: Option<GenericStorage>)
                           -> GenericResult<(ArrayDynF, GenericStorage)> {
        let mut assigner = KeyAssigner::new();
        let config = BatchConfig::new_not_train();
        let gpu = self.get_gpu();
        let mut cache = prev_iteration_cache.unwrap_or_default();

        let result = forward_layer(
            &self.main_layer,
            ForwardData {
                inputs: prepare_inputs(inputs, gpu.clone())?,
                assigner: &mut assigner,
                storage: &self.storage,
                forward_cache: None,
                batch_config: &config,
                prev_iteration_cache: Some(&mut cache),
                gpu,
            },
        )?.into_memory()?;

        self.finish_method()?;
        Ok((result, cache))
    }
}
