use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::gpu::shaders;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::StoredArray;

pub(crate) struct ReluLayer;

fn gen_name() -> String {
    "relu".to_owned()
}

impl LayerOps<()> for ReluLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { assigner, inputs, gpu, .. } = data;

        match gpu {
            Some(gpu) => {
                let shape = inputs.shape().iter().copied().collect();
                let times= inputs.len();
                let runner = ShaderRunner2::new_inplace(
                    gpu.clone(),
                    |d| shaders::relu_forward::load(d),
                    "main",
                    &shaders::relu_forward::SpecializationConstants{},
                    inputs
                )?;
                let data = runner.execute([times as u32, 1, 1], shaders::relu_forward::BLOCK_SIZE)?;

                Ok(StoredArray::GpuLocal {
                    shape,
                    gpu,
                    data,
                })
            },
            None => {
                Ok(StoredArray::Memory {
                    data: inputs.into_memory()?.mapv_into(|o| if o > 0.0 { o } else { 0.0 })
                })
            }
        }
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());

        let [cache] = remove_from_storage1(forward_cache, &key);
        Ok(StoredArray::Memory{
            data: grad * cache.mapv_into(|o| if o > 0.0 { 1.0 } else { 0.0 })
        })
    }
}