use crate::ArrayDynF;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_context::{BufferConfig, ContextBinding, ShaderBinding, ShaderContext};
use crate::gpu::shader_runner_2::{ShaderRunner2};
use crate::gpu::{BufferChecksumMethod, shaders};
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::shape_length;
use crate::utils::GenericResult;

pub(crate) struct ReluLayer;

fn gen_name() -> String {
    "relu".to_owned()
}

impl LayerOps<()> for ReluLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { inputs, assigner, gpu, forward_cache, .. } = data;
        let key = assigner.get_key(gen_name());

        if let Some(forward_cache) = forward_cache{
            forward_cache.insert(key.clone(), vec![inputs.to_memory()?]);
        }

        // Only use the GPU if the data is already there
        if matches!(inputs, StoredArray::GpuLocal {..}) {
            match forward_gpu(key, &inputs, gpu.unwrap()) {
                Ok(v) => Ok(v),
                Err(e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("{}", e);
                    Ok(forward_cpu(inputs.into_memory()?))
                }
            }
        } else {
            Ok(forward_cpu(inputs.into_memory()?))
        }
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());

        let [cache] = remove_from_storage1(forward_cache, &key);
        Ok(StoredArray::Memory {
            data: grad * cache.mapv_into(|o| if o > 0.0 { 1.0 } else { 0.0 })
        })
    }
}

fn forward_cpu(inputs: ArrayDynF) -> StoredArray {
    inputs.mapv_into(|o| if o > 0.0 { o } else { 0.0 }).into()
}

fn forward_gpu(id: String, inputs: &StoredArray, gpu: GlobalGpu) -> GenericResult<StoredArray> {
    let shape= inputs.shape().to_vec();
    let key = (id, "forward".to_owned());

    ShaderContext::register(&key, gpu.clone(), &[BufferConfig::floats(shape_length(&shape))], |mut b| {
        b.register_shader("forward", shaders::relu_forward::load, vec![
            (ContextBinding(0), ShaderBinding(0)),
        ], &shaders::relu_forward::SpecializationConstants {})?;
        Ok(b)
    })?;

    let mut runner = ShaderRunner2::new(key, gpu.clone())?;

    runner
        .update_buffer_with_stored_array(ContextBinding(0), inputs, BufferChecksumMethod::None)?
        .dispatch("forward", [shape_length(&shape) as u32, 1, 1], shaders::relu_forward::BLOCK_SIZE)?;
    Ok(StoredArray::GpuLocal { data: runner.finish()?, gpu, shape })
}

#[cfg(test)]
mod tests {
    use crate::gpu::buffers::upload_array_to_gpu;
    use crate::gpu::gpu_data::GpuData;
    use crate::utils::{Array3F, arrays_almost_equal};
    use super::*;

    #[test]
    fn test_forward_gpu() {
        let inputs = Array3F::from_shape_vec((2, 2, 2),
                                             vec![1.0, 2.0, 3.0, -1.0, -9.0, 0.0, -1.5, 1.0]).unwrap().into_dyn();
        let expected_array = Array3F::from_shape_vec((2, 2, 2),
                                                     vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap().into_dyn();

        let gpu = GpuData::new_global().unwrap();
        let inputs = StoredArray::GpuLocal {data:  upload_array_to_gpu(&inputs, &gpu).unwrap(), shape: inputs.shape().to_vec(), gpu: gpu.clone()};


        let output = forward_gpu("".to_owned(), &inputs, gpu)
            .unwrap().into_memory().unwrap();

        assert!(arrays_almost_equal(&output, &expected_array));
    }
}