use crate::ArrayDynF;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::gpu::shaders;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::{GpuBuffer, StoredArray};
use crate::nn::utils::shape_length;
use crate::utils::GenericResult;

pub(crate) struct ReluLayer;

fn gen_name() -> String {
    "relu".to_owned()
}

impl LayerOps<()> for ReluLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { inputs,  .. } = data;

        // let inputs = {
        //     let shape = Vec::from(inputs.shape());
        //     let b = inputs.into_gpu_local(gpu.clone().unwrap())?;
        //     StoredArray::GpuLocal { gpu: gpu.unwrap(), data: b, shape }
        // };

        // Only use the GPU if the data is already there
        Ok(match inputs {
            StoredArray::Memory { data } => StoredArray::Memory {
                data: forward_cpu(data)
            },
            StoredArray::GpuLocal { data, gpu, shape } => StoredArray::GpuLocal {
                data: forward_gpu(data, &gpu, &shape)?,
                gpu,
                shape,
            }
        })
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

fn forward_cpu(inputs: ArrayDynF) -> ArrayDynF {
    inputs.mapv_into(|o| if o > 0.0 { o } else { 0.0 })
}

fn forward_gpu(inputs: GpuBuffer, gpu: &GlobalGpu, shape: &[usize]) -> GenericResult<GpuBuffer> {
    let runner = ShaderRunner2::new_inplace(
        gpu.clone(),
        shaders::relu_forward::load,
        "main",
        &shaders::relu_forward::SpecializationConstants {},
        inputs,
        shape,
    )?;
    runner.execute([shape_length(shape) as u32, 1, 1], shaders::relu_forward::BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use crate::gpu::gpu_data::GpuData;
    use crate::utils::{Array3F, arrays_almost_equal};
    use super::*;

    #[test]
    fn test_forward_gpu() {
        let inputs = Array3F::from_shape_vec((2,2,2),
                                                     vec![1.0, 2.0, 3.0, -1.0, -9.0, 0.0, -1.5, 1.0]).unwrap().into_dyn();
        let expected_array = Array3F::from_shape_vec((2,2,2),
                                                     vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap().into_dyn();

        let gpu = GpuData::new_global().unwrap();
        let inputs = StoredArray::Memory {data: inputs}.into_gpu_local(gpu.clone()).unwrap();


        let output = forward_gpu(inputs, &gpu, &[2,2,2]).unwrap();
        let output = StoredArray::GpuLocal {gpu, data: output, shape: vec![2,2,2]}.into_memory().unwrap();

        assert!(arrays_almost_equal(&output, &expected_array));
    }
}