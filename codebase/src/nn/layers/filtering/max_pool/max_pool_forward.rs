use ndarray::s;
use crate::{Array4F};
use crate::gpu::buffers::GpuBuffer;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::gpu::shaders;
use crate::nn::layers::filtering::max_pool::{gen_name, MaxPoolConfig};
use crate::nn::layers::filtering::pad4d;
use crate::nn::layers::nn_layers::{ForwardData, LayerResult};
use crate::nn::layers::stored_array::StoredArray;
use crate::nn::utils::shape_length;
use crate::utils::{GenericResult, get_dims_after_filter_4};

pub fn forward(data: ForwardData, layer_config: &MaxPoolConfig) -> LayerResult {
    let ForwardData { inputs, forward_cache, assigner, .. } = data;

    let key = assigner.get_key(gen_name());
    forward_cache.insert(key, vec![inputs.to_memory()?.into_dyn()]);

    let result = match inputs {
        StoredArray::Memory { data } => {
            forward_cpu(data.into_dimensionality()?, layer_config.size, layer_config.stride, layer_config.padding)
        }
        StoredArray::GpuLocal { data, gpu, shape } => {
            forward_gpu(data, gpu, layer_config, shape)?
        }
    };

    Ok(result)
}

fn forward_cpu(inputs: Array4F, size: usize, stride: usize, padding: usize) -> StoredArray {
    let inputs = pad4d(inputs, padding);
    let [batch_size, channels, new_height, new_width] = get_dims_after_filter_4(inputs.shape(), size, stride);

    Array4F::from_shape_fn((batch_size, channels, new_height, new_width), |(b, c, h, w)| {
        let h_offset = h * stride;
        let w_offset = w * stride;
        let area = inputs.slice(s![b, c, h_offset..(h_offset + size), w_offset..(w_offset + size)]);
        area.into_iter().copied().reduce(f32::max).unwrap_or(0.0)
    }).into_dyn().into()
}

fn forward_gpu(inputs: GpuBuffer, gpu: GlobalGpu, layer_config: &MaxPoolConfig, in_shape: Vec<usize>) -> GenericResult<StoredArray> {
    let padded_ish = [in_shape[0], in_shape[1], in_shape[2]+2*layer_config.padding, in_shape[3]+2*layer_config.padding];
    let out_shape = get_dims_after_filter_4(&padded_ish, layer_config.size, layer_config.stride);

    let mut runner = ShaderRunner2::new_separate_io(gpu.clone(), shaders::max_pool_forward::load, "main", &shaders::max_pool_forward::SpecializationConstants {
        in_channels: out_shape[1] as u32,
        size: layer_config.size as u32,
        stride: layer_config.stride as u32,
        out_height: out_shape[2] as u32,
        out_width: out_shape[3] as u32,
        input_height: in_shape[2] as u32,
        input_width: in_shape[3] as u32,
        padding: layer_config.padding as u32,
    }, shape_length(&out_shape))?;

    runner.create_input_buffer(StoredArray::GpuLocal { gpu: gpu.clone(), data: inputs.clone(), shape: in_shape.clone() })?;
    let result = runner.execute([out_shape[0] * out_shape[1], out_shape[2], out_shape[3]].map(|o| o as u32),
                   shaders::max_pool_forward::BLOCK_SIZE)?;

    Ok(StoredArray::GpuLocal {gpu, data: result, shape: out_shape.to_vec()})
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::ArrayDynF;
    use crate::gpu::buffers::upload_array_to_gpu;
    use crate::gpu::gpu_data::GpuData;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::filtering::max_pool::tests::{create_forward_outputs, create_inputs};
    use crate::nn::layers::nn_layers::GenericStorage;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward_2x2_cpu() {
        let inputs = create_inputs();
        let expected = create_forward_outputs();

        fn action(inputs: ArrayDynF, size: usize, stride: usize) -> ArrayDynF {
            forward(ForwardData {
                inputs: inputs.into(),
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                storage: &mut GenericStorage::new(),
                forward_cache: &mut GenericStorage::new(),
                prev_iteration_cache: None,
                gpu: None,
            }, &MaxPoolConfig { size, stride, padding: 0 }).unwrap().into_memory().unwrap()
        }

        assert_eq!(expected.into_dyn(), action(inputs, 2, 2));
    }

    #[test]
    fn test_forward_2x2_gpu() {
        let inputs = create_inputs();
        let expected = create_forward_outputs();

        fn action(inputs: ArrayDynF, size: usize, stride: usize) -> ArrayDynF {
            let gpu = GpuData::new_global().unwrap();

            forward(ForwardData {
                gpu: Some(gpu.clone()),
                inputs: StoredArray::GpuLocal {shape: inputs.shape().to_vec(), data:upload_array_to_gpu(&inputs, &gpu).unwrap(), gpu},
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                storage: &mut GenericStorage::new(),
                forward_cache: &mut GenericStorage::new(),
                prev_iteration_cache: None,
            }, &MaxPoolConfig { size, stride, padding: 0 }).unwrap().into_memory().unwrap()
        }

        assert_eq!(expected.into_dyn(), action(inputs, 2, 2));
    }

    #[test]
    fn test_cpu_gpu_equal() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let inputs = Array4F::random((8, 3, 6, 6), &dist);
        let config = MaxPoolConfig {
            stride: 2,
            padding: 1,
            size: 2,
        };

        let gpu = GpuData::new_global().unwrap();
        let shape = inputs.shape().to_vec();

        let cpu_out = forward_cpu(inputs.clone(), config.size, config.stride, config.padding)
            .into_memory().unwrap();
        let gpu_out = forward_gpu(upload_array_to_gpu(&inputs.into_dyn(), &gpu).unwrap(), gpu, &config, shape)
            .unwrap().into_memory().unwrap();

        assert_eq!(cpu_out.shape(), gpu_out.shape());
        println!("{:?}\n----------\n{:?}", cpu_out, gpu_out);
        assert!(arrays_almost_equal(&cpu_out, &gpu_out));
    }
}