use ndarray::{Axis, concatenate};
use vulkano::buffer::BufferAccess;
use vulkano::command_buffer::BufferCopy;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_context::{BufferConfig, ContextBinding, ShaderContext};
use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::nn::layers::concat_layer::{ConcatConfig, gen_name};
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::{Array1F, GenericResult, shape_length};

pub fn forward(data: ForwardData, layer_config: &ConcatConfig) -> LayerResult {
    let mut results: Vec<StoredArray> = Vec::with_capacity(layer_config.layers.len());
    let mut splits = Vec::with_capacity(layer_config.layers.len());
    let key = data.assigner.get_key(gen_name(layer_config));
    let concat_dim = layer_config.dim + 1;

    let ForwardData {
        inputs, mut forward_cache, storage, gpu, assigner,
        batch_config, mut prev_iteration_cache
    } = data;

    for layer in &layer_config.layers {
        let result = forward_layer(layer, ForwardData {
            inputs: inputs.clone(),
            forward_cache: forward_cache.as_deref_mut(),
            storage,
            gpu: gpu.clone(),
            assigner,
            batch_config,
            prev_iteration_cache: prev_iteration_cache.as_deref_mut(),
        })?;

        if let Some(first_result) = results.get(0) {
            if first_result.shape().len() != result.shape().len() {
                return Err(anyhow::anyhow!("Results differ in dimensionality"));
            }

            for dim in 0..first_result.shape().len() {
                // Skip checking the dimension that will be concatenated
                if concat_dim == dim {
                    continue;
                }

                if first_result.shape()[dim] != result.shape()[dim] {
                    return Err(anyhow::anyhow!("Results differ in shape for dimension {}", dim));
                }
            }
        }

        splits.push(result.shape()[concat_dim]);
        results.push(result);
    }

    if results.is_empty() {
        return Err(anyhow::anyhow!("Layer is empty"));
    }

    if let Some(forward_cache) = forward_cache {
        forward_cache.insert(key.clone(), vec![Array1F::from_iter(splits.iter().map(|o| *o as f32)).into_dyn()]);
    }

    if results.iter().all(|o| matches!(o, StoredArray::GpuLocal {..})) {
        match forward_gpu(results.clone(), &splits, gpu.unwrap(), key, concat_dim) {
            Ok(v) => Ok(v),
            Err(e) => {
                eprintln!("{}", e);
                forward_cpu(results, concat_dim)
            }
        }
    } else {
        forward_cpu(results, concat_dim)
    }
}

fn forward_gpu(results: Vec<StoredArray>, sections: &[usize], gpu: GlobalGpu, id: String, concat_dim: usize) -> GenericResult<StoredArray> {
    const ELEMENT_SIZE: u64 = std::mem::size_of::<f32>() as u64;
    let key = (id, "forward".to_owned());

    let mut osh = results[0].shape().to_vec();
    osh[concat_dim] = sections.iter().copied().sum();
    let copy_count = shape_length(&osh[..concat_dim]) as u64;
    let total_copy_size = shape_length(&osh[concat_dim..]) as u64 * ELEMENT_SIZE;

    ShaderContext::register(&key, gpu.clone(), &[
        BufferConfig::floats(shape_length(&osh) as u64)
    ], Ok)?;
    let mut runner = ShaderRunner2::new(key, gpu.clone())?;
    let mut small_offset = 0;

    for r in results {
        if let StoredArray::GpuLocal { data, shape, .. } = r {
            if data.size() % ELEMENT_SIZE != 0 {
                return Err(anyhow::anyhow!("Buffer size is not a multiple of {}. This probably means that the buffer type is not f32", ELEMENT_SIZE));
            }

            let copy_size = shape_length(&shape[concat_dim..]) as u64 * ELEMENT_SIZE;

            let mut copies = Vec::new();
            for c in 0..copy_count {
                copies.push(BufferCopy {
                    size: copy_size,
                    src_offset: c * copy_size,
                    dst_offset: small_offset + c * total_copy_size,
                    ..Default::default()
                })
            }

            runner.update_buffer_with_buffer_custom(ContextBinding(0), data, copies)?;
            small_offset += copy_size;
        } else {
            return Err(anyhow::anyhow!("All buffers should be in the GPU"));
        }
    }

    let result = runner.finish()?;
    Ok(StoredArray::GpuLocal { gpu, shape: osh, data: result })
}

fn forward_cpu(results: Vec<StoredArray>, concat_dim: usize) -> GenericResult<StoredArray> {
    let mut array_results = Vec::new();
    for r in results {
        array_results.push(r.into_memory()?);
    }

    let results_views: Vec<_> = array_results.iter().map(|o| o.view()).collect();
    let concat = concatenate(Axis(concat_dim), &results_views)?;
    Ok(StoredArray::Memory { data: concat })
}

#[cfg(test)]
mod tests {
    use crate::Array4F;
    use crate::gpu::gpu_data::GpuData;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_gpu_equal_cpu() {
        let mut i = 0.0;
        let arrays = vec![
            Array4F::from_shape_fn((3, 2, 4, 3), |_| {
                i += 1.0;
                i
            }),
            Array4F::from_shape_fn((3, 1, 4, 3), |_| {
                i += 1.0;
                i
            }),
            Array4F::from_shape_fn((3, 3, 4, 3), |_| {
                i += 1.0;
                i
            }),
        ];
        let arrays: Vec<_> = arrays.into_iter().map(|o| o.into_dyn().into()).collect();

        let gpu = GpuData::new_global().unwrap();
        let expected = forward_cpu(arrays.clone(), 1)
            .unwrap().into_memory().unwrap();

        let arrays = arrays.into_iter().map(|o| {
            StoredArray::GpuLocal { shape: o.shape().to_vec(), gpu: gpu.clone(), data: o.into_gpu_local(gpu.clone()).unwrap() }
        }).collect();
        let actual = forward_gpu(arrays, &[2, 1, 3], gpu, "any".to_owned(), 1)
            .unwrap().into_memory().unwrap();

        println!("{:?}", expected.iter().collect::<Vec<_>>());
        println!("{:?}\n---------------\n{:?}", actual, expected);
        assert!(arrays_almost_equal(&expected, &actual));
    }
}