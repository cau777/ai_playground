use ndarray::parallel::prelude::*;
use ndarray::{ArrayView3, ArrayViewMut3, Axis, s, stack, Zip};
use crate::{Array4F, ArrayDynF};
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_runner_2::{ShaderRunner2};
use crate::gpu::{BufferChecksumMethod, shaders};
use crate::gpu::shader_context::{BufferConfig, ContextBinding, ShaderBinding, ShaderContext};
use crate::nn::generic_storage::clone_from_storage1;
use crate::nn::layers::filtering::convolution::{ConvolutionConfig, gen_name};
use crate::nn::layers::filtering::{find_useful_from_prev, pad4d};
use crate::nn::layers::nn_layers::{ForwardData, LayerResult};
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::{shape_length};
use crate::utils::{Array3F, GenericResult, get_dims_after_filter_4};

pub fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
    let ForwardData { inputs, storage, assigner, forward_cache, mut prev_iteration_cache, .. } = data;
    let key = assigner.get_key(gen_name(layer_config));

    if let Some(forward_cache) = forward_cache {
        forward_cache.insert(key.clone(), vec![inputs.to_memory()?]);
    }

    let [kernel] = clone_from_storage1(storage, &key);

    let result = match data.gpu {
        Some(gpu) => match gpu_forward_with_cache(key, inputs.clone(), &kernel, gpu, layer_config) {
            Ok(v) => v,
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("{}", _e);
                cpu_forward(inputs, kernel, layer_config)?
            }
        }
        None => {
            let cache_enabled = layer_config.cache && prev_iteration_cache.is_some();
            let inputs_to_cache = if cache_enabled {
                Some(inputs.to_memory()?)
            } else {
                None
            };

            let prev_values: Option<[ArrayDynF; 2]> = if cache_enabled {
                prev_iteration_cache.as_mut()
                    .and_then(|o| o.remove(&key))
                    .and_then(|o| o.try_into().ok())
            } else {
                None
            };

            let result = match prev_values {
                Some([prev_inputs, prev_result]) => {
                    cpu_forward_cache(inputs, prev_inputs, prev_result, kernel, layer_config)?
                }
                None => {
                    cpu_forward(inputs, kernel, layer_config)?
                }
            };

            if let Some(inputs_to_cache) = inputs_to_cache {
                prev_iteration_cache.unwrap()
                    .insert(key, vec![inputs_to_cache, result.to_memory()?]);
            }

            result
        }
    };

    Ok(result)
}

pub fn cpu_forward(inputs: StoredArray, kernel: ArrayDynF, layer_config: &ConvolutionConfig) -> GenericResult<StoredArray> {
    let kernel = kernel.into_dimensionality()?;
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;
    let inputs = inputs.into_memory()?.into_dimensionality()?;
    let inputs = pad4d(inputs, layer_config.padding);

    let [batch, _, new_height, new_width] = get_dims_after_filter_4(inputs.shape(), *kernel_size, *stride);
    let mut batches = Vec::with_capacity(batch);
    inputs.outer_iter()
        .into_par_iter()
        .map(|inputs| {
            let mut result = Array3F::zeros((layer_config.out_channels, new_height, new_width));
            for h in 0..new_height {
                for w in 0..new_width {
                    apply_conv_filter(&kernel, stride, kernel_size, &inputs, &mut result.view_mut(), h, w);
                }
            }
            result
        })
        .collect_into_vec(&mut batches);

    let mut views = Vec::with_capacity(batch);
    views.extend(batches.iter().map(|o| o.view()));
    Ok(StoredArray::Memory { data: stack(Axis(0), &views)?.into_dyn() })
}

pub fn cpu_forward_cache(inputs: StoredArray, prev_inputs: ArrayDynF, prev_results: ArrayDynF, kernel: ArrayDynF,
                         layer_config: &ConvolutionConfig) -> GenericResult<StoredArray> {
    let kernel = kernel.into_dimensionality()?;
    let ConvolutionConfig { stride, kernel_size, out_channels, .. } = layer_config;
    let inputs = inputs.into_memory()?.into_dimensionality()?;
    let inputs = pad4d(inputs, layer_config.padding);
    let prev_inputs = pad4d(prev_inputs.into_dimensionality()?, layer_config.padding);
    let prev_results = prev_results.into_dimensionality()?;

    let [batch, _, new_height, new_width] = get_dims_after_filter_4(inputs.shape(), *kernel_size, *stride);

    let useful_cache = find_useful_from_prev(
        &prev_inputs,
        &prev_results,
        &inputs, *kernel_size, *stride,
    );
    let mut result = Array4F::zeros((batch, layer_config.out_channels, new_height, new_width));

    Zip::from(inputs.outer_iter())
        .and(useful_cache.outer_iter())
        .and(result.outer_iter_mut())
        .into_par_iter()
        .for_each(|(inputs, cache, mut result)| {
            for h in 0..new_height {
                for w in 0..new_width {
                    // A channels is either cached entirely or not cached at all
                    if cache[(0, h, w)].is_some() {
                        for och in 0..*out_channels {
                            result[(och, h, w)] = cache[(och, h, w)].unwrap();
                        }
                    } else {
                        apply_conv_filter(&kernel, stride, kernel_size, &inputs, &mut result, h, w);
                    }
                }
            }
        });

    Ok(StoredArray::Memory { data: result.into_dyn() })
}

pub fn gpu_forward_with_cache(id: String, inputs: StoredArray, kernel: &ArrayDynF, gpu: GlobalGpu,
                              layer_config: &ConvolutionConfig) -> GenericResult<StoredArray> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;
    let key = (id, "forward".to_owned());

    let ish = inputs.shape();
    let padded_ish = [ish[0], ish[1], ish[2] + 2 * layer_config.padding, ish[3] + 2 * layer_config.padding];
    let [batch_size, _, new_height, new_width] = get_dims_after_filter_4(&padded_ish, *kernel_size, *stride);
    let osh = [batch_size, layer_config.out_channels, new_height, new_width];

    let buffer_lengths = [
        BufferConfig::floats(shape_length(&osh)),
        BufferConfig::floats(kernel.len()),
        BufferConfig::floats(inputs.len()),
        BufferConfig::bools(inputs.len()),
        BufferConfig::bools(ish[0] * ish[2] * ish[3]),
        BufferConfig::bools(osh[0] * osh[2] * osh[3]),
    ];

    ShaderContext::register(&key, gpu.clone(), &buffer_lengths, |mut b| {
        let constants = shaders::convolution_forward::forward::SpecializationConstants {
            in_channels: layer_config.in_channels as u32,
            out_channels: layer_config.out_channels as u32,
            kernel_size: layer_config.kernel_size as u32,
            stride: layer_config.stride as u32,
            out_height: osh[2] as u32,
            out_width: osh[3] as u32,
            input_height: ish[2] as u32,
            input_width: ish[3] as u32,
            padding: layer_config.padding as u32,
            cache_enabled: layer_config.cache as u32,
        };

        b.register_shader("forward", shaders::convolution_forward::forward::load, vec![
            (ContextBinding(0), ShaderBinding(0)),
            (ContextBinding(1), ShaderBinding(1)),
            (ContextBinding(2), ShaderBinding(2)),
            (ContextBinding(5), ShaderBinding(3)),
        ], &constants)?;

        b.register_shader("validate_cache_1", shaders::convolution_forward::validate_cache_1::load, vec![
            (ContextBinding(4), ShaderBinding(0)),
            (ContextBinding(2), ShaderBinding(1)),
            (ContextBinding(3), ShaderBinding(2)),
        ], &constants)?;

        b.register_shader("validate_cache_2", shaders::convolution_forward::validate_cache_2::load, vec![
            (ContextBinding(5), ShaderBinding(0)),
            (ContextBinding(4), ShaderBinding(1)),
        ], &constants)?;
        Ok(b)
    })?;

    let mut runner = ShaderRunner2::new(key, gpu.clone())?;
    let mut changed = false;
    runner.update_buffer_with_memory_checked(ContextBinding(1), kernel, BufferChecksumMethod::Single, &mut changed)?;

    if changed {
        runner.update_buffer_with_val(ContextBinding(3), 0.0)?;
    }

    runner.update_buffer_with_stored_array(ContextBinding(2), &inputs, BufferChecksumMethod::Split)?;

    if layer_config.cache {
        runner.dispatch("validate_cache_1", [ish[0], ish[2], ish[3]].map(|o| o as u32), shaders::convolution_forward::validate_cache_1::BLOCK_SIZE)?
            .dispatch("validate_cache_2", [osh[0], osh[2], osh[3]].map(|o| o as u32), shaders::convolution_forward::validate_cache_2::BLOCK_SIZE)?;
    }

    runner
        .dispatch("forward", [osh[0] * osh[1], osh[2], osh[3]].map(|o| o as u32), shaders::convolution_forward::forward::BLOCK_SIZE)?
    ;
    if layer_config.cache {
        runner.update_buffer_with_binding(ContextBinding(2), ContextBinding(3))?;
    }

    let result = runner.finish()?;
    Ok(StoredArray::GpuLocal { gpu, shape: osh.to_vec(), data: result })
}

fn apply_conv_filter(kernel: &Array4F, stride: &usize, kernel_size: &usize, inputs: &ArrayView3<f32>, result: &mut ArrayViewMut3<f32>, h: usize, w: usize) {
    let h_offset = h * stride;
    let w_offset = w * stride;
    let area = inputs.slice(s![
        ..,
        h_offset..(h_offset + kernel_size),
        w_offset..(w_offset + kernel_size)
    ]);
    let area = area.insert_axis(Axis(0));
    let out: Array4F = &area * kernel;

    out.outer_iter()
        .map(|o| o.sum())
        .enumerate()
        .for_each(|(index, o)| result[(index, h, w)] = o);
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand::{Rng, thread_rng};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::gpu::gpu_data::{get_global_gpu};
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::filtering::convolution::ConvolutionInitMode::HeNormal;
    use crate::nn::layers::filtering::convolution::test_values::*;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward_cpu() {
        let inputs = get_inputs();
        let expected = get_forward_result();
        let mut storage = get_storage();
        let config = get_config();

        let result = forward(
            ForwardData {
                inputs: inputs.into(),
                forward_cache: None,
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                batch_config: &BatchConfig::new_train(),
                prev_iteration_cache: None,
                gpu: None,
            },
            &config,
        ).unwrap();

        let result = result.into_memory().unwrap();
        assert_eq!(expected.shape(), result.shape());
        assert!(arrays_almost_equal(&expected, &result));
    }

    #[test]
    fn test_forward_gpu() {
        let inputs = get_inputs();
        let expected = get_forward_result();
        let mut storage = get_storage();
        let config = get_config();

        let result = forward(
            ForwardData {
                inputs: inputs.into(),
                forward_cache: None,
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                batch_config: &BatchConfig::new_train(),
                prev_iteration_cache: None,
                gpu: Some(get_global_gpu().unwrap()),
            },
            &config,
        ).unwrap();

        let result = result.into_memory().unwrap();
        assert_eq!(expected.shape(), result.shape());
        println!("{:?}\n------------------\n{:?}", expected, result);
        assert!(arrays_almost_equal(&expected, &result));
    }

    #[test]
    fn test_gpu_cpu_equal_forward() {
        let config = ConvolutionConfig {
            in_channels: 6,
            out_channels: 3,
            stride: 2,
            padding: 2,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            kernel_size: 2,
            cache: false,
        };

        let dist = Normal::new(0.0, 1.0).unwrap();
        let inputs = Array4F::random((8, config.in_channels, 8, 8), &dist);
        let mut kernels = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist).into_dyn();

        for x in 0..10 {
            if x == 5 {
                kernels = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist).into_dyn();
            }
            let mut inputs = inputs.clone();
            if x % 3 == 0 {
                let mut rng = thread_rng();
                inputs.iter_mut().for_each(|o| {
                    if rng.gen_bool(0.1) {
                        *o *= rng.gen_range(0.0..2.0);
                    }
                });
            }

            let expected = cpu_forward(StoredArray::Memory { data: inputs.clone().into_dyn() }, kernels.clone(), &config).unwrap().into_memory().unwrap();
            let actual = gpu_forward_with_cache("test_gpu_cpu_equal_forward".to_owned(), StoredArray::Memory { data: inputs.into_dyn() }, &kernels, get_global_gpu().unwrap(), &config)
                .unwrap().into_memory().unwrap();

            println!("{:?}\n---------------\n{:?}", actual, expected);
            assert!(arrays_almost_equal(&expected, &actual));
        }
    }
}