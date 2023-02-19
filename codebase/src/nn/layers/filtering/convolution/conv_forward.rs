use ndarray::parallel::prelude::*;
use ndarray::{ArrayView3, ArrayViewMut3, Axis, s, ShapeError, stack, Zip};
use crate::{Array4F, ArrayDynF};
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_runner::ShaderRunner;
use crate::gpu::shaders;
use crate::nn::generic_storage::clone_from_storage1;
use crate::nn::layers::filtering::convolution::{ConvolutionConfig, gen_name};
use crate::nn::layers::filtering::{find_useful_from_prev, pad4d};
use crate::nn::layers::nn_layers::{ForwardData, LayerResult};
use crate::utils::{Array3F, GenericResult, get_dims_after_filter_4};

pub fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
    let ForwardData { inputs, storage, assigner, forward_cache, mut prev_iteration_cache, .. } = data;

    let inputs: Array4F = inputs.into_dimensionality()?;
    let key = assigner.get_key(gen_name(layer_config));

    let [kernel] = clone_from_storage1(storage, &key);
    let kernel: Array4F = kernel.into_dimensionality()?;

    let prev_values: Option<[ArrayDynF; 2]> = if layer_config.cache {
        prev_iteration_cache.as_mut()
            .and_then(|o| o.remove(&key))
            .map(|o| o.try_into().unwrap())
    } else {
        None
    };

    let inputs = pad4d(inputs, layer_config.padding);

    let result = match prev_values {
        Some([prev_inputs, prev_result]) => {
            cpu_forward_cache(&inputs, &prev_inputs.into_dimensionality()?,
                              &prev_result.into_dimensionality()?, &kernel, layer_config)?
        }
        None => {
            match data.gpu {
                Some(gpu) => match gpu_forward(&inputs, &kernel, gpu, layer_config) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{:?}", e);
                        cpu_forward(&inputs, &kernel, layer_config)?
                    }
                }
                None => cpu_forward(&inputs, &kernel, layer_config)?
            }
        }
    };

    let result = result.into_dyn();
    let inputs = inputs.into_dyn();

    forward_cache.insert(key.clone(), vec![inputs.clone()]);
    if let Some(cache) = prev_iteration_cache {
        cache.insert(key, vec![inputs, result.clone()]);
    }
    Ok(result)
}

pub fn cpu_forward(inputs: &Array4F, kernel: &Array4F, layer_config: &ConvolutionConfig) -> Result<Array4F, ShapeError> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;

    let [batch, _, new_height, new_width] = get_dims_after_filter_4(inputs, *kernel_size, *stride);
    let mut batches = Vec::with_capacity(batch);
    inputs.outer_iter()
        .into_par_iter()
        .map(|inputs| {
            let mut result = Array3F::zeros((layer_config.out_channels, new_height, new_width));
            for h in 0..new_height {
                for w in 0..new_width {
                    apply_conv_filter(kernel, stride, kernel_size, &inputs, &mut result.view_mut(), h, w);
                }
            }
            result
        })
        .collect_into_vec(&mut batches);

    let mut views = Vec::with_capacity(batch);
    views.extend(batches.iter().map(|o| o.view()));
    stack(Axis(0), &views)
}

pub fn cpu_forward_cache(inputs: &Array4F, prev_inputs: &Array4F, prev_results: &Array4F, kernel: &Array4F, layer_config: &ConvolutionConfig) -> Result<Array4F, ShapeError> {
    let ConvolutionConfig { stride, kernel_size, out_channels, .. } = layer_config;

    let [batch, _, new_height, new_width] = get_dims_after_filter_4(inputs, *kernel_size, *stride);

    let useful_cache = find_useful_from_prev(prev_inputs, prev_results, inputs, *kernel_size, *stride);
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
                        apply_conv_filter(kernel, stride, kernel_size, &inputs, &mut result, h, w);
                    }
                }
            }
        });

    Ok(result)
}

pub fn gpu_forward(inputs: &Array4F, kernel: &Array4F, gpu: GlobalGpu, layer_config: &ConvolutionConfig) -> GenericResult<Array4F> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;
    let [batch_size, _, new_height, new_width] = get_dims_after_filter_4(inputs, *kernel_size, *stride);

    let ish = inputs.shape();
    let out_shape = (batch_size, layer_config.out_channels, new_height, new_width);
    let mut runner = ShaderRunner::new(gpu, shaders::convolution_forward::load, "main", &shaders::convolution_forward::SpecializationConstants {
        in_channels: layer_config.in_channels as u32,
        out_channels: layer_config.out_channels as u32,
        kernel_size: layer_config.kernel_size as u32,
        stride: layer_config.stride as u32,
        out_height: out_shape.2 as u32,
        out_width: out_shape.3 as u32,
        input_height: ish[2] as u32,
        input_width: ish[3] as u32,
    })?;

    let results_buffer = runner.create_download_buffer(out_shape.0 * out_shape.1 * out_shape.2 * out_shape.3)?;
    runner.create_gpu_only_buffer(kernel)?;
    runner.create_gpu_only_buffer(inputs)?;

    runner.execute([out_shape.0 * out_shape.1, out_shape.2, out_shape.3].map(|o| o as u32),
                   shaders::convolution_forward::BLOCK_SIZE)?;

    let vec = results_buffer.read()?.to_vec();

    Ok(Array4F::from_shape_vec(out_shape, vec)?)
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
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::gpu::gpu_data::GpuData;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::filtering::convolution::ConvolutionInitMode::HeNormal;
    use crate::nn::layers::filtering::convolution::test_values::*;
    use crate::nn::layers::nn_layers::GenericStorage;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward() {
        let inputs = get_inputs();
        let expected = get_forward_result();
        let mut storage = get_storage();
        let config = get_config();

        let result = forward(
            ForwardData {
                inputs,
                forward_cache: &mut GenericStorage::new(),
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                batch_config: &BatchConfig::new_train(),
                prev_iteration_cache: None,
                gpu: None,
            },
            &config,
        ).unwrap();

        assert!(arrays_almost_equal(&expected, &result));
    }

    #[test]
    fn test_gpu_cpu_equal_forward() {
        let config = ConvolutionConfig {
            in_channels: 6,
            out_channels: 3,
            stride: 2,
            padding: 0,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            kernel_size: 2,
            cache: false,
        };

        let dist = Normal::new(0.0, 1.0).unwrap();
        let inputs = Array4F::random((8, config.in_channels, 5, 5), &dist);
        let kernels = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist);
        let expected = cpu_forward(&inputs, &kernels, &config).unwrap();
        let actual = gpu_forward(&inputs, &kernels, GpuData::new_global().unwrap(), &config).unwrap();

        println!("{:?}\n\n------\n\n{:?}", expected, actual);
        assert!(arrays_almost_equal(&expected, &actual));
    }
}