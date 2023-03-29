use std::ops::AddAssign;
use ndarray::parallel::prelude::*;
use ndarray::{Axis, s, stack};
use crate::Array4F;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_context::{BufferConfig, ContextBinding, ShaderBinding, ShaderContext};

use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::gpu::{BufferChecksumMethod, shaders};
use crate::gpu::buffers::download_array_from_gpu;
use crate::nn::generic_storage::{clone_from_storage1, remove_from_storage1};
use crate::nn::layers::filtering::convolution::{ConvolutionConfig, gen_name};
use crate::nn::layers::filtering::{pad4d, remove_padding_4d};
use crate::nn::layers::nn_layers::{BackwardData, LayerResult};
use crate::utils::{Array2F, Array5F, GenericResult, get_dims_after_filter_4, shape_length};

pub fn backward(data: BackwardData, layer_config: &ConvolutionConfig) -> LayerResult {
    let BackwardData {
        assigner, forward_cache, storage,
        grad, backward_cache, ..
    } = data;

    let key = assigner.get_key(gen_name(layer_config));

    let [kernel] = clone_from_storage1(storage, &key);
    let kernel: Array4F = kernel.into_dimensionality()?;

    let [inputs] = remove_from_storage1(forward_cache, &key);
    let inputs = inputs.into_dimensionality()?;
    let inputs = pad4d(inputs, layer_config.padding);

    let grad = grad.into_dimensionality()?;

    let kernels_grad = calc_kernel_grad(&inputs, &grad, layer_config);
    backward_cache.insert(key.clone(), vec![kernels_grad.into_dyn()]);

    let inputs_grad = match data.gpu {
        Some(gpu) => match gpu_inputs_grad(key, &inputs, &grad, &kernel, gpu, layer_config) {
            Ok(v) => v,
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("{}", _e);
                cpu_inputs_grad(inputs, grad, kernel, layer_config)
            }
        }
        None => cpu_inputs_grad(inputs, grad, kernel, layer_config)
    };

    Ok(inputs_grad.into_dyn().into())
}

pub fn calc_kernel_grad(inputs: &Array4F, grad: &Array4F, layer_config: &ConvolutionConfig) -> Array4F {
    let ConvolutionConfig { in_channels, out_channels, kernel_size, stride, .. } = layer_config;
    let kernel_size = *kernel_size;
    let stride = *stride;

    let shape = inputs.shape();
    let batch = shape[0];
    let height = shape[2];
    let width = shape[3];

    let mean_inputs = inputs.mean_axis(Axis(0)).unwrap();
    let mean_grad = grad.mean_axis(Axis(0)).unwrap();
    let mean_grad = mean_grad.insert_axis(Axis(1));

    let mut parts = Vec::with_capacity(kernel_size * kernel_size);

    (0..kernel_size * kernel_size)
        .into_par_iter()
        .with_min_len(1)
        .map(|o| (o / kernel_size, o % kernel_size))
        .map(|(h, w)| {
            let affected = mean_inputs.slice(s![
                    ..,
                    h..height - (kernel_size - h - 1); stride,
                    w..width - (kernel_size - w - 1); stride]);
            let mul: Array4F = &mean_grad * &affected;
            Array2F::from_shape_fn((*out_channels, *in_channels), |(out_c, in_c)| {
                mul.index_axis(Axis(0), out_c)
                    .index_axis(Axis(0), in_c)
                    .mean().unwrap()
            })
        })
        .collect_into_vec(&mut parts);

    let mut views = Vec::with_capacity(batch);
    views.extend(parts.iter().map(|o| o.view()));

    let joined = stack(Axis(2), &views).unwrap();
    let mut reshaped = joined.into_shape((*out_channels, *in_channels, kernel_size, kernel_size)).unwrap();
    reshaped.swap_axes(2, 3);

    reshaped / (layer_config.kernel_size.pow(2) as f32)
}

pub fn cpu_inputs_grad(inputs: Array4F, grad: Array4F, kernel: Array4F, layer_config: &ConvolutionConfig) -> Array4F {
    let inputs_shape = inputs.shape();
    let ConvolutionConfig { kernel_size, stride, padding, .. } = layer_config;

    // Put height and width in front
    let grad = grad.permuted_axes((2, 3, 0, 1));
    let kernel = kernel.permuted_axes((1, 2, 3, 0));
    let kernel = kernel.insert_axis(Axis(3));

    let [batch_size, in_channels, new_height, new_width] =
        get_dims_after_filter_4(inputs.shape(), *kernel_size, *stride);

    let mut parts = Vec::with_capacity(new_height * new_width);
    (0..(new_height * new_width))
        .into_par_iter()
        .map(|o| (o % new_width, o / new_width))
        .map(|(h, w)| {
            let current_grad = grad.slice(s![h, w, .., ..]);
            let batch_mul: Array5F = &kernel * &current_grad;
            let batch_sum = batch_mul.sum_axis(Axis(4));
            batch_sum.permuted_axes((3, 0, 1, 2))
        })
        .collect_into_vec(&mut parts);

    let mut padded_result =
        Array4F::zeros((batch_size, in_channels, inputs_shape[2], inputs_shape[3]));
    parts.into_iter()
        .enumerate()
        .for_each(|(i, arr)| {
            let h = i % new_width;
            let w = i / new_width;
            let h_offset = h * stride;
            let w_offset = w * stride;
            padded_result.slice_mut(s![
                ..,
                ..,
                h_offset..(h_offset + kernel_size),
                w_offset..(w_offset + kernel_size)
            ]).add_assign(&arr);
        });

    remove_padding_4d(padded_result, *padding)
}


pub fn gpu_inputs_grad(id: String, inputs: &Array4F, grad: &Array4F, kernel: &Array4F,
                       gpu: GlobalGpu, layer_config: &ConvolutionConfig) -> GenericResult<Array4F> {
    let key = (id, "backward".to_owned());

    let ish: [usize; 4] = inputs.shape().try_into()?;
    let osh = [ish[0], ish[1], ish[2] - 2 * layer_config.padding, ish[3] - 2 * layer_config.padding];

    let ish = ish.map(|o| o as u32);
    let gsh: [usize; 4] = grad.shape().try_into()?;
    let gsh = gsh.map(|o| o as u32);
    let buffers = [
        BufferConfig::floats(shape_length(&osh)),
        BufferConfig::floats(shape_length(kernel.shape())),
        BufferConfig::floats(shape_length(grad.shape())),
    ];
    ShaderContext::register(&key, gpu.clone(), &buffers, |mut b| {
        b.register_shader("backward", shaders::convolution_inputs_grad::load, vec![
            (ContextBinding(0), ShaderBinding(0)),
            (ContextBinding(1), ShaderBinding(1)),
            (ContextBinding(2), ShaderBinding(2)),
        ], &shaders::convolution_inputs_grad::SpecializationConstants {
            batch_size: ish[0],
            grad_height: gsh[2],
            grad_width: gsh[3],
            input_height: ish[2],
            input_width: ish[3],
            in_channels: layer_config.in_channels as u32,
            out_channels: layer_config.out_channels as u32,
            kernel_size: layer_config.kernel_size as u32,
            stride: layer_config.stride as u32,
            padding: layer_config.padding as u32,
            out_height: osh[2] as u32,
            out_width: osh[3] as u32,
        })?;
        Ok(b)
    })?;

    let mut runner = ShaderRunner2::new(key, gpu.clone())?;

    runner.update_buffer_with_memory(ContextBinding(1), kernel, BufferChecksumMethod::Single)?
        .update_buffer_with_memory(ContextBinding(2), grad, BufferChecksumMethod::None)?
        .dispatch("backward", [osh[0] * osh[1], osh[2], osh[3]].map(|o| o as u32),
                  shaders::convolution_inputs_grad::BLOCK_SIZE)?;
    let result = runner.finish()?;
    let result = download_array_from_gpu(&result, osh.to_vec(), &gpu)?;
    Ok(result.into_dimensionality()?)
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::ArrayDynF;
    use crate::gpu::gpu_data::{get_global_gpu};
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
    fn test_backward() {
        let inputs: ArrayDynF = get_grad();
        let cache = get_inputs();
        let expected = get_backward_result();

        let mut storage = get_storage();
        let config = get_config();

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("convolution_2_3_2_2_1_0".to_owned(), vec![cache]);

        let result = backward(
            BackwardData {
                grad: inputs,
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                forward_cache: &mut forward_cache,
                backward_cache: &mut GenericStorage::new(),
                batch_config: &BatchConfig::new_train(),
                gpu: None,
            },
            &config,
        ).unwrap();

        // println!("{:?}\r\n--------\r\n{:?}", result, expected);
        assert!(arrays_almost_equal(&result.into_memory().unwrap(), &expected));
    }

    #[test]
    fn test_calc_kernel_grad() {
        let config = get_config();
        let inputs = get_inputs().into_dimensionality().unwrap();
        let inputs = pad4d(inputs, config.padding);
        let grad = get_grad().into_dimensionality().unwrap();
        let expected = get_kernels_grad();
        let result = calc_kernel_grad(&inputs, &grad, &config);

        // println!("{:?}\r\n--------\r\n{:?}", result, expected);
        assert!(arrays_almost_equal(&expected, &result.into_dyn()));
    }

    #[test]
    fn test_gpu_cpu_equal_inputs_grad() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let config = ConvolutionConfig {
            in_channels: 3,
            out_channels: 4,
            kernel_size: 2,
            padding: 1,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            stride: 2,
            cache: false,
        };
        let inputs = Array4F::random((8, config.in_channels, 10, 10), &dist);
        let grad_shape = (inputs.shape()[0], config.out_channels,
                          (inputs.shape()[2] - config.kernel_size) / config.stride + 1,
                          (inputs.shape()[3] - config.kernel_size) / config.stride + 1);

        let grad = Array4F::random(grad_shape, &dist);
        let kernel = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist);

        let expected = cpu_inputs_grad(inputs.clone(), grad.clone(), kernel.clone(), &config);
        let actual = gpu_inputs_grad(String::new(), &inputs, &grad, &kernel, get_global_gpu().unwrap(), &config).unwrap();
        println!("{:?}\n---------\n{:?}", actual, expected);
        assert!(arrays_almost_equal(&expected, &actual));
    }
}