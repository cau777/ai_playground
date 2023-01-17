use std::ops::AddAssign;
use ndarray::{ArrayView3, ArrayViewMut3, Axis, s, ShapeError, stack, Zip};
use ndarray::parallel::prelude::*;
use crate::Array4F;
use crate::nn::layers::filtering::convolution::convolution_layer::ConvolutionConfig;
use crate::nn::layers::filtering::{find_useful_from_prev, remove_padding_4d};
use crate::utils::{Array2F, Array3F, Array5F, get_dims_after_filter_4, GetBatchSize};

pub fn calc_inputs_grad(inputs: Array4F, grad: Array4F, kernel: Array4F, layer_config: &ConvolutionConfig) -> Array4F {
    let inputs_shape = inputs.shape();
    let ConvolutionConfig { kernel_size, stride, padding, .. } = layer_config;

    // Put height and width in front
    let grad = grad.permuted_axes((2, 3, 0, 1));
    let kernel = kernel.permuted_axes((1, 2, 3, 0));
    let kernel = kernel.insert_axis(Axis(3));

    let [batch_size, in_channels, new_height, new_width] =
        get_dims_after_filter_4(&inputs, *kernel_size, *stride);

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

pub fn calc_kernel_grad(inputs: &Array4F, grad: &Array4F, layer_config: &ConvolutionConfig) -> Array4F {
    let ConvolutionConfig { in_channels, out_channels, kernel_size, stride, .. } = layer_config;
    let kernel_size = *kernel_size;
    let stride = *stride;

    let shape = inputs.shape();
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

    let mut views = Vec::with_capacity(inputs.batch_size());
    views.extend(parts.iter().map(|o| o.view()));

    let joined = stack(Axis(2), &views).unwrap();
    let mut reshaped = joined.into_shape((*out_channels, *in_channels, kernel_size, kernel_size)).unwrap();
    reshaped.swap_axes(2, 3);

    reshaped / (layer_config.kernel_size.pow(2) as f32)
}

pub fn calc_forward(inputs: &Array4F, kernel: &Array4F, layer_config: &ConvolutionConfig) -> Result<Array4F, ShapeError> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;

    let [_, _, new_height, new_width] = get_dims_after_filter_4(inputs, *kernel_size, *stride);
    let mut batches = Vec::with_capacity(inputs.batch_size());
    inputs.outer_iter().into_par_iter()
        .map(|batch| {
            let mut result = Array3F::zeros((layer_config.out_channels, new_height, new_width));
            for h in 0..new_height {
                for w in 0..new_width {
                    apply_conv_filter(kernel, stride, kernel_size, &batch, &mut result.view_mut(), h, w);
                }
            }
            result
        })
        .collect_into_vec(&mut batches);

    let mut views = Vec::with_capacity(inputs.batch_size());
    views.extend(batches.iter().map(|o| o.view()));
    stack(Axis(0), &views)
}

pub fn calc_forward_with_cache(inputs: &Array4F, prev_inputs: &Array4F, prev_results: &Array4F, kernel: &Array4F, layer_config: &ConvolutionConfig) -> Result<Array4F, ShapeError> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;

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
                    for och in 0..layer_config.out_channels {
                        let cached = cache[(och, h, w)];
                        match cached {
                            Some(v) => {
                                result[(och, h, w)] = v;
                            }
                            None => {
                                apply_conv_filter(kernel, stride, kernel_size, &inputs, &mut result, h, w);
                                break;
                            }
                        }
                    }
                }
            }
        });

    Ok(result)
}

fn apply_conv_filter(kernel: &Array4F, stride: &usize, kernel_size: &usize, batch: &ArrayView3<f32>, result: &mut ArrayViewMut3<f32>, h: usize, w: usize) {
    let h_offset = h * stride;
    let w_offset = w * stride;
    let area = batch.slice(s![
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
    use crate::nn::layers::filtering::convolution::convolution_gpu::{calc_forward_gpu, calc_inputs_grad_gpu};
    use crate::nn::layers::filtering::convolution::ConvolutionInitMode::HeNormal;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::arrays_almost_equal;
    use super::*;

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
        };
        let inputs = Array4F::random((8, config.in_channels, 10, 10), &dist);
        let grad_shape = (inputs.shape()[0], config.out_channels,
                          (inputs.shape()[2] - config.kernel_size) / config.stride + 1,
                          (inputs.shape()[3] - config.kernel_size) / config.stride + 1);

        let grad = Array4F::random(grad_shape, &dist);
        let kernel = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist);

        let expected = calc_inputs_grad(inputs.clone(), grad.clone(), kernel.clone(), &config);
        let actual = calc_inputs_grad_gpu(&inputs, &grad, &kernel, GpuData::new_global().unwrap(), &config).unwrap();
        assert!(arrays_almost_equal(&expected, &actual));
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
        };

        let dist = Normal::new(0.0, 1.0).unwrap();
        let inputs = Array4F::random((8, config.in_channels, 5, 5), &dist);
        let kernels = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist);
        let expected = calc_forward(&inputs, &kernels, &config).unwrap();
        let actual = calc_forward_gpu(&inputs, &kernels, GpuData::new_global().unwrap(), &config).unwrap();

        println!("{:?}\n\n------\n\n{:?}", expected, actual);
        assert!(arrays_almost_equal(&expected, &actual));
    }
}