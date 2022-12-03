use std::ops::AddAssign;
use ndarray::{Axis, s, stack};
use ndarray::parallel::prelude::*;
use crate::Array4F;
use crate::nn::layers::convolution::convolution_layer::ConvolutionConfig;
use crate::utils::{Array2F, Array5F, get_dims_after_filter_4, GetBatchSize};

pub fn pad4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2];
    let width = shape[3];
    let mut result = Array4F::zeros(
        (
            shape[0],
            shape[1],
            height + 2 * padding,
            width + 2 * padding,
        ),
    );
    let mut slice = result.slice_mut(s![
        ..,
        ..,
        padding..height + padding,
        padding..width + padding
    ]);
    slice.assign(&array);
    result
}

pub fn remove_padding_4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2] - padding;
    let width = shape[3] - padding;
    array.slice_move(s![.., .., padding..height, padding..width])
}

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