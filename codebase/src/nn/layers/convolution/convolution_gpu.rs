use std::error::Error;
use crate::Array4F;
use crate::gpu::shader_runner::{GlobalGpu, ShaderRunner};
use crate::gpu::shaders;
use crate::nn::layers::convolution::convolution_layer::ConvolutionConfig;
use crate::utils::{get_dims_after_filter_4, ShapeAsArray};

pub fn calc_inputs_grad_gpu(inputs: &Array4F, grad: &Array4F, kernel: &Array4F, gpu: GlobalGpu, layer_config: &ConvolutionConfig) -> Result<Array4F, Box<dyn Error>> {
    let ish = inputs.shape_arr();
    let out_shape = (ish[0], ish[1], ish[2] - 2 * layer_config.padding, ish[3] - 2 * layer_config.padding);
    let ish = ish.map(|o| o as u32);
    let gsh = grad.shape_arr().map(|o| o as u32);

    let mut runner = ShaderRunner::new(gpu, shaders::convolution_inputs_grad::load,
                                       "main", &shaders::convolution_inputs_grad::SpecializationConstants {
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
            out_height: out_shape.2 as u32,
            out_width: out_shape.3 as u32,
        })?;

    let results_buffer = runner.create_buffer_from_array(&Array4F::zeros(out_shape))?;
    runner.create_buffer_from_array(kernel)?;
    runner.create_buffer_from_array(grad)?;

    runner.execute([out_shape.0 * out_shape.1, out_shape.2, out_shape.3].map(|o| o as u32), shaders::convolution_inputs_grad::BLOCK_SIZE)?;
    let vec = results_buffer.read()?.to_vec();
    Ok(Array4F::from_shape_vec(out_shape, vec)?)
}

pub fn calc_forward_gpu(inputs: &Array4F, kernel: &Array4F, gpu: GlobalGpu, layer_config: &ConvolutionConfig) -> Result<Array4F, Box<dyn Error>> {
    let ConvolutionConfig { stride, kernel_size, .. } = layer_config;
    let [batch_size, _, new_height, new_width] = get_dims_after_filter_4(&inputs, *kernel_size, *stride);

    let ish = inputs.shape();
    let out_shape = (batch_size, layer_config.out_channels, new_height, new_width);
    let mut runner = ShaderRunner::new(gpu, shaders::convolution_forward::load, "main",&shaders::convolution_forward::SpecializationConstants {
        in_channels: layer_config.in_channels as u32,
        out_channels: layer_config.out_channels as u32,
        kernel_size: layer_config.kernel_size as u32,
        stride: layer_config.stride as u32,
        out_height: out_shape.2 as u32,
        out_width: out_shape.3 as u32,
        input_height: ish[2] as u32,
        input_width: ish[3] as u32,
    })?;

    let results_buffer = runner.create_buffer_from_array(&Array4F::zeros(out_shape))?;
    runner.create_buffer_from_array(kernel)?;
    runner.create_buffer_from_array(inputs)?;

    runner.execute([out_shape.0 * out_shape.1, out_shape.2, out_shape.3].map(|o| o as u32), shaders::convolution_forward::BLOCK_SIZE)?;
    let vec = results_buffer.read()?.to_vec();
    Ok(Array4F::from_shape_vec(out_shape, vec)?)
}