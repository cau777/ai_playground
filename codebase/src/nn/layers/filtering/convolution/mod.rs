use crate::Array4F;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};
use crate::nn::lr_calculators::lr_calculator::LrCalc;

mod conv_forward;
mod conv_init;
mod conv_backward;
mod conv_train;

#[cfg(test)]
mod test_values;

#[derive(Clone, Debug)]
pub struct ConvolutionConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub init_mode: ConvolutionInitMode,
    pub lr_calc: LrCalc,
    pub cache: bool,
}

#[derive(Clone, Debug)]
pub enum ConvolutionInitMode {
    Kernel(Array4F),
    HeNormal(),
}

/// Apply the convolution operation with 2D filters. That means passing a filter through the last
/// 2 dimension of the input (usually height and width). In position of the filter, the sum of the
/// product of those input values and the kernel is computed. Requires a 4 dimensional input (one being
/// the batch).
/// ### Trainable
/// * Kernel
/// https://en.wikipedia.org/wiki/Convolutional_neural_network
pub struct ConvolutionLayer;

fn gen_name(config: &ConvolutionConfig) -> String {
    format!("convolution_{}_{}_{}_{}_{}", config.in_channels, config.out_channels, config.kernel_size,
            config.stride, config.padding)
}

impl LayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        conv_init::init(data, layer_config)
    }

    #[inline(never)]
    fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        conv_forward::forward(data, layer_config)
    }

    #[inline(never)]
    fn backward(data: BackwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        conv_backward::backward(data, layer_config)
    }
}