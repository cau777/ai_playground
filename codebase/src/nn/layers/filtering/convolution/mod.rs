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
}

#[derive(Clone, Debug)]
pub enum ConvolutionInitMode {
    Kernel(Array4F),
    HeNormal(),
}

pub struct ConvolutionLayer;

fn gen_name(config: &ConvolutionConfig) -> String {
    format!("convolution_{}_{}", config.in_channels, config.out_channels)
}

impl LayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        conv_init::init(data, layer_config)
    }

    fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        conv_forward::forward(data, layer_config)
    }

    fn backward(data: BackwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        conv_backward::backward(data, layer_config)
    }
}