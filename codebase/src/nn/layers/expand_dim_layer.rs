use ndarray::Axis;

use super::nn_layers::{ForwardData, LayerOps, InitData, EmptyLayerResult, LayerResult, BackwardData};

pub struct ExpandDimLayer{}

#[derive(Clone, Debug)]
pub struct ExpandDimConfig {
    pub dim: usize,
}

impl LayerOps<ExpandDimConfig> for ExpandDimLayer {
    fn init(_: InitData, _: &ExpandDimConfig) -> EmptyLayerResult {Ok(())}

    fn forward(data: ForwardData, layer_config: &ExpandDimConfig) -> LayerResult {
        let mut inputs = data.inputs;
        inputs.insert_axis_inplace(Axis(layer_config.dim+1));
        Ok(inputs)
    }

    fn backward(data: BackwardData, layer_config: &ExpandDimConfig) -> LayerResult {
        let grad = data.grad;
        Ok(grad.remove_axis(Axis(layer_config.dim+1)))
    }
}