use ndarray::Axis;

use super::nn_layers::{ForwardData, LayerOps, InitData, EmptyLayerResult, LayerResult, BackwardData};

/// Adds a new axis of length 1 in the specified place. By default, it ignores Batch dimension (which is always
/// the first), so adding an axis in 1 would result: (Batch, Dim_1, 1, Dim_2, Dim_3).
pub struct ExpandDimLayer;

#[derive(Clone, Debug)]
pub struct ExpandDimConfig {
    pub dim: usize,
}

impl LayerOps<ExpandDimConfig> for ExpandDimLayer {
    fn init(_: InitData, _: &ExpandDimConfig) -> EmptyLayerResult {Ok(())}

    fn forward(data: ForwardData, layer_config: &ExpandDimConfig) -> LayerResult {
        let mut inputs = data.inputs.into_memory()?;
        inputs.insert_axis_inplace(Axis(layer_config.dim+1));
        Ok(inputs.into())
    }

    fn backward(data: BackwardData, layer_config: &ExpandDimConfig) -> LayerResult {
        let grad = data.grad;
        Ok(grad.remove_axis(Axis(layer_config.dim+1)).into())
    }
}