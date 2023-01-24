use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};
use crate::utils::{Array1F, Array2F};

pub struct TwoComplementsTransformerLayer;

impl LayerOps<()> for TwoComplementsTransformerLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult {
        Ok(())
    }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let inputs: Array2F = data.inputs.into_dimensionality()?;

        if inputs.shape()[1] == 2 {
            Ok(Array2F::from_shape_fn((inputs.shape()[0], 1), |(b, _)| {
                inputs[(b, 0)] - inputs[(b, 1)]
            }).into_dyn())
        } else {
            Err("TwoComplementsTransformerLayer needs exactly 2 values as inputs")?
        }
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let grad: Array2F = data.grad.into_dimensionality()?;

        if grad.shape()[1] != 1 {
            Err("TwoComplementsTransformerLayer needs exactly 1 value as gradient")?
        }

        Ok(Array2F::from_shape_fn((grad.shape()[0], 2), |(b, i)| {
            grad[(b, 0)] * if i == 0 { 0.5 } else { -0.5 }
        }).into_dyn())
    }
}