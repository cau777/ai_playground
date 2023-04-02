use crate::nn::layers::nn_layers::*;
use crate::utils::{Array2F};

pub struct TwoComplementsTransformerLayer;

/// Receives 2 inputs A and B, and outputs A - B
/// It's useful because it makes it easier for the model to produce negative values with ReLu layers
impl LayerOps<()> for TwoComplementsTransformerLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult {
        Ok(())
    }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let inputs: Array2F = data.inputs.into_memory()?.into_dimensionality()?;

        if inputs.shape()[1] == 2 {
            Ok(Array2F::from_shape_fn((inputs.shape()[0], 1), |(b, _)| {
                inputs[(b, 0)] - inputs[(b, 1)]
            }).into_dyn().into())
        } else {
            Err(anyhow::anyhow!("TwoComplementsTransformerLayer needs exactly 2 values as inputs"))?
        }
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let grad: Array2F = data.grad.into_dimensionality()?;

        if grad.shape()[1] != 1 {
            Err(anyhow::anyhow!("TwoComplementsTransformerLayer needs exactly 1 value as gradient"))?
        }

        Ok(Array2F::from_shape_fn((grad.shape()[0], 2), |(b, i)| {
            if i == 0 { grad[(b, 0)] } else { -grad[(b, 0)] }
        }).into_dyn().into())
    }
}