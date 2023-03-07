use crate::nn::layers::nn_layers::*;
use crate::utils::{Array2F};

pub struct TwoComplementsTransformerLayer;

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
            Err("TwoComplementsTransformerLayer needs exactly 2 values as inputs")?
        }
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let grad: Array2F = data.grad.into_dimensionality()?;

        if grad.shape()[1] != 1 {
            Err("TwoComplementsTransformerLayer needs exactly 1 value as gradient")?
        }

        Ok(Array2F::from_shape_fn((grad.shape()[0], 2), |(b, i)| {
            if i == 0 { grad[(b, 0)] } else { -grad[(b, 0)] }
        }).into_dyn().into())
    }
}