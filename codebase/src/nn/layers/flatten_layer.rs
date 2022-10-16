use crate::utils::{Array1F};

use super::nn_layers::{LayerOps, LayerResult, EmptyLayerResult, ForwardData, BackwardData};

pub struct FlattenLayer;

fn gen_name() -> String {
    "Flatten".to_owned()
}

impl LayerOps<()> for FlattenLayer {
    fn init(_: super::nn_layers::InitData, _: &()) -> EmptyLayerResult {
        Ok(())
    }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { inputs, assigner, forward_cache,.. } = data;
        
        // Skip the first axis (batch) and multiply the others
        let flat = inputs.shape().iter().skip(1).cloned().reduce(|acc, v| acc * v).unwrap();
        let new_shape = [inputs.shape()[0], flat];
        
        let key = assigner.get_key(gen_name());
        let shape_vec = inputs.shape().iter().cloned().map(|o|o as f32).collect();
        let shape_array = Array1F::from_shape_vec(inputs.shape().len(), shape_vec).unwrap();
        forward_cache.insert(key, vec![shape_array.into_dyn()]);
        
        Ok(inputs.into_shape(new_shape)?.into_dyn())
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData {grad, assigner, forward_cache,..} = data;
        let key = assigner.get_key(gen_name());
        let mut stored = forward_cache.remove(&key).unwrap();
        let stored = stored.remove(0);
        let shape_vec: Vec<_> = stored.iter().map(|o| o.round() as usize).collect();
        
        Ok(grad.into_shape(shape_vec)?)
    }
}