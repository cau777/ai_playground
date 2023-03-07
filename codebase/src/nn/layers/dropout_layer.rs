use ndarray_rand::RandomExt;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};
use crate::utils::Array1F;

pub struct DropoutLayer;

#[derive(Clone, Debug)]
pub struct DropoutConfig {
    pub drop: f32,
}

fn gen_name(config: &DropoutConfig) -> String {
    format!("dropout_{}", config.drop)
}

impl LayerOps<DropoutConfig> for DropoutLayer {
    fn init(_: InitData, _: &DropoutConfig) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, layer_config: &DropoutConfig) -> LayerResult {
        let ForwardData { forward_cache, assigner, inputs, batch_config, .. } = data;
        let key = assigner.get_key(gen_name(layer_config));
        let inputs = inputs.into_memory()?;

        if batch_config.is_training { // Only perform dropout while training
            let factor = layer_config.drop;
            let length = inputs.shape().iter().copied().reduce(|acc, val| acc * val).unwrap_or(1);
            let dist = ndarray_rand::rand_distr::Uniform::new(0.0, 1.0);
            let dropout = Array1F::random(length, &dist)
                .mapv_into(|o| if o < factor { 0.0 } else { 1.0 })
                .into_shape(inputs.shape())?;

            let result = inputs * &dropout;
            forward_cache.insert(key, vec![dropout]);
            Ok(result.into())
        } else {
            forward_cache.insert(key, vec![]);
            Ok(inputs.into())
        }
    }

    fn backward(data: BackwardData, layer_config: &DropoutConfig) -> LayerResult {
        let BackwardData {forward_cache, assigner, grad, ..} = data;
        let key = assigner.get_key(gen_name(layer_config));
        match forward_cache[&key].as_slice() {
            [dropout] => {
                Ok((grad * dropout).into())
            }
            _ => {
                Ok(grad.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::nn_layers::GenericStorage;
    use crate::utils::Array2F;
    use super::*;

    #[test]
    fn test_manual() {
        let dist = ndarray_rand::rand_distr::Uniform::new(0.0, 1.0);
        let inputs = Array2F::random((5, 5), &dist).into_dyn();
        let mut cache = GenericStorage::new();
        let config = DropoutConfig{drop: 0.1};
        let batch_config = BatchConfig::new_train();

        let forward_data = ForwardData {
            inputs: inputs.into(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut cache,
            storage: &GenericStorage::new(),
            batch_config:  &batch_config,
            prev_iteration_cache: None,
            gpu: None,
        };
        let result = DropoutLayer::forward(forward_data, &config).unwrap().into_memory().unwrap();
        println!("{:?}", result);
        println!("{:?}", cache);
    }
}