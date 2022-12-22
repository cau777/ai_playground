use std::iter::zip;
use ndarray::{Axis, concatenate, Slice};
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{backward_layer, forward_layer, Layer, train_layer, TrainableLayerOps, TrainData};
use crate::utils::Array1F;

use super::nn_layers::{ForwardData, LayerOps, InitData, EmptyLayerResult, LayerResult, BackwardData};

pub struct ConcatLayer;

#[derive(Clone, Debug)]
pub struct ConcatConfig {
    pub dim: usize,
    pub layers: Vec<Layer>,
}

fn gen_name(config: &ConcatConfig) -> String {
    format!("concat_{}", config.dim)
}

impl LayerOps<ConcatConfig> for ConcatLayer {
    fn init(_: InitData, _: &ConcatConfig) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, layer_config: &ConcatConfig) -> LayerResult {
        let mut results = Vec::with_capacity(layer_config.layers.len());
        let mut splits = Vec::with_capacity(layer_config.layers.len());
        let key = data.assigner.get_key(gen_name(layer_config));
        let dim = layer_config.dim + 1;

        for layer in &layer_config.layers {
            let result = forward_layer(layer, ForwardData {
                inputs: data.inputs.clone(),
                forward_cache: data.forward_cache,
                storage: data.storage,
                gpu: data.gpu.clone(),
                assigner: data.assigner,
                batch_config: data.batch_config,
            })?;

            splits.push(result.shape()[dim]);
            results.push(result);
        }

        let results_views: Vec<_> = results.iter().map(|o| o.view()).collect();
        let concat = concatenate(Axis(dim), &results_views)?;

        data.forward_cache.insert(key, vec![Array1F::from_iter(splits.iter().map(|o| *o as f32)).into_dyn()]);

        Ok(concat)
    }

    fn backward(data: BackwardData, layer_config: &ConcatConfig) -> LayerResult {
        let key = data.assigner.get_key(gen_name(layer_config));
        let [cache] = remove_from_storage1(data.forward_cache, &key);
        let dim = layer_config.dim + 1;
        let mut prev_split = 0;
        let mut result = None;

        for (layer, split) in zip(&layer_config.layers, cache.iter().map(|o| o.round() as usize)) {
            let grad = data.grad.slice_axis(Axis(dim), Slice::from(prev_split..(prev_split + split)));

            let layer_result = backward_layer(layer, BackwardData {
                grad: grad.to_owned(),
                batch_config: data.batch_config,
                assigner: data.assigner,
                storage: data.storage,
                gpu: data.gpu.clone(),
                forward_cache: data.forward_cache,
                backward_cache: data.backward_cache,
            })?;

            println!("{:?}", result);
            result = Some(match result {
                Some(v) => v + layer_result,
                None => layer_result,
            });
            prev_split += split;
        }

        Ok(result.unwrap())
    }
}

impl TrainableLayerOps<ConcatConfig> for ConcatLayer {
    fn train(data: TrainData, layer_config: &ConcatConfig) -> EmptyLayerResult {
        for layer in &layer_config.layers {
            train_layer(layer, TrainData {
                storage: data.storage,
                batch_config: data.batch_config,
                backward_cache: data.backward_cache,
                assigner: data.assigner,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::nn_layers::GenericStorage;
    use crate::nn::layers::sequential_layer::SequentialConfig;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward() {
        let inputs = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]].into_dyn();
        let expected = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ].into_dyn();

        let config = ConcatConfig {
            dim: 0,
            layers: vec![
                Layer::Sequential(SequentialConfig { layers: vec![] }),
                Layer::Sequential(SequentialConfig { layers: vec![] }),
                Layer::Sequential(SequentialConfig { layers: vec![] }),
            ],
        };

        let cache = &mut GenericStorage::new();
        let result = ConcatLayer::forward(ForwardData {
            batch_config: &BatchConfig::new_not_train(),
            gpu: None,
            forward_cache: cache,
            inputs,
            assigner: &mut KeyAssigner::new(),
            storage: &GenericStorage::new(),
        }, &config).unwrap();

        assert!(arrays_almost_equal(&result, &expected));
        let shape: Vec<_> = cache["concat_0_0"][0].iter().map(|o| o.round() as usize).collect();
        assert_eq!(&shape, &vec![2, 2, 2]);
    }

    #[test]
    fn test_backward() {
        let inputs = array![[
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        ]].into_dyn();
        let expected = array![[[3.0, 6.0, 9.0], [12.0, 15.0, 18.0]]].into_dyn();

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("concat_0_0".to_owned(), vec![array![2.0, 2.0, 2.0].into_dyn()]);

        let config = ConcatConfig {
            dim: 0,
            layers: vec![
                Layer::Sequential(SequentialConfig { layers: vec![] }),
                Layer::Sequential(SequentialConfig { layers: vec![] }),
                Layer::Sequential(SequentialConfig { layers: vec![] }),
            ],
        };
        let result = ConcatLayer::backward(BackwardData {
            forward_cache: &mut forward_cache,
            backward_cache: &mut GenericStorage::new(),
            gpu: None,
            assigner: &mut KeyAssigner::new(),
            batch_config: &BatchConfig::new_not_train(),
            storage: &mut GenericStorage::new(),
            grad: inputs.into_dyn(),
        }, &config).unwrap();

        assert!(arrays_almost_equal(&expected.into_dyn(), &result));
    }
}