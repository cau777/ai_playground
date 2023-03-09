use ndarray::{Axis, concatenate, Slice};
use crate::ArrayDynF;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{backward_layer, forward_layer, init_layer, Layer, train_layer, TrainableLayerOps, TrainData};
use crate::nn::layers::stored_array::StoredArray;
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
    fn init(data: InitData, layer_config: &ConcatConfig) -> EmptyLayerResult {
        for layer in layer_config.layers.iter() {
            init_layer(layer, InitData {
                assigner: data.assigner,
                storage: data.storage,
            })?;
        }
        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &ConcatConfig) -> LayerResult {
        let mut results = Vec::with_capacity(layer_config.layers.len());
        let mut splits = Vec::with_capacity(layer_config.layers.len());
        let key = data.assigner.get_key(gen_name(layer_config));
        let dim = layer_config.dim + 1;

        let ForwardData {
            inputs, mut forward_cache, storage, gpu, assigner,
            batch_config, mut prev_iteration_cache
        } = data;
        let inputs = inputs.into_memory()?;

        for layer in &layer_config.layers {
            let result = forward_layer(layer, ForwardData {
                inputs: inputs.clone().into(),
                forward_cache: forward_cache.as_deref_mut(),
                storage,
                gpu: gpu.clone(),
                assigner,
                batch_config,
                prev_iteration_cache: prev_iteration_cache.as_deref_mut(),
            })?;

            splits.push(result.shape()[dim]);
            results.push(result.into_memory()?);
        }

        let results_views: Vec<_> = results.iter().map(|o| o.view()).collect();
        let concat = concatenate(Axis(dim), &results_views)?;

        if let Some(forward_cache) = forward_cache {
            forward_cache.insert(key, vec![Array1F::from_iter(splits.iter().map(|o| *o as f32)).into_dyn()]);
        }

        Ok(StoredArray::Memory { data: concat })
    }

    fn backward(data: BackwardData, layer_config: &ConcatConfig) -> LayerResult {
        let key = data.assigner.get_key(gen_name(layer_config));
        let [cache] = remove_from_storage1(data.forward_cache, &key);
        let dim = layer_config.dim + 1;
        let splits: Vec<_> = cache.iter().map(|o| o.round() as usize).collect();

        let mut end = data.grad.shape()[dim];
        let mut result: Option<ArrayDynF> = None;
        let layer_count = layer_config.layers.len();

        for i in 0..layer_count {
            let i = layer_count - i - 1;
            let layer = &layer_config.layers[i];
            let split = splits[i];
            let grad = data.grad.slice_axis(Axis(dim), Slice::from((end - split)..end));

            let layer_result = backward_layer(layer, BackwardData {
                grad: grad.to_owned(),
                batch_config: data.batch_config,
                assigner: data.assigner,
                storage: data.storage,
                gpu: data.gpu.clone(),
                forward_cache: data.forward_cache,
                backward_cache: data.backward_cache,
            })?.into_memory()?;

            result = Some(match result {
                Some(v) => v + layer_result,
                None => layer_result,
            });
            end -= split;
        }

        Ok(result.unwrap().into())
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

// TODO: test - doesn't affect single layer output
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
            forward_cache: Some(cache),
            inputs: inputs.into(),
            assigner: &mut KeyAssigner::new(),
            storage: &GenericStorage::new(),
            prev_iteration_cache: None,
        }, &config).unwrap();

        assert!(arrays_almost_equal(&result.into_memory().unwrap(), &expected));
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

        assert!(arrays_almost_equal(&expected.into_dyn(), &result.into_memory().unwrap()));
    }
}