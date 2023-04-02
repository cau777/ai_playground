mod concat_forward;
mod concat_backward;

use crate::nn::layers::nn_layers::*;

/// Feed the same input to all its immediate children and concatenate the result.
/// No extra axis is created
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
        concat_forward::forward(data, layer_config)
    }

    fn backward(data: BackwardData, layer_config: &ConcatConfig) -> LayerResult {
        concat_backward::backward(data, layer_config)
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
    use crate::nn::layers::concat_layer::ConcatLayer;
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