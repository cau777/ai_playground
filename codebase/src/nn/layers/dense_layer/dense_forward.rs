use crate::nn::generic_storage::clone_from_storage2;
use crate::nn::layers::dense_layer::*;
use crate::nn::layers::nn_layers::*;
use crate::utils::{Array1F, Array2F};

pub fn forward(data: ForwardData, layer_config: &DenseConfig) -> LayerResult {
    let ForwardData {
        assigner,
        storage,
        inputs,
        forward_cache,
        ..
    } = data;
    let key = assigner.get_key(gen_name(layer_config));

    let [weights, biases] = clone_from_storage2(storage, &key);
    let weights: Array2F = weights.into_dimensionality()?;
    let biases: &Array1F = &biases.into_dimensionality()?;

    let inputs: Array2F = inputs.into_memory()?.into_dimensionality()?;

    let mut dot_prod = Vec::with_capacity(inputs.batch_size());
    inputs
        .outer_iter()
        .into_par_iter()
        .map(|o| weights.dot(&o))
        .map(|o| o + biases)
        .collect_into_vec(&mut dot_prod);

    let mut views = Vec::with_capacity(inputs.batch_size());
    views.extend(dot_prod.iter().map(|o| o.view()));

    let result = stack(Axis(0), &views)?;

    if let Some(forward_cache) = forward_cache {
        forward_cache.insert(key, vec![inputs.into_dyn()]);
    }
    Ok(result.into_dyn().into())
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::dense_layer::tests::get_config;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward() {
        let input = array![[1.0, 2.0], [2.0, 3.0]].into_dyn();
        let weights = array![[0.7, 0.0], [0.1, 0.4], [0.8, 0.6]];
        let expected = array![[0.7, 0.9, 2.0], [1.4, 1.4, 3.4]].into_dyn();

        let config = get_config(DenseLayerInit::WeightsAndBiases(weights, Array1F::zeros(3)));

        let mut storage = GenericStorage::new();
        DenseLayer::init(
            InitData {
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
            },
            &config,
        )
            .unwrap();

        let output = forward(
            ForwardData {
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                storage: &mut storage,
                inputs: input.into(),
                forward_cache: None,
                prev_iteration_cache: None,
                gpu: None,
            },
            &config,
        ).unwrap();

        assert!(arrays_almost_equal(&output.into_memory().unwrap(), &expected));
    }
}