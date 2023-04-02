use ndarray::{Axis, stack, Zip};
use ndarray::parallel::prelude::*;
use crate::nn::generic_storage::*;
use crate::nn::layers::dense_layer::{DenseConfig, gen_name};
use crate::nn::layers::nn_layers::*;
use crate::utils::{Array2F, GetBatchSize};

/// Calculates the weights' error by performing matrix multiplication between the gradient and the inputs.
/// Calculates the biases' error as a mean of the gradient.
/// Outputs the matrix multiplication between the gradient and the weights.
pub fn backward(data: BackwardData, layer_config: &DenseConfig) -> LayerResult {
    let BackwardData {
        assigner,
        storage,
        forward_cache,
        grad,
        backward_cache,
        ..
    } = data;
    let key = assigner.get_key(gen_name(layer_config));

    let [weights] = clone_from_storage1(storage, &key);
    let weights: Array2F = weights.into_dimensionality()?;

    let [inputs] = remove_from_storage1(forward_cache, &key);
    let inputs: Array2F = inputs.into_dimensionality()?;

    let grad: Array2F = grad.into_dimensionality()?;

    let batches = inputs.shape()[0];
    // Divide by number of weights to avoid exploding weights
    let factor = 1.0 / (batches * layer_config.out_values * layer_config.in_values) as f32;

    let weights_error = Zip::from(inputs.outer_iter()).and(grad.outer_iter())
        .into_par_iter()
        .map(|(i, g)| {
            let gt = g.insert_axis(Axis(1));
            let it = i.insert_axis(Axis(0));
            gt.dot(&it)
        })
        .map(|o| o * factor)
        .reduce(|| Array2F::default((layer_config.out_values, layer_config.in_values)),
                |acc, val| acc + val);

    let biases_grad = grad.mean_axis(Axis(0)).unwrap().into_dyn();

    let weights_grad = weights_error.into_dyn();
    backward_cache.insert(key, vec![weights_grad, biases_grad]);

    let weights_t = weights.t();
    let mut dot_prod = Vec::with_capacity(inputs.batch_size());
    grad.outer_iter()
        .into_par_iter()
        .map(|o| weights_t.dot(&o))
        .collect_into_vec(&mut dot_prod);

    let mut views = Vec::with_capacity(inputs.batch_size());
    views.extend(dot_prod.iter().map(|o| o.view()));
    Ok(stack(Axis(0), &views)?.into_dyn().into())
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::dense_layer::{DenseLayer, DenseLayerInit};
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{Array1F, arrays_almost_equal};
    use super::*;

    #[test]
    fn test_backward() {
        let inputs: Array2F = array![[0.9, 0.7, 0., 0.5, 0.6], [0.3, 0.2, 0.6, 0.8, 0.2]];
        let weights: Array2F = array![
            [0.9, 0.9, 0.2, 0.7, 0.3],
            [0.5, 0.6, 0.3, 0., 0.1],
            [0.4, 0.6, 0.3, 0.9, 0.3]
        ];
        let biases: Array1F = array![0.8, 0.1, 0.9];
        let grad: Array2F = array![[-2.37, -0.43, -2.11], [-1.79, -0.37, -1.8]];
        let expected: Array2F = array![
            [-3.192, -3.657, -1.236, -3.558, -1.387],
            [-2.516, -2.913, -1.009, -2.873, -1.114]
        ];
        let expected_weights_grad: Array2F = array![
            [-0.089, -0.06723334, -0.0358, -0.0872, -0.059],
            [-0.0166, -0.0125, -0.0074, -0.01703, -0.011],
            [-0.0813, -0.0612, -0.036, -0.08316, -0.0542]
        ];
        let expected_biases_grad: Array1F = array![-2.08, -0.4, -1.955];
        let config = DenseConfig {
            in_values: 5,
            out_values: 3,
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
            init_mode: DenseLayerInit::WeightsAndBiases(weights, biases),
        };

        let mut storage = GenericStorage::new();
        DenseLayer::init(
            InitData {
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
            },
            &config,
        )
            .unwrap();

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("dense_5_3_0".to_owned(), vec![inputs.into_dyn()]);
        let mut backward_cache = GenericStorage::new();
        let result = backward(
            BackwardData {
                grad: grad.into_dyn(),
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                forward_cache: &mut forward_cache,
                storage: &mut storage,
                backward_cache: &mut backward_cache,
                gpu: None,
            },
            &config,
        )
            .unwrap();

        assert!(arrays_almost_equal(
            &expected,
            &result.into_memory().unwrap().into_dimensionality().unwrap(),
        ));
        let cache = &backward_cache["dense_5_3_0"];

        assert!(arrays_almost_equal(&expected_weights_grad, &cache[0].clone().into_dimensionality().unwrap()));
        assert!(arrays_almost_equal(&expected_biases_grad, &cache[1].clone().into_dimensionality().unwrap()));
    }
}