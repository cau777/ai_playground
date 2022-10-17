use crate::nn::generic_storage::{
    clone_from_storage1, clone_from_storage2, get_mut_from_storage, remove_from_storage1,
    remove_from_storage2,
};
use crate::nn::layers::nn_layers::{
    BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult, TrainData,
    TrainableLayerOps,
};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};
use crate::utils::{Array1F, Array2F};
use ndarray::{s, Axis, ShapeBuilder};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::iter::zip;
use std::ops::AddAssign;

#[derive(Clone, Debug)]
pub struct DenseLayerConfig {
    pub out_values: usize,
    pub in_values: usize,
    pub init_mode: DenseLayerInit,
    pub weights_lr_calc: LrCalc,
    pub biases_lr_calc: LrCalc,
}

#[derive(Clone, Debug)]
pub enum DenseLayerInit {
    WeightsAndBiases(Array2F, Array1F),
    Random(),
}

pub struct DenseLayer {}

fn gen_name(config: &DenseLayerConfig) -> String {
    format!("dense_{}_{}", config.in_values, config.out_values)
}

impl LayerOps<DenseLayerConfig> for DenseLayer {
    fn init(data: InitData, layer_config: &DenseLayerConfig) -> EmptyLayerResult {
        let InitData { assigner, storage } = data;
        let key = assigner.get_key(gen_name(layer_config));

        if !storage.contains_key(&key) {
            let weights: Array2F;
            let biases: Array1F;

            match &layer_config.init_mode {
                DenseLayerInit::WeightsAndBiases(w, b) => {
                    weights = w.clone();
                    biases = b.clone();
                }
                DenseLayerInit::Random() => {
                    let std_dev = (layer_config.out_values as f32).powf(-0.5);
                    let dist = Normal::new(0.0, std_dev)?;
                    weights = Array2F::random(
                        (layer_config.out_values, layer_config.in_values).f(),
                        dist,
                    );
                    biases = Array1F::zeros((layer_config.out_values).f());
                }
            }

            storage.insert(key, vec![weights.into_dyn(), biases.into_dyn()]);
        }

        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &DenseLayerConfig) -> LayerResult {
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

        let inputs: Array2F = inputs.into_dimensionality()?;

        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, layer_config.out_values).f());

        inputs
            .outer_iter()
            .map(|o| weights.dot(&o))
            .map(|o| o + biases)
            .enumerate()
            .for_each(|(index, o)| result.slice_mut(s![index, ..]).assign(&o));

        forward_cache.insert(key, vec![inputs.into_dyn()]);
        Ok(result.into_dyn())
    }

    fn backward(data: BackwardData, layer_config: &DenseLayerConfig) -> LayerResult {
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
        let mut weights_error = Array2F::default((layer_config.out_values, layer_config.in_values));
        let batches = inputs.shape()[0];

        zip(inputs.outer_iter(), grad.outer_iter())
            .map(|(i, g)| {
                let gt = g.insert_axis(Axis(1));
                let it = i.insert_axis(Axis(0));
                let result = gt.dot(&it);
                result
            })
            .for_each(|o| weights_error += &o);
        
        let biases_grad = grad.mean_axis(Axis(0)).unwrap().into_dyn();

        let factor = 1.0 / batches as f32;
        let weights_grad = (weights_error * factor).into_dyn();
        backward_cache.insert(key, vec![weights_grad, biases_grad]);

        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, layer_config.in_values));
        let weights_t = weights.t();
        grad.outer_iter()
            .map(|o| weights_t.dot(&o))
            .enumerate()
            .for_each(|(index, o)| result.slice_mut(s![index, ..]).assign(&o));

        Ok(result.into_dyn())
    }
}

impl TrainableLayerOps<DenseLayerConfig> for DenseLayer {
    fn train(data: TrainData, layer_config: &DenseLayerConfig) -> EmptyLayerResult {
        let TrainData {
            backward_cache,
            assigner,
            storage,
            batch_config,
            ..
        } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let [weights_grad, biases_grad] = remove_from_storage2(backward_cache, &key);

        let weights_grad = apply_lr_calc(
            &layer_config.weights_lr_calc,
            weights_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?;
        let biases_grad = apply_lr_calc(
            &layer_config.biases_lr_calc,
            biases_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?;

        get_mut_from_storage(storage, &key, 0).add_assign(&weights_grad);
        get_mut_from_storage(storage, &key, 1).add_assign(&biases_grad);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::controller::NNController;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::dense_layer::{DenseLayer, DenseLayerConfig, DenseLayerInit};
    use crate::nn::layers::nn_layers::{
        BackwardData, ForwardData, GenericStorage, InitData, Layer, LayerOps,
    };
    use crate::nn::loss::loss_func::LossFunc;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{arrays_almost_equal, Array1F, Array2F};
    use ndarray::array;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

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

        let output = DenseLayer::forward(
            ForwardData {
                batch_config: &BatchConfig { epoch: 1 },
                assigner: &mut KeyAssigner::new(),
                storage: &mut storage,
                inputs: input,
                forward_cache: &mut GenericStorage::new(),
            },
            &config,
        )
        .unwrap();

        assert!(arrays_almost_equal(&output, &expected));
    }

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
            [-1.335, -1.0085, -0.537, -1.3085, -0.89],
            [-0.249, -0.1875, -0.111, -0.2555, -0.166],
            [-1.2195, -0.9185, -0.54, -1.2475, -0.813]
        ];
        let expected_biases_grad: Array1F = array![-2.08, -0.4, -1.955];
        let config = DenseLayerConfig {
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
        let result = DenseLayer::backward(
            BackwardData {
                grad: grad.into_dyn(),
                batch_config: &BatchConfig { epoch: 1 },
                assigner: &mut KeyAssigner::new(),
                forward_cache: &mut forward_cache,
                storage: &mut storage,
                backward_cache: &mut backward_cache,
            },
            &config,
        )
        .unwrap();
        
        assert!(arrays_almost_equal(
            &expected,
            &result.into_dimensionality().unwrap()
        ));
        let cache = &backward_cache["dense_5_3_0"];
        
        assert!(arrays_almost_equal(&expected_weights_grad, &cache[0].clone().into_dimensionality().unwrap()));
        assert!(arrays_almost_equal(&expected_biases_grad, &cache[1].clone().into_dimensionality().unwrap()));
    }

    #[test]
    fn test_train() {
        let mut controller = NNController::new(
            Layer::Dense(DenseLayerConfig {
                in_values: 10,
                out_values: 10,
                init_mode: DenseLayerInit::Random(),
                weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            }),
            LossFunc::Mse,
        )
        .unwrap();
        let inputs = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let expected = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let mut last_loss = 0.0;
        let mut first_loss = None;

        for _ in 0..100 {
            let inputs = inputs.clone();
            last_loss = controller.train_batch(inputs, &expected).unwrap();
            if first_loss.is_none() {
                first_loss = Some(last_loss);
            }
            println!("{}", last_loss);
        }

        assert!(last_loss < first_loss.unwrap());
    }

    fn get_config(init_mode: DenseLayerInit) -> DenseLayerConfig {
        DenseLayerConfig {
            init_mode,
            in_values: 2,
            out_values: 3,
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        }
    }
}
