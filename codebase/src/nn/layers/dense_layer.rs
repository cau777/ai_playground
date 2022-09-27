use std::iter::zip;
use std::ops::AddAssign;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray::{Axis, s, ShapeBuilder};
use crate::utils::{Array1F, Array2F, ArrayDynF};
use crate::nn::layers::nn_layers::{BackwardData, ForwardData, InitData, LayerOps, TrainableLayerOps, TrainData};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};

pub struct DenseLayerConfig {
    pub out_values: usize,
    pub in_values: usize,
    pub init_mode: DenseLayerInit,
    pub weights_lr_calc: LrCalc,
    pub biases_lr_calc: LrCalc,
}

pub enum DenseLayerInit {
    WeightsAndBiases(Array2F, Array1F),
    Random(),
}

pub struct DenseLayer {}

fn gen_name(config: &DenseLayerConfig) -> String {
    format!("dense_{}_{}", config.in_values, config.out_values)
}

impl LayerOps<DenseLayerConfig> for DenseLayer {
    fn init(data: InitData, layer_config: &DenseLayerConfig) {
        let InitData { assigner, storage } = data;
        let key = assigner.get_key(gen_name(layer_config));
        let weights: Array2F;
        let biases: Array1F;

        match &layer_config.init_mode {
            DenseLayerInit::WeightsAndBiases(w, b) => {
                weights = w.clone();
                biases = b.clone();
            }
            DenseLayerInit::Random() => {
                let std_dev = (layer_config.out_values as f32).powf(-0.5);
                let dist = Normal::new(0.0, std_dev).unwrap();
                weights = Array2F::random((layer_config.out_values, layer_config.in_values).f(), dist);
                biases = Array1F::zeros((layer_config.out_values).f());
            }
        }

        storage.insert(key, vec![weights.into_dyn(), biases.into_dyn()]);
    }

    fn forward(data: ForwardData, layer_config: &DenseLayerConfig) -> ArrayDynF {
        let ForwardData { assigner, storage, inputs, forward_cache, .. } = data;
        let key = assigner.get_key(gen_name(layer_config));
        let data = storage.get(&key).unwrap();
        let weights: Array2F = data[0].clone().into_dimensionality().unwrap();
        let biases: &Array1F = &data[1].clone().into_dimensionality().unwrap();

        let inputs: Array2F = inputs.into_dimensionality().unwrap();

        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, layer_config.out_values).f());

        inputs.outer_iter()
            .map(|o| weights.dot(&o))
            .map(|o| o + biases)
            .enumerate()
            .for_each(|(index, o)| result.slice_mut(s![index, ..]).assign(&o));

        forward_cache.insert(key, vec![inputs.into_dyn()]);
        result.into_dyn()
    }

    fn backward(data: BackwardData, layer_config: &DenseLayerConfig) -> ArrayDynF {
        let BackwardData { assigner, storage, forward_cache, grad, backward_cache, .. } = data;
        let key = assigner.get_key(gen_name(layer_config));
        let data = storage.get(&key).unwrap();
        let cache = forward_cache.get(&key).unwrap();
        let weights: Array2F = data[0].clone().into_dimensionality().unwrap();

        let grad: Array2F = grad.into_dimensionality().unwrap();
        let inputs: Array2F = cache[0].clone().into_dimensionality().unwrap();
        let mut weights_error = Array2F::default((layer_config.out_values, layer_config.in_values));

        zip(inputs.outer_iter(), grad.outer_iter())
            .map(|(i, g)| {
                let gt = g.insert_axis(Axis(1));
                let it = i.insert_axis(Axis(0));
                gt.dot(&it)
            })
            .for_each(|o| weights_error += &o);

        let weights_grad = weights_error.mean_axis(Axis(0)).unwrap().into_dyn();
        let biases_grad = grad.mean_axis(Axis(0)).unwrap().into_dyn();
        backward_cache.insert(key, vec![weights_grad, biases_grad]);

        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, layer_config.in_values));
        let weights_t = weights.t();
        grad.outer_iter()
            // .map(|o| o.insert_axis(Axis(1)))
            .map(|o| weights_t.dot(&o))
            .enumerate()
            .for_each(|(index, o)| {
                // println!("{:?}", o.shape());
                result.slice_mut(s![index, ..]).assign(&o)
            });

        result.into_dyn()
    }
}

impl TrainableLayerOps<DenseLayerConfig> for DenseLayer {
    fn train(data: TrainData, layer_config: &DenseLayerConfig) {
        let TrainData { backward_cache, assigner, storage, batch_config, .. } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let cache = backward_cache.get(&key).unwrap();
        let weights_grad = cache[0].clone();
        let biases_grad = cache[1].clone();

        let weights_grad = apply_lr_calc(&layer_config.weights_lr_calc, weights_grad, LrCalcData { batch_config, storage, assigner });
        let biases_grad = apply_lr_calc(&layer_config.biases_lr_calc, biases_grad, LrCalcData { batch_config, storage, assigner });

        let mut data = storage.get_mut(&key).unwrap();
        let weights = data.get_mut(0).unwrap();
        weights.add_assign(&weights_grad);
        let biases = data.get_mut(1).unwrap();
        biases.add_assign(&biases_grad);

        // let mut data = storage.remove(&key).unwrap();
        // let weights = data.remove(0);
        // let biases = data.remove(0);
        //
        // storage.insert(key, vec![weights + weights_grad, biases + biases_grad]);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::nn::batch_config::{BatchConfig};
    use crate::nn::controller::NNController;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::dense_layer::{DenseLayer, DenseLayerConfig, DenseLayerInit};
    use crate::nn::layers::nn_layers::{BackwardData, ForwardData, GenericStorage, InitData, Layer, LayerOps};
    use crate::nn::layers::sequential_layer::SequentialLayerConfig;
    use crate::nn::loss::mse_loss::MseLossFunc;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{Array1F, Array2F, arrays_almost_equal};

    #[test]
    fn test_forward() {
        let input = array![
            [1.0, 2.0],
            [2.0, 3.0]
        ].into_dyn();
        let weights = array![
            [0.7, 0.0],
            [0.1, 0.4],
            [0.8, 0.6]
        ];
        let expected = array![
            [0.7, 0.9, 2.0],
            [1.4, 1.4, 3.4]
        ].into_dyn();

        let config = DenseLayerConfig {
            in_values: 2,
            out_values: 3,
            init_mode: DenseLayerInit::WeightsAndBiases(weights, Array1F::zeros(3)),
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        };

        let mut storage = GenericStorage::new();
        DenseLayer::init(InitData { storage: &mut storage, assigner: &mut KeyAssigner::new() }, &config);
        let output = DenseLayer::forward(ForwardData {
            batch_config: &BatchConfig { epoch: 1 },
            assigner: &mut KeyAssigner::new(),
            storage: &mut storage,
            inputs: input,
            forward_cache: &mut GenericStorage::new(),
        }, &config);

        assert!(arrays_almost_equal(&output, &expected));
    }

    #[test]
    fn test_backward() {
        let inputs: Array2F = array![[0.8, 0.7]];
        let grad = array![[0.1, 0.2, 0.3]].into_dyn();
        let weights = array![
            [0.7, 0.0],
            [0.1, 0.4],
            [0.8, 0.6]
        ];

        let config = DenseLayerConfig {
            in_values: 2,
            out_values: 3,
            init_mode: DenseLayerInit::WeightsAndBiases(weights, Array1F::zeros(3)),
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        };

        let mut storage = GenericStorage::new();
        DenseLayer::init(InitData { storage: &mut storage, assigner: &mut KeyAssigner::new() }, &config);

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("dense_2_3_0".to_owned(), vec![inputs.into_dyn()]);
        DenseLayer::backward(BackwardData {
            grad,
            batch_config: &BatchConfig { epoch: 1 },
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut forward_cache,
            storage: &mut storage,
            backward_cache: &mut GenericStorage::new(),
        }, &config);
    }

    #[test]
    fn test_train() {
        let mut controller = NNController::new(Layer::Dense(DenseLayerConfig {
            in_values: 10,
            out_values: 10,
            init_mode: DenseLayerInit::Random(),
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        }));
        let inputs = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let expected = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let loss = MseLossFunc {};
        let mut last_loss = 0.0;

        for n in 0..100 {
            let inputs = inputs.clone();
            last_loss = controller.train_batch(inputs, &expected, &loss);
            println!("{}", last_loss);
        }

        assert!(last_loss < 0.3);
    }

    #[test]
    fn test_train_multiple() {
        let mut controller = NNController::new(Layer::Sequential(SequentialLayerConfig {
            layers: vec![
                Layer::Dense(DenseLayerConfig {
                    in_values: 10,
                    out_values: 15,
                    init_mode: DenseLayerInit::Random(),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                }),
                Layer::Relu,
                Layer::Dense(DenseLayerConfig {
                    in_values: 15,
                    out_values: 15,
                    init_mode: DenseLayerInit::Random(),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                }),
                Layer::Relu,
                Layer::Dense(DenseLayerConfig {
                    in_values: 15,
                    out_values: 3,
                    init_mode: DenseLayerInit::Random(),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                }),
            ]
        }));

        let inputs = Array2F::random((2, 10), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let expected = Array2F::random((2, 3), Normal::new(0.0, 0.5).unwrap()).into_dyn();

        let loss = MseLossFunc {};
        let mut last_loss = 0.0;

        for n in 0..100 {
            let inputs = inputs.clone();
            last_loss = controller.train_batch(inputs, &expected, &loss);
            println!("{}", last_loss);
        }

        assert!(last_loss < 0.05);
    }
}