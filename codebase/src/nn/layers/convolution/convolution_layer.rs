use crate::nn::generic_storage::{clone_from_storage1, get_mut_from_storage, remove_from_storage1};
use crate::nn::layers::nn_layers::{
    BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult, TrainData,
    TrainableLayerOps,
};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};
use crate::utils::*;
use ndarray::{ErrorKind, ShapeError};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::ops::AddAssign;
use crate::nn::layers::convolution::convolution_cpu::{calc_forward, calc_inputs_grad, calc_kernel_grad, pad4d};
use crate::nn::layers::convolution::convolution_gpu::{calc_forward_gpu, calc_inputs_grad_gpu};

#[derive(Clone, Debug)]
pub struct ConvolutionConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub init_mode: ConvolutionInitMode,
    pub lr_calc: LrCalc,
}

#[derive(Clone, Debug)]
pub enum ConvolutionInitMode {
    Kernel(Array4F),
    HeNormal(),
}

pub struct ConvolutionLayer {}

fn gen_name(config: &ConvolutionConfig) -> String {
    format!("convolution_{}_{}", config.in_channels, config.out_channels)
}

impl LayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let InitData { assigner, storage } = data;
        let ConvolutionConfig { in_channels, out_channels, kernel_size, init_mode, .. } = layer_config.clone();
        let key = assigner.get_key(gen_name(layer_config));

        if let std::collections::hash_map::Entry::Vacant(e) = storage.entry(key) {
            let kernel = match init_mode {
                ConvolutionInitMode::Kernel(k) => {
                    let shape = k.shape();
                    if shape[0] != out_channels
                        || shape[1] != in_channels
                        || shape[2] != kernel_size
                        || shape[3] != kernel_size
                    {
                        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
                    }
                    k
                }
                ConvolutionInitMode::HeNormal() => {
                    let fan_in = in_channels * kernel_size * kernel_size;
                    let std_dev = (2.0 / fan_in as f32).sqrt();
                    let dist = Normal::new(0.0, std_dev)?;
                    Array4F::random((out_channels, in_channels, kernel_size, kernel_size), dist)
                }
            };

            e.insert(vec![kernel.into_dyn()]);
        }

        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        let ForwardData { inputs, storage, assigner, forward_cache, .. } = data;

        let inputs: Array4F = inputs.into_dimensionality()?;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel] = clone_from_storage1(storage, &key);
        let kernel: Array4F = kernel.into_dimensionality()?;

        let inputs = pad4d(inputs, layer_config.padding);
        let result = match data.gpu {
            Some(gpu) => match calc_forward_gpu(&inputs, &kernel, gpu, layer_config) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{:?}", e);
                    calc_forward(&inputs, &kernel, layer_config)?
                }
            }
            None => calc_forward(&inputs, &kernel, layer_config)?
        };

        forward_cache.insert(key, vec![inputs.into_dyn()]);
        Ok(result.into_dyn())
    }

    fn backward(data: BackwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        let BackwardData {
            assigner, forward_cache, storage,
            grad, backward_cache, ..
        } = data;

        let key = assigner.get_key(gen_name(layer_config));

        let [kernel] = clone_from_storage1(storage, &key);
        let kernel: Array4F = kernel.into_dimensionality()?;

        let [inputs] = remove_from_storage1(forward_cache, &key);
        let inputs = inputs.into_dimensionality()?;

        let grad = grad.into_dimensionality()?;

        let kernels_grad = calc_kernel_grad(&inputs, &grad, layer_config);
        backward_cache.insert(key, vec![kernels_grad.into_dyn()]);
        let inputs_grad = match data.gpu {
            Some(gpu) => match calc_inputs_grad_gpu(&inputs, &grad, &kernel, gpu, layer_config) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{:?}", e);
                    calc_inputs_grad(inputs, grad, kernel, layer_config)
                }
            }
            None => calc_inputs_grad(inputs, grad, kernel, layer_config)
        };

        Ok(inputs_grad.into_dyn())
    }
}

impl TrainableLayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn train(data: TrainData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let TrainData {
            storage,
            backward_cache,
            assigner,
            batch_config,
        } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel_grad] = remove_from_storage1(backward_cache, &key);
        let kernel_grad = apply_lr_calc(
            &layer_config.lr_calc,
            kernel_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?;

        let kernel = get_mut_from_storage(storage, &key, 0);
        kernel.add_assign(&kernel_grad);
        Ok(())
    }
}

// TODO: gpu tests
#[cfg(test)]
mod tests {
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::convolution::convolution_layer::ConvolutionInitMode::{HeNormal, Kernel};
    use crate::nn::layers::convolution::convolution_layer::{ConvolutionConfig, ConvolutionLayer};
    use crate::nn::layers::nn_layers::{
        BackwardData, ForwardData, GenericStorage, InitData, LayerOps, init_layer, Layer,
    };
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{arrays_almost_equal, ArrayDynF};
    use crate::Array4F;
    use ndarray::{array, stack, Axis};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::nn::layers::convolution::convolution_cpu::{calc_kernel_grad};
    use crate::nn::layers::convolution::convolution_layer;
    use crate::nn::train_config::TrainConfig;

    #[test]
    #[ignore]
    fn test_bench() {
        let config = ConvolutionConfig {
            init_mode: HeNormal(),
            stride: 1,
            kernel_size: 5,
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            in_channels: 32,
            out_channels: 64,
            padding: 2,
        };
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut storage = GenericStorage::new();
        convolution_layer::ConvolutionLayer::init(InitData {
            storage: &mut storage,
            assigner: &mut KeyAssigner::new(),
        }, &config).unwrap();
        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("convolution_32_64_0".to_owned(), vec![Array4F::random((64, 32, 18, 18), &dist).into_dyn()]);

        ConvolutionLayer::backward(BackwardData {
            grad: Array4F::random((64, 64, 14, 14), &dist).into_dyn(),
            storage: &storage,
            batch_config: &BatchConfig::new_not_train(),
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut forward_cache,
            backward_cache: &mut GenericStorage::new(),
            gpu: None,
        }, &config).unwrap();
    }

    #[test]
    fn test_forward() {
        let inputs = get_inputs();
        let expected = get_forward_result();
        let mut storage = GenericStorage::new();
        let config = get_config();

        ConvolutionLayer::init(
            InitData {
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
            },
            &config,
        )
            .unwrap();

        let result = ConvolutionLayer::forward(
            ForwardData {
                inputs,
                forward_cache: &mut GenericStorage::new(),
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                batch_config: &BatchConfig::new_train(),
                gpu: None,
            },
            &config,
        )
            .unwrap();

        assert!(arrays_almost_equal(&expected, &result));
    }

    #[test]
    fn test_backward() {
        let inputs: ArrayDynF = get_grad();
        let cache = get_forward_cache();
        let expected = get_backward_result();

        let mut storage = GenericStorage::new();
        let config = get_config();

        ConvolutionLayer::init(
            InitData {
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
            },
            &config,
        )
            .unwrap();

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("convolution_2_3_0".to_owned(), vec![cache]);

        let result = ConvolutionLayer::backward(
            BackwardData {
                grad: inputs,
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
                forward_cache: &mut forward_cache,
                backward_cache: &mut GenericStorage::new(),
                batch_config: &BatchConfig::new_train(),
                gpu: None,
            },
            &config,
        )
            .unwrap();

        println!("{:?}\r\n--------\r\n{:?}", result, expected);
        assert!(arrays_almost_equal(&result, &expected));
    }

    #[test]
    fn test_calc_kernel_grad() {
        let inputs = get_forward_cache().into_dimensionality().unwrap();
        let grad = get_grad().into_dimensionality().unwrap();
        let config = get_config();
        let expected = get_kernels_grad();
        let result = calc_kernel_grad(&inputs, &grad, &config);

        // println!("{:?}\r\n--------\r\n{:?}", result, expected);
        assert!(arrays_almost_equal(&expected, &result.into_dyn()));
    }

    #[test]
    fn test_init() {
        let config = ConvolutionConfig {
            in_channels: 1,
            out_channels: 4,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        };

        let mut assigner = KeyAssigner::new();
        let mut storage = GenericStorage::new();
        let init = InitData {
            assigner: &mut assigner,
            storage: &mut storage,
        };
        init_layer(&Layer::Convolution(config), init).unwrap();
        println!("{:?}", storage);
    }

    fn get_inputs() -> ArrayDynF {
        let inputs = array![
            [
                [0.22537, 0.51686, 0.5185, 0.60037, 0.53262],
                [0.01331, 0.5241, 0.89588, 0.7699, 0.12285],
                [0.29587, 0.61202, 0.72614, 0.4635, 0.76911],
                [0.19163, 0.55787, 0.55078, 0.47223, 0.79188],
                [0.11525, 0.6813, 0.36233, 0.34421, 0.44952]
            ],
            [
                [0.02694, 0.41525, 0.92223, 0.09121, 0.31512],
                [0.52802, 0.32806, 0.44892, 0.01633, 0.09703],
                [0.69259, 0.83594, 0.42432, 0.84877, 0.54679],
                [0.3541, 0.72725, 0.09385, 0.89286, 0.33626],
                [0.89183, 0.29685, 0.30165, 0.80624, 0.83761]
            ]
        ];
        stack![Axis(0), inputs].into_dyn()
    }

    fn get_kernels() -> Array4F {
        stack![
            Axis(0),
            array![
                [[-0.2341, -0.41141], [-0.03269, -0.35668]],
                [[0.45318, 0.38312], [0.41303, -0.66184]]
            ],
            array![
                [[-0.87622, 0.50122], [0.2724, 0.94758]],
                [[-0.38468, -0.70155], [-0.31623, -0.27944]]
            ],
            array![
                [[-0.61662, -0.21975], [0.45739, 0.13252]],
                [[-0.69169, 0.34276], [0.22805, -0.23069]]
            ]
        ]
    }

    fn get_forward_result() -> ArrayDynF {
        let result = array![
            [
                [-0.09822, -0.6407, -0.38049],
                [-0.3671, -0.38519, -0.48701],
                [-0.57453, -0.22021, -0.29585]
            ],
            [
                [0.20603, 0.24309, 0.55135],
                [-0.27693, 0.02055, -0.25353],
                [-0.29237, -0.20759, -0.56553]
            ],
            [
                [0.02365, 0.18707, 0.2933],
                [0.0575, -0.12417, -0.09843],
                [-0.1112, -0.57813, -0.75988]
            ]
        ];
        stack![Axis(0), result].into_dyn()
    }

    fn get_forward_cache() -> ArrayDynF {
        let cache = array![
            [
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0.22537, 0.51686, 0.5185, 0.60037, 0.53262, 0.],
                [0., 0.01331, 0.5241, 0.89588, 0.7699, 0.12285, 0.],
                [0., 0.29587, 0.61202, 0.72614, 0.4635, 0.76911, 0.],
                [0., 0.19163, 0.55787, 0.55078, 0.47223, 0.79188, 0.],
                [0., 0.11525, 0.6813, 0.36233, 0.34421, 0.44952, 0.],
                [0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0.02694, 0.41525, 0.92223, 0.09121, 0.31512, 0.],
                [0., 0.52802, 0.32806, 0.44892, 0.01633, 0.09703, 0.],
                [0., 0.69259, 0.83594, 0.42432, 0.84877, 0.54679, 0.],
                [0., 0.3541, 0.72725, 0.09385, 0.89286, 0.33626, 0.],
                [0., 0.89183, 0.29685, 0.30165, 0.80624, 0.83761, 0.],
                [0., 0., 0., 0., 0., 0., 0.]
            ]
        ];
        stack![Axis(0), cache].into_dyn()
    }

    fn get_backward_result() -> ArrayDynF {
        let result = array![
            [
                [-0.4668, -0.34546, -0.96733, -0.59356, -1.39406],
                [0.00082, -0.29748, -0.39212, -0.79371, -0.18983],
                [0.24772, 0.07721, -0.28081, 0.19632, 0.15916],
                [-0.22853, -1.17987, -0.2272, -2.06669, -0.01049],
                [0.17371, 0.62756, 0.38954, 0.98389, 1.06211]
            ],
            [
                [-0.00396, 0.59767, -0.62592, 0.52924, -0.0602],
                [-0.1467, 0.19314, 0.40909, 0.11018, 0.0849],
                [-0.61417, 0.38781, -0.55568, 0.28683, -0.83175],
                [0.10623, -0.75989, 0.27379, -1.21815, -0.0459],
                [-0.97521, 0.31429, -0.67425, 0.23328, -1.05826]
            ]
        ];
        stack![Axis(0), result].into_dyn()
    }

    fn get_grad() -> ArrayDynF {
        get_forward_result() * -2.0
    }

    fn get_kernels_grad() -> ArrayDynF {
        let grad: Array4F = stack![
            Axis(0),
            array![
                [[0.046632495, 0.04863675],
                 [0.07071798, 0.08882607]],
                [[0.03103437, 0.04097781],
                 [0.07444477, 0.12355672]]
            ],
            array![
                [[0.03151616, 0.035256498],
                 [-0.0008683999, 0.008833285]],
                [[0.03629486, 0.026376387],
                 [0.03135302, 0.039738197]]
            ],
            array![
                [[0.045678787, 0.059112985],
                 [0.028015418, 0.025232412]],
                [[0.063403, 0.021337919],
                 [0.048176333, 0.03950825]]
            ]
        ];
        grad.into_dyn()
    }

    fn get_config() -> ConvolutionConfig {
        ConvolutionConfig {
            kernel_size: 2,
            stride: 2,
            padding: 1,
            init_mode: Kernel(get_kernels()),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            in_channels: 2,
            out_channels: 3,
        }
    }
}
