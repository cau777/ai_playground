use std::ops::AddAssign;
use ndarray::{Axis, ErrorKind, s, ShapeBuilder, ShapeError};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use crate::nn::generic_storage::{clone_from_storage1, get_mut_from_storage, remove_from_storage1};
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult, TrainableLayerOps, TrainData};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};
use crate::utils::{Array4F, Array5F, get_dims_after_filter_4};

#[derive(Clone)]
pub struct ConvolutionConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    init_mode: ConvolutionInitMode,
    lr_calc: LrCalc,
}

#[derive(Clone)]
pub enum ConvolutionInitMode {
    Kernel(Array4F),
    HeNormal(),
}

pub fn pad4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2];
    let width = shape[3];
    let mut result = Array4F::zeros((shape[0], shape[1], height + 2 * padding, width + 2 * padding).f());
    let mut slice = result.slice_mut(s![.., .., padding..height+padding, padding..width+padding]);
    slice.assign(&array);
    result
}

pub fn remove_padding_4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2] - padding;
    let width = shape[3] - padding;
    array.slice_move(s![.., .., padding..height, padding..width])
}

pub struct ConvolutionLayer {}

fn gen_name(config: &ConvolutionConfig) -> String {
    format!("convolution_{}_{}", config.in_channels, config.out_channels)
}

impl ConvolutionLayer {
    fn calc_kernel_grad(inputs: &Array4F, grad: &Array4F, layer_config: &ConvolutionConfig) -> Array4F {
        let ConvolutionConfig { in_channels, out_channels, kernel_size, stride, .. } = layer_config;
        let kernel_size = *kernel_size;
        let stride = *stride;

        let shape = inputs.shape();
        let height = shape[2];
        let width = shape[3];

        let mut kernels_grad = Array4F::zeros((*out_channels, *in_channels, kernel_size, kernel_size));
        let mean_inputs = inputs.mean_axis(Axis(0)).unwrap();
        let mean_grad = grad.mean_axis(Axis(0)).unwrap();
        let mean_grad = mean_grad.insert_axis(Axis(1));

        for h in 0..kernel_size {
            for w in 0..kernel_size {
                let affected = mean_inputs.slice(s![
                    ..,
                    h..height - (kernel_size - h - 1); stride,
                    w..width - (kernel_size - w - 1); stride]
                );
                let mul: Array4F = &mean_grad * &affected;
                mul.outer_iter().enumerate().for_each(|(out_c, arr)| {
                    arr.outer_iter().enumerate().for_each(|(in_c, arr)| {
                        let mean = arr.mean().unwrap();
                        kernels_grad[(out_c, in_c, h, w)] = mean;
                    })
                })
            }
        }

        kernels_grad
    }
}

impl LayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let InitData { assigner, storage } = data;
        let ConvolutionConfig { in_channels, out_channels, kernel_size, init_mode, .. } = layer_config.clone();
        let key = assigner.get_key(gen_name(layer_config));

        if !storage.contains_key(&key) {
            let kernel = match init_mode {
                ConvolutionInitMode::Kernel(k) => {
                    let shape = k.shape();
                    if shape[0] != out_channels || shape[1] != in_channels || shape[2] != kernel_size || shape[3] != kernel_size {
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

            storage.insert(key, vec![kernel.into_dyn()]);
        }

        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        let ForwardData { inputs, storage, assigner, forward_cache, .. } = data;
        let ConvolutionConfig { padding, stride, kernel_size, .. } = layer_config;

        let inputs: Array4F = inputs.into_dimensionality()?;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel] = clone_from_storage1(storage, &key);
        let kernel: Array4F = kernel.into_dimensionality()?;

        let inputs = pad4d(inputs, *padding);
        let [batch_size, _, new_height, new_width] = get_dims_after_filter_4(&inputs, *kernel_size, *stride);
        let mut result = Array4F::zeros((batch_size, layer_config.out_channels, new_height, new_width));

        inputs.outer_iter().enumerate().for_each(|(b, batch)| {
            for h in 0..new_height {
                for w in 0..new_width {
                    let h_offset = h * stride;
                    let w_offset = w * stride;
                    let area = batch.slice(s![.., h_offset..(h_offset + kernel_size), w_offset..(w_offset + kernel_size)]);
                    let area = area.insert_axis(Axis(0));
                    let out: Array4F = &area * &kernel;

                    out.outer_iter()
                        .map(|o| o.sum())
                        .enumerate()
                        .for_each(|(index, o)| result[(b, index, h, w)] = o); // TODO: replace by this
                }
            }
        });

        forward_cache.insert(key, vec![inputs.into_dyn()]);
        Ok(result.into_dyn())
    }

    fn backward(data: BackwardData, layer_config: &ConvolutionConfig) -> LayerResult {
        let BackwardData { assigner, forward_cache, storage, grad, backward_cache, .. } = data;
        let ConvolutionConfig { padding, stride, kernel_size, .. } = layer_config;

        let key = assigner.get_key(gen_name(layer_config));

        let [kernel] = clone_from_storage1(storage, &key);
        let kernel: Array4F = kernel.into_dimensionality()?;

        let [inputs] = remove_from_storage1(forward_cache, &key);
        let inputs = inputs.into_dimensionality()?;
        let inputs_shape = inputs.shape();

        let grad = grad.into_dimensionality()?;

        let kernels_grad = ConvolutionLayer::calc_kernel_grad(&inputs, &grad, layer_config);
        backward_cache.insert(key, vec![kernels_grad.into_dyn()]);

        // Put height and width in front
        let grad = grad.permuted_axes((2, 3, 0, 1));
        let kernel = kernel.permuted_axes((1, 2, 3, 0));
        let kernel = kernel.insert_axis(Axis(3));

        let [batch_size, in_channels, new_height, new_width] = get_dims_after_filter_4(&inputs, *kernel_size, *stride);
        let mut padded_result = Array4F::zeros((batch_size, in_channels, inputs_shape[2], inputs_shape[3]));

        for h in 0..new_height {
            for w in 0..new_width {
                let h_offset = h * stride;
                let w_offset = w * stride;

                let current_grad = grad.slice(s![h, w, .., ..]);
                let batch_mul: Array5F = &kernel * &current_grad;
                let batch_sum = batch_mul.sum_axis(Axis(4));
                let batch_t = batch_sum.permuted_axes((3, 0, 1, 2));
                padded_result.slice_mut(s![.., .., h_offset..(h_offset + kernel_size), w_offset..(w_offset + kernel_size)])
                    .add_assign(&batch_t);
            }
        }

        let result = remove_padding_4d(padded_result, *padding);
        Ok(result.into_dyn())
    }
}

impl TrainableLayerOps<ConvolutionConfig> for ConvolutionLayer
{
    fn train(data: TrainData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let TrainData { storage, backward_cache, assigner, batch_config } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel_grad] = remove_from_storage1(backward_cache, &key);
        let kernel_grad = apply_lr_calc(&layer_config.lr_calc, kernel_grad, LrCalcData { batch_config, storage, assigner })?;

        let kernel = get_mut_from_storage(storage, &key, 0);
        kernel.add_assign(&kernel_grad);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis, stack};
    use crate::Array4F;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::convolution_layer::{ConvolutionConfig, ConvolutionLayer};
    use crate::nn::layers::convolution_layer::ConvolutionInitMode::Kernel;
    use crate::nn::layers::nn_layers::{BackwardData, ForwardData, GenericStorage, InitData, LayerOps};
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{ArrayDynF, arrays_almost_equal};

    #[test]
    fn test_forward()
    {
        let inputs = get_inputs();
        let expected = get_forward_result();
        let mut storage = GenericStorage::new();
        let config = get_config();

        ConvolutionLayer::init(InitData {
            storage: &mut storage,
            assigner: &mut KeyAssigner::new(),
        }, &config).unwrap();

        let result = ConvolutionLayer::forward(ForwardData {
            inputs,
            forward_cache: &mut GenericStorage::new(),
            storage: &mut storage,
            assigner: &mut KeyAssigner::new(),
            batch_config: &BatchConfig { epoch: 1 },
        }, &config).unwrap();

        assert!(arrays_almost_equal(&expected, &result));
    }

    #[test]
    fn test_backward() {
        let inputs: ArrayDynF = get_grad();
        let cache = get_forward_cache();
        let expected = get_backward_result();

        let mut storage = GenericStorage::new();
        let config = get_config();

        ConvolutionLayer::init(InitData {
            storage: &mut storage,
            assigner: &mut KeyAssigner::new(),
        }, &config).unwrap();

        let mut forward_cache = GenericStorage::new();
        forward_cache.insert("convolution_2_3_0".to_owned(), vec![cache]);

        let result = ConvolutionLayer::backward(BackwardData {
            grad: inputs,
            storage: &mut storage,
            assigner: &mut KeyAssigner::new(),
            forward_cache: &mut forward_cache,
            backward_cache: &mut GenericStorage::new(),
            batch_config: &BatchConfig { epoch: 1 },
        }, &config).unwrap();
        assert!(arrays_almost_equal(&result, &expected));
    }

    #[test]
    fn test_calc_kernel_grad() {
        let inputs = get_forward_cache().into_dimensionality().unwrap();
        let grad = get_grad().into_dimensionality().unwrap();
        let config = get_config();
        let expected = get_kernels_grad();
        let result = ConvolutionLayer::calc_kernel_grad(&inputs, &grad, &config);

        assert!(arrays_almost_equal(&expected, &result.into_dyn()));
    }

    fn get_inputs() -> ArrayDynF {
        let inputs = array![
            [[0.22537, 0.51686, 0.5185 , 0.60037, 0.53262],
             [0.01331, 0.5241 , 0.89588, 0.7699 , 0.12285],
             [0.29587, 0.61202, 0.72614, 0.4635 , 0.76911],
             [0.19163, 0.55787, 0.55078, 0.47223, 0.79188],
             [0.11525, 0.6813 , 0.36233, 0.34421, 0.44952]],
            [[0.02694, 0.41525, 0.92223, 0.09121, 0.31512],
             [0.52802, 0.32806, 0.44892, 0.01633, 0.09703],
             [0.69259, 0.83594, 0.42432, 0.84877, 0.54679],
             [0.3541 , 0.72725, 0.09385, 0.89286, 0.33626],
             [0.89183, 0.29685, 0.30165, 0.80624, 0.83761]]
        ];
        stack![Axis(0), inputs].into_dyn()
    }

    fn get_kernels() -> Array4F {
        stack![Axis(0),
            array![
                [[-0.2341 , -0.41141],
                 [-0.03269, -0.35668]],
                [[ 0.45318,  0.38312],
                 [ 0.41303, -0.66184]]
            ],
            array![
                [[-0.87622,  0.50122],
                 [ 0.2724 ,  0.94758]],
                [[-0.38468, -0.70155],
                 [-0.31623, -0.27944]]
            ],
            array![
                [[-0.61662, -0.21975],
                 [ 0.45739,  0.13252]],
                [[-0.69169,  0.34276],
                 [ 0.22805, -0.23069]]
            ]
        ]
    }

    fn get_forward_result() -> ArrayDynF {
        let result = array![
            [[-0.09822, -0.6407 , -0.38049],
             [-0.3671 , -0.38519, -0.48701],
             [-0.57453, -0.22021, -0.29585]],
            [[ 0.20603,  0.24309,  0.55135],
             [-0.27693,  0.02055, -0.25353],
             [-0.29237, -0.20759, -0.56553]],
            [[ 0.02365,  0.18707,  0.2933 ],
             [ 0.0575 , -0.12417, -0.09843],
             [-0.1112 , -0.57813, -0.75988]]
        ];
        stack![Axis(0), result].into_dyn()
    }

    fn get_forward_cache() -> ArrayDynF {
        let cache = array![
            [[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
             [0.     , 0.22537, 0.51686, 0.5185 , 0.60037, 0.53262, 0.     ],
             [0.     , 0.01331, 0.5241 , 0.89588, 0.7699 , 0.12285, 0.     ],
             [0.     , 0.29587, 0.61202, 0.72614, 0.4635 , 0.76911, 0.     ],
             [0.     , 0.19163, 0.55787, 0.55078, 0.47223, 0.79188, 0.     ],
             [0.     , 0.11525, 0.6813 , 0.36233, 0.34421, 0.44952, 0.     ],
             [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ]],
            [[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ],
             [0.     , 0.02694, 0.41525, 0.92223, 0.09121, 0.31512, 0.     ],
             [0.     , 0.52802, 0.32806, 0.44892, 0.01633, 0.09703, 0.     ],
             [0.     , 0.69259, 0.83594, 0.42432, 0.84877, 0.54679, 0.     ],
             [0.     , 0.3541 , 0.72725, 0.09385, 0.89286, 0.33626, 0.     ],
             [0.     , 0.89183, 0.29685, 0.30165, 0.80624, 0.83761, 0.     ],
             [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ]]
        ];
        stack![Axis(0), cache].into_dyn()
    }

    fn get_backward_result() -> ArrayDynF {
        let result = array![
            [[-0.4668 , -0.34546, -0.96733, -0.59356, -1.39406],
             [ 0.00082, -0.29748, -0.39212, -0.79371, -0.18983],
             [ 0.24772,  0.07721, -0.28081,  0.19632,  0.15916],
             [-0.22853, -1.17987, -0.2272 , -2.06669, -0.01049],
             [ 0.17371,  0.62756,  0.38954,  0.98389,  1.06211]],
            [[-0.00396,  0.59767, -0.62592,  0.52924, -0.0602 ],
             [-0.1467 ,  0.19314,  0.40909,  0.11018,  0.0849 ],
             [-0.61417,  0.38781, -0.55568,  0.28683, -0.83175],
             [ 0.10623, -0.75989,  0.27379, -1.21815, -0.0459 ],
             [-0.97521,  0.31429, -0.67425,  0.23328, -1.05826]]
        ];
        stack![Axis(0), result].into_dyn()
    }

    fn get_grad() -> ArrayDynF {
        get_forward_result() * -2.0
    }

    fn get_kernels_grad() -> ArrayDynF {
        let grad: Array4F = stack![Axis(0),
            array![
                [[ 0.18653,  0.19455],
                 [ 0.28287,  0.3553 ]],
                [[ 0.12414,  0.16391],
                 [ 0.29778,  0.49423]]
            ],
            array![
                [[ 0.12606,  0.14103],
                 [-0.00347,  0.03533]],
                [[ 0.14518,  0.10551],
                 [ 0.12541,  0.15895]]
            ],
            array![
                [[ 0.18271,  0.23645],
                 [ 0.11206,  0.10093]],
                [[ 0.25361,  0.08535],
                 [ 0.19271,  0.15803]]
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