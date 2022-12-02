use std::error::Error;
use crate::nn::generic_storage::{clone_from_storage1, get_mut_from_storage, remove_from_storage1};
use crate::nn::layers::nn_layers::{
    BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult, TrainData,
    TrainableLayerOps,
};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalc, LrCalcData};
use crate::utils::*;
use ndarray::{s, Axis, ErrorKind, ShapeBuilder, ShapeError, stack};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::ops::AddAssign;
use std::sync::Arc;
use ndarray::parallel::prelude::*;
use crate::gpu::shader_runner::{GlobalGpu, GpuData, ShaderRunner};
use crate::gpu::shaders;
use crate::utils::ShapeAsArray;

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

pub fn pad4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2];
    let width = shape[3];
    let mut result = Array4F::zeros(
        (
            shape[0],
            shape[1],
            height + 2 * padding,
            width + 2 * padding,
        ).f(),
    );
    let mut slice = result.slice_mut(s![
        ..,
        ..,
        padding..height + padding,
        padding..width + padding
    ]);
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

// TODO: move to separate file
impl ConvolutionLayer {
    fn calc_kernel_grad(inputs: &Array4F, grad: &Array4F, layer_config: &ConvolutionConfig) -> Array4F {
        let ConvolutionConfig { in_channels, out_channels, kernel_size, stride, .. } = layer_config;
        let kernel_size = *kernel_size;
        let stride = *stride;

        let shape = inputs.shape();
        let height = shape[2];
        let width = shape[3];

        let mean_inputs = inputs.mean_axis(Axis(0)).unwrap();
        let mean_grad = grad.mean_axis(Axis(0)).unwrap();
        let mean_grad = mean_grad.insert_axis(Axis(1));

        let mut parts = Vec::with_capacity(kernel_size * kernel_size);
        (0..kernel_size * kernel_size)
            .into_par_iter()
            .with_min_len(1)
            .map(|o| (o / kernel_size, o % kernel_size))
            .map(|(h, w)| {
                let affected = mean_inputs.slice(s![
                    ..,
                    h..height - (kernel_size - h - 1); stride,
                    w..width - (kernel_size - w - 1); stride]);
                let mul: Array4F = &mean_grad * &affected;
                Array2F::from_shape_fn((*out_channels, *in_channels), |(out_c, in_c)| {
                    mul.index_axis(Axis(0), out_c)
                        .index_axis(Axis(0), in_c)
                        .mean().unwrap()
                })
            })
            .collect_into_vec(&mut parts);

        let mut views = Vec::with_capacity(inputs.batch_size());
        views.extend(parts.iter().map(|o| o.view()));

        let joined = stack(Axis(2), &views).unwrap();
        let mut reshaped = joined.into_shape((*out_channels, *in_channels, kernel_size, kernel_size)).unwrap();
        reshaped.swap_axes(2, 3);

        reshaped / (layer_config.kernel_size.pow(2) as f32)
    }

    fn calc_inputs_grad(inputs: Array4F, grad: Array4F, kernel: Array4F, layer_config: &ConvolutionConfig) -> Array4F {
        let inputs_shape = inputs.shape();
        let ConvolutionConfig { kernel_size, stride, padding, .. } = layer_config;

        // Put height and width in front
        let grad = grad.permuted_axes((2, 3, 0, 1));
        let kernel = kernel.permuted_axes((1, 2, 3, 0));
        let kernel = kernel.insert_axis(Axis(3));

        let [batch_size, in_channels, new_height, new_width] =
            get_dims_after_filter_4(&inputs, *kernel_size, *stride);

        let mut parts = Vec::with_capacity(new_height * new_width);
        (0..(new_height * new_width))
            .into_par_iter()
            .map(|o| (o % new_width, o / new_width))
            .map(|(h, w)| {
                let current_grad = grad.slice(s![h, w, .., ..]);
                let batch_mul: Array5F = &kernel * &current_grad;
                let batch_sum = batch_mul.sum_axis(Axis(4));
                batch_sum.permuted_axes((3, 0, 1, 2))
            })
            .collect_into_vec(&mut parts);

        let mut padded_result =
            Array4F::zeros((batch_size, in_channels, inputs_shape[2], inputs_shape[3]));
        parts.into_iter()
            .enumerate()
            .for_each(|(i, arr)| {
                let h = i % new_width;
                let w = i / new_width;
                let h_offset = h * stride;
                let w_offset = w * stride;
                padded_result.slice_mut(s![
                    ..,
                    ..,
                    h_offset..(h_offset + kernel_size),
                    w_offset..(w_offset + kernel_size)
                ]).add_assign(&arr);
            });

        remove_padding_4d(padded_result, *padding)
    }

    fn calc_inputs_grad_2(inputs: Array4F, grad: Array4F, kernel: Array4F, layer_config: &ConvolutionConfig) -> Array4F {
        let grad_max_h = grad.shape()[2];
        let grad_max_w = grad.shape()[3];

        let ConvolutionConfig { out_channels, kernel_size, padding, stride, .. } = layer_config;
        let kernel_size = *kernel_size;

        let ish = inputs.shape();
        Array4F::from_shape_fn((ish[0], ish[1], ish[2] - 2 * padding, ish[3] - 2 * padding), |(b, in_c, h, w)| {
            let padded_h = h + padding;
            let padded_w = w + padding;
            let mut result = 0.0;

            let max_h = kernel_size.min(padded_h + 1);// Asserts the condition grad_h >= 0
            let min_h = (padded_h / stride + 1).max(grad_max_h) - grad_max_h;// Asserts the condition grad_h < grad_max_h

            // Rest of the division by the stride to get the position relative to the filter
            let mut kernel_h = (padded_h % stride).max(min_h);
            while kernel_h < max_h {
                let grad_h = (padded_h - kernel_h) / stride;

                let max_w = kernel_size.min(padded_w + 1); // Asserts the condition grad_w >= 0
                let min_w = (padded_w / stride + 1).max(grad_max_w) - grad_max_w; // Asserts the condition grad_w < grad_max_w

                let mut kernel_w = (padded_w % stride).max(min_w);
                while kernel_w < max_w {
                    let grad_w = (padded_w - kernel_w) / stride;

                    for out_c in 0..*out_channels {
                        result += grad[(b, out_c, grad_h, grad_w)] * kernel[(out_c, in_c, kernel_h, kernel_w)]
                    }

                    kernel_w += stride;
                }
                kernel_h += stride;
            }
            result
        })
    }

    fn calc_inputs_grad_gpu(inputs: &Array4F, grad: &Array4F, kernel: &Array4F, gpu: GlobalGpu, layer_config: &ConvolutionConfig) -> Result<Array4F, Box<dyn Error>> {
        let ish = inputs.shape_arr();
        let out_shape = (ish[0], ish[1], ish[2] - 2 * layer_config.padding, ish[3] - 2 * layer_config.padding);
        let ish = ish.map(|o| o as u32);
        let gsh = grad.shape_arr().map(|o| o as u32);

        let mut runner = ShaderRunner::new(gpu, |d| shaders::convolution_inputs_grad::load(d),
                                           "main", &shaders::convolution_inputs_grad::SpecializationConstants {
                batch_size: ish[0],
                grad_height: gsh[2],
                grad_width: gsh[3],
                input_height: ish[2],
                input_width: ish[3],
                in_channels: layer_config.in_channels as u32,
                out_channels: layer_config.out_channels as u32,
                kernel_size: layer_config.kernel_size as u32,
                stride: layer_config.stride as u32,
                padding: layer_config.padding as u32,
                out_height: out_shape.2 as u32,
                out_width: out_shape.3 as u32,
            })?;
        let results_buffer = runner.create_buffer_from_array(&Array4F::zeros(out_shape))?; // TODO: init
        let kernel_buffer = runner.create_buffer_from_array(kernel)?;
        let grad_buffer = runner.create_buffer_from_array(grad)?;
        println!("{:?}", out_shape);
        runner.execute([out_shape.0 * out_shape.1, out_shape.2, out_shape.3].map(|o| o as u32), shaders::convolution_inputs_grad::BLOCK_SIZE)?;
        let vec = results_buffer.read()?.to_vec();
        Ok(Array4F::from_shape_vec(out_shape, vec)?)
    }
}

impl LayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let InitData { assigner, storage } = data;
        let ConvolutionConfig {
            in_channels,
            out_channels,
            kernel_size,
            init_mode,
            ..
        } = layer_config.clone();
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
        let ForwardData {
            inputs,
            storage,
            assigner,
            forward_cache,
            ..
        } = data;
        let ConvolutionConfig {
            padding,
            stride,
            kernel_size,
            ..
        } = layer_config;

        let inputs: Array4F = inputs.into_dimensionality()?;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel] = clone_from_storage1(storage, &key);
        let kernel: Array4F = kernel.into_dimensionality()?;

        let inputs = pad4d(inputs, *padding);
        let [_, _, new_height, new_width] =
            get_dims_after_filter_4(&inputs, *kernel_size, *stride);

        let mut batches = Vec::with_capacity(inputs.batch_size());
        inputs.outer_iter().into_par_iter()
            .map(|batch| {
                let mut result = Array3F::zeros((layer_config.out_channels, new_height, new_width));
                for h in 0..new_height {
                    for w in 0..new_width {
                        let h_offset = h * stride;
                        let w_offset = w * stride;
                        let area = batch.slice(s![
                        ..,
                        h_offset..(h_offset + kernel_size),
                        w_offset..(w_offset + kernel_size)
                    ]);
                        let area = area.insert_axis(Axis(0));
                        let out: Array4F = &area * &kernel;


                        out.outer_iter()
                            .map(|o| o.sum())
                            .enumerate()
                            .for_each(|(index, o)| result[(index, h, w)] = o);
                    }
                }
                result
            })
            .collect_into_vec(&mut batches);

        let mut views = Vec::with_capacity(inputs.batch_size());
        views.extend(batches.iter().map(|o| o.view()));
        let result = stack(Axis(0), &views)?;
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

        let kernels_grad = ConvolutionLayer::calc_kernel_grad(&inputs, &grad, layer_config);
        backward_cache.insert(key, vec![kernels_grad.into_dyn()]);
        let inputs_grad = match data.gpu {
            Some(gpu) => match ConvolutionLayer::calc_inputs_grad_gpu(&inputs, &grad, &kernel, gpu, layer_config) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{:?}", e);
                    ConvolutionLayer::calc_inputs_grad_2(inputs, grad, kernel, layer_config)
                }
            }
            None => ConvolutionLayer::calc_inputs_grad_2(inputs, grad, kernel, layer_config)
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
    use std::sync::Arc;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::convolution_layer::ConvolutionInitMode::{HeNormal, Kernel};
    use crate::nn::layers::convolution_layer::{ConvolutionConfig, ConvolutionLayer};
    use crate::nn::layers::nn_layers::{
        BackwardData, ForwardData, GenericStorage, InitData, LayerOps, init_layer, Layer,
    };
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use crate::utils::{arrays_almost_equal, ArrayDynF, as_array, get_dims_after_filter_4};
    use crate::Array4F;
    use ndarray::{array, stack, Axis};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::gpu::shader_runner::GpuData;
    use crate::nn::layers::convolution_layer;
    use crate::nn::train_config::TrainConfig;

    #[test]
    fn test_no_regression() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let config = ConvolutionConfig {
            in_channels: 3,
            out_channels: 4,
            kernel_size: 2,
            padding: 1,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            stride: 2,
        };
        let inputs = Array4F::random((8, config.in_channels, 10, 10), &dist);
        let grad_shape = (inputs.shape()[0], config.out_channels,
                          (inputs.shape()[2] - config.kernel_size) / config.stride + 1,
                          (inputs.shape()[3] - config.kernel_size) / config.stride + 1);

        let grad = Array4F::random(grad_shape, &dist);
        let kernel = Array4F::random((config.out_channels, config.in_channels, config.kernel_size, config.kernel_size), &dist);

        // let expected = ConvolutionLayer::calc_inputs_grad(inputs.clone(), grad.clone(), kernel.clone(), &config);
        let expected = ConvolutionLayer::calc_inputs_grad_2(inputs.clone(), grad.clone(), kernel.clone(), &config);
        let actual = ConvolutionLayer::calc_inputs_grad_gpu(&inputs, &grad, &kernel, GpuData::new_global().unwrap(),&config).unwrap();
        // println!("Expected\n {:?}\n-------\nActual\n {:?}", expected, actual);
        assert!(arrays_almost_equal(&expected, &actual));
    }

    #[test]
    fn test_bench() {
        let config = convolution_layer::ConvolutionConfig {
            init_mode: convolution_layer::ConvolutionInitMode::HeNormal(),
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

        convolution_layer::ConvolutionLayer::backward(BackwardData {
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
                batch_config: &BatchConfig::new_train(TrainConfig::default()),
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
                batch_config: &BatchConfig::new_train(TrainConfig::default()),
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
        let result = ConvolutionLayer::calc_kernel_grad(&inputs, &grad, &config);

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
            init_mode: super::ConvolutionInitMode::HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        };

        let mut assigner = KeyAssigner::new();
        let mut storage = GenericStorage::new();
        let init = InitData {
            assigner: &mut assigner,
            storage: &mut storage,
        };
        let result = init_layer(&Layer::Convolution(config), init);
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
