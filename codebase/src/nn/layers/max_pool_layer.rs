use ndarray::{s};
use crate::Array4F;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};
use crate::nn::utils::{pad4d, remove_padding_4d};
use crate::utils::{get_dims_after_filter_4};

#[derive(Clone, Debug)]
pub struct MaxPoolConfig {
    pub size: usize,
    pub stride: usize,
    pub padding: usize,
}

pub struct MaxPoolLayer;

fn gen_name() -> String {
    "max_pool".to_owned()
}

impl LayerOps<MaxPoolConfig> for MaxPoolLayer {
    fn init(_: InitData, _: &MaxPoolConfig) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, layer_config: &MaxPoolConfig) -> LayerResult {
        let ForwardData { inputs, forward_cache, assigner, .. } = data;
        let stride = layer_config.stride;
        let size = layer_config.size;
        let inputs: Array4F = inputs.into_dimensionality()?;
        let inputs = pad4d(inputs, layer_config.padding);

        let [batch_size, channels, new_height, new_width] =
            get_dims_after_filter_4(&inputs, size, stride);

        let result = Array4F::from_shape_fn((batch_size, channels, new_height, new_width), |(b, c, h, w)| {
            let h_offset = h * stride;
            let w_offset = w * stride;
            let area = inputs.slice(s![b, c, h_offset..(h_offset + size), w_offset..(w_offset + size)]);
            area.into_iter().copied().reduce(f32::max).unwrap_or(0.0)
        });

        let key = assigner.get_key(gen_name());
        forward_cache.insert(key, vec![inputs.into_dyn()]);
        Ok(result.into_dyn())
    }

    fn backward(data: BackwardData, layer_config: &MaxPoolConfig) -> LayerResult {
        let BackwardData { forward_cache, assigner, grad, .. } = data;
        let grad: Array4F = grad.into_dimensionality()?;

        let key = assigner.get_key(gen_name());
        let [inputs] = remove_from_storage1(forward_cache, &key);
        let inputs: Array4F = inputs.into_dimensionality()?;

        let size = layer_config.size;
        let stride = layer_config.stride;
        let [_, _, new_height, new_width] = get_dims_after_filter_4(&inputs, size, stride);

        let mut result: Array4F = &inputs * 0.0;

        inputs.outer_iter().enumerate().for_each(|(b, batch)| {
            batch.outer_iter().enumerate().for_each(|(c, channel)| {
                for h in 0..new_height {
                    for w in 0..new_width {
                        let h_offset = h * stride;
                        let w_offset = w * stride;
                        let area = channel.slice(s![h_offset..(h_offset + size), w_offset..(w_offset + size)]);
                        let argmax = area.into_iter()
                            .copied()
                            .enumerate()
                            .reduce(|accum, item| {
                                if item.1 > accum.1 { item } else { accum }
                            }).unwrap();
                        result[(b, c, h_offset + argmax.0 / size, w_offset + argmax.0 % size)] += grad[(b, c, h, w)];
                    }
                }
            })
        });

        let result = remove_padding_4d(result, layer_config.padding);
        Ok(result.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis, stack};
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::max_pool_layer::{MaxPoolConfig, MaxPoolLayer};
    use crate::nn::layers::nn_layers::{BackwardData, ForwardData, GenericStorage, LayerOps};
    use crate::utils::{Array3F, ArrayDynF};

    fn create_inputs() -> ArrayDynF {
        let arr: Array3F = array![
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [-8.0, 0.0, 0.0, 4.0]
            ],
            [
                [-1.0, 3.0, -5.0, 1.0],
                [2.0, 4.0, -99.0, 32.0],
                [16.0, 69.0, -69.0, 1.0],
                [-8.0, 0.0, 0.0, 4.0]
            ]
        ];
        stack![Axis(0), arr].into_dyn()
    }

    fn create_forward_outputs() -> ArrayDynF {
        let result: Array3F = array![
            [
                [ 6.,  8.],
                [10., 12.]
            ],
            [
                [ 4., 32.],
                [69.,  4.]
            ]
        ];
        stack![Axis(0), result].into_dyn()
    }

    #[test]
    fn test_forward_2x2() {
        let inputs = create_inputs();
        let expected = create_forward_outputs();

        fn forward(inputs: ArrayDynF, size: usize, stride: usize) -> ArrayDynF {
            MaxPoolLayer::forward(ForwardData {
                inputs,
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                storage: &mut GenericStorage::new(),
                forward_cache: &mut GenericStorage::new(),
                gpu: None,
            }, &MaxPoolConfig { size, stride, padding: 0 }).unwrap()
        }

        assert_eq!(expected.into_dyn(), forward(inputs, 2, 2));
    }

    #[test]
    fn test_backprop_2x2() {
        let inputs = create_inputs();
        let grad = create_forward_outputs() * -0.7;
        let expected: Array3F = array![
            [[  0. ,   0. ,   0. ,   0. ],
             [  0. ,  -4.2,   0. ,  -5.6],
             [  0. ,  -7. ,   0. ,  -8.4],
             [ -0. ,   0. ,   0. ,   0. ]],

            [[ -0. ,   0. ,  -0. ,   0. ],
             [  0. ,  -2.8,  -0. , -22.4],
             [  0. , -48.3,  -0. ,   0. ],
             [ -0. ,   0. ,   0. ,  -2.8]]
        ];
        let expected = stack![Axis(0), expected].into_dyn();

        fn backward(inputs: ArrayDynF, grad: ArrayDynF, size: usize, stride: usize) -> ArrayDynF {
            let mut forward_cache = GenericStorage::new();
            forward_cache.insert("max_pool_0".to_owned(), vec![inputs]);
            MaxPoolLayer::backward(BackwardData {
                grad,
                assigner: &mut KeyAssigner::new(),
                storage: &mut GenericStorage::new(),
                forward_cache: &mut forward_cache,
                backward_cache: &mut GenericStorage::new(),
                batch_config: &BatchConfig::new_train(),
                gpu: None,
            }, &MaxPoolConfig { size, stride, padding: 0 }).unwrap()
        }

        let result = backward(inputs, grad, 2, 2);
        assert_eq!(expected, result);
    }
}