use ndarray::s;
use crate::Array4F;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::filtering::max_pool::{gen_name, MaxPoolConfig};
use crate::nn::layers::filtering::{pad4d, remove_padding_4d};
use crate::nn::layers::nn_layers::*;
use crate::utils::get_dims_after_filter_4;

pub fn backward(data: BackwardData, layer_config: &MaxPoolConfig) -> LayerResult {
    let BackwardData { forward_cache, assigner, grad, .. } = data;
    let grad: Array4F = grad.into_dimensionality()?;

    let key = assigner.get_key(gen_name());
    let [inputs] = remove_from_storage1(forward_cache, &key);
    let inputs: Array4F = inputs.into_dimensionality()?;
    let inputs = pad4d(inputs, layer_config.padding);

    let size = layer_config.size;
    let stride = layer_config.stride;
    let [_, _, new_height, new_width] = get_dims_after_filter_4(inputs.shape(), size, stride);

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
    Ok(result.into_dyn().into())
}

#[cfg(test)]
mod tests {
    use ndarray::{array, stack, Axis};
    use crate::ArrayDynF;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::filtering::max_pool::tests::{create_forward_outputs, create_inputs};
    use crate::utils::Array3F;
    use super::*;

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
        let expected = stack![Axis(0), expected, expected, expected, expected].into_dyn();

        fn action(inputs: ArrayDynF, grad: ArrayDynF, size: usize, stride: usize) -> ArrayDynF {
            let mut forward_cache = GenericStorage::new();
            forward_cache.insert("max_pool_0".to_owned(), vec![inputs]);
            backward(BackwardData {
                grad,
                assigner: &mut KeyAssigner::new(),
                storage: &mut GenericStorage::new(),
                forward_cache: &mut forward_cache,
                backward_cache: &mut GenericStorage::new(),
                batch_config: &BatchConfig::new_train(),
                gpu: None,
            }, &MaxPoolConfig { size, stride, padding: 0 }).unwrap().into_memory().unwrap()
        }

        let result = action(inputs, grad, 2, 2);
        assert_eq!(expected, result);
    }
}