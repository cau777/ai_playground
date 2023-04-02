mod max_pool_forward;
mod max_pool_backward;

use crate::nn::layers::nn_layers::*;

#[derive(Clone, Debug)]
pub struct MaxPoolConfig {
    pub size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Apply MAX operation with 2D filters, That means passing a filter through the last
/// 2 dimension of the input (usually height and width). In position of the filter, the maximum
/// value of those input values is computed. Requires a 4 dimensional input (one being the batch).
/// Use for reducing the size of arrays after **Convolution**.
/// https://deepai.org/machine-learning-glossary-and-terms/max-pooling
pub struct MaxPoolLayer;

fn gen_name() -> String {
    "max_pool".to_owned()
}

impl LayerOps<MaxPoolConfig> for MaxPoolLayer {
    fn init(_: InitData, _: &MaxPoolConfig) -> EmptyLayerResult { Ok(()) }

    #[inline(never)]
    fn forward(data: ForwardData, layer_config: &MaxPoolConfig) -> LayerResult {
        max_pool_forward::forward(data, layer_config)
    }

    fn backward(data: BackwardData, layer_config: &MaxPoolConfig) -> LayerResult {
       max_pool_backward::backward(data, layer_config)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis, stack};
    use crate::utils::{Array3F, ArrayDynF};

    pub fn create_inputs() -> ArrayDynF {
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
            ],
        ];
        stack![Axis(0), arr, arr, arr, arr].into_dyn()
    }

    pub fn create_forward_outputs() -> ArrayDynF {
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
        stack![Axis(0), result, result, result, result].into_dyn()
    }
}
