use std::any::Any;
use std::collections::HashMap;
use ndarray::prelude::*;
use wasm_bindgen::prelude::*;

type Array4F = Array4<f32>;

trait Cache {

}

#[wasm_bindgen]
pub fn test() -> String {
    let arr = Array4F::zeros((5, 5, 5, 5).f());
    arr.to_string()
}

// fn pad4d(array: Array4<f32>, padding: usize) -> Array4<f32> {
//     let shape = array.shape();
//     let height = shape[2];
//     let width = shape[3];
//     let mut result = Array4::<f32>::zeros((shape[0], shape[1], height + 2 * padding, width + 2 * padding).f());
//     let mut slice = result.slice_mut(s![.., .., .., ..]);
//     slice.assign(&array);
//     result
// }
//
// fn a(){
//     let tp = 5;
//     let mut gen = HashMap::<u8, Box<dyn Any>>::new();
//     gen.insert(0, Box::new(0));
//     let ob = gen.get(&0).unwrap();
//     ob.downcast
// }

/*
def pad4d(array: np.ndarray, padding: int):
    shape = array.shape
    height = shape[-2]
    width = shape[-1]
    result = np.zeros((*shape[:-2], height + 2 * padding, width + 2 * padding), dtype="float32")
    result[:, :, padding:height + padding, padding:width + padding] = array
    return result


def remove_padding4d(array: np.ndarray, padding: int):
    shape = array.shape
    return array[:, :, padding:shape[-2] - padding, padding:shape[-1] - padding]


def extract_fragments4d(array: np.ndarray, size: int, stride: int):
    batch, channels, new_height, new_width = get_dims_after_filter(array.shape, size, stride)
    result = np.zeros((new_height, new_width, batch, channels, size, size), dtype="float32")
    for h in range(new_height):
        for w in range(new_width):
            h_offset = h * stride
            w_offset = w * stride
            result[h, w] = array[:, :, h_offset:h_offset + size, w_offset:w_offset + size]
    return result


class ConvolutionLayer(NNLayer):
    def __init__(self, kernels: np.ndarray, optimizer: LrOptimizer, stride: int = 1, padding: int = 0):
        self.kernels = kernels
        self.kernels_grad = kernels * 0
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding
        self.out_channels = kernels.shape[0]
        self.in_channels = kernels.shape[1]
        self.kernel_size = kernels.shape[2]

    @staticmethod
    def create_random(in_channels: int, out_channels: int, kernel_size: int, optimizer: LrOptimizer,
                      stride: int = 1, padding: int = 0):
        if stride < 1:
            raise ValueError("Stride can't be less than 1")
        if padding < 0:
            raise ValueError("Padding can't be negative")

        # 'He normal' initialization
        fan_in = in_channels * kernel_size * kernel_size
        std_dev = sqrt(2 / fan_in)
        kernels = np.random.normal(0, std_dev, (out_channels, in_channels, kernel_size, kernel_size))
        return ConvolutionLayer(kernels, optimizer, stride, padding)

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        padded = pad4d(inputs, self.padding)
        # batch x in_channels x height x width

        fragments = extract_fragments4d(padded, self.kernel_size, self.stride)
        # height x width x batch x in_channels x kernelSize x kernelSize

        reshaped_fragments: np.ndarray = np.expand_dims(np.moveaxis(fragments, 2, 0), 3)
        # batch x height x width x 1 x in_channels x kernelSize x kernelSize

        multiplied: np.ndarray = reshaped_fragments * self.kernels
        # batch x height x width x out_channels x in_channels x kernelSize x kernelSize

        # Collapse 3 last dimensions into one
        new_shape = [*multiplied.shape]
        new_shape[-3] = new_shape[-3] * new_shape[-2] * new_shape[-1]

        reshaped = multiplied.reshape(new_shape[:-2])
        # batch x height x width x out_channels x elementsToSum

        summed = reshaped.sum(-1)
        # batch x height x width x out_channels

        result = np.moveaxis(summed, 3, 1)
        return result, padded

    # @profiler
    def backward(self, grad: np.ndarray, cache: np.ndarray, config: BatchConfig):
        padded = cache
        self.apply_gradient(padded, grad)

        u_current_gradient = np.transpose(grad, [2, 3, 0, 1])
        u_kernels = np.expand_dims(np.moveaxis(self.kernels, 0, 3), -2)

        batch, channels, new_height, new_width = get_dims_after_filter(padded.shape, self.kernel_size, self.stride)
        padded_input_grad = np.zeros(padded.shape, dtype="float32")
        for h in range(new_height):
            for w in range(new_width):
                h_offset = h * self.stride
                w_offset = w * self.stride

                batch_mul = u_kernels * u_current_gradient[h, w]
                batch_sum = np.sum(batch_mul, -1)
                batch_t = np.moveaxis(batch_sum, 3, 0)
                padded_input_grad[:, :, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size] \
                    += batch_t

        result = remove_padding4d(padded_input_grad, self.padding)
        return result

    def apply_gradient(self, inputs: np.ndarray, gradient: np.ndarray):
        factor = 1
        kernels_grad = np.zeros(self.kernels.shape, dtype="float32")
        shape = inputs.shape
        mean_inputs: np.ndarray = inputs.mean(0)
        mean_gradient: np.ndarray = gradient.mean(0)

        for h in range(self.kernel_size):
            for w in range(self.kernel_size):
                affected: np.ndarray = mean_inputs[:,
                                       h:shape[-2] - (self.kernel_size - h - 1): self.stride,
                                       w:shape[-1] - (self.kernel_size - w - 1): self.stride]
                mean = (np.expand_dims(mean_gradient, 1) * affected).mean((2, 3))
                kernels_grad[:, :, h, w] += mean

        self.kernels_grad += factor * kernels_grad
 */

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
