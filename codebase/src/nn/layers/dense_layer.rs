use std::iter::zip;
use ndarray::{Axis, Ix2, Ix3, s, ShapeBuilder, stack};
use crate::nn::batch_config::BatchConfig;
use crate::nn::layers::nn_layer::{Cache, LayerOutput, NNLayer};
use crate::utils::{Array1F, Array2F, ArrayF};

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

pub struct DenseLayer {
    weights: Array2F,
    biases: Array1F,
    weights_grad: Array2F,
    biases_grad: Array1F,
    out_values: usize,
    in_values: usize,
}

impl DenseLayer {
    pub fn new(weights: Array2F, biases: Array1F) -> Self {
        let shape = weights.shape();
        Self {
            out_values: shape[0],
            in_values: shape[1],
            biases_grad: &biases * 0.0,
            weights_grad: &weights * 0.0,
            biases,
            weights,
        }
    }

    /*
    @staticmethod
    def create_random(in_values: int, out_values: int, weights_optimizer: LrOptimizer, biases_optimizer: LrOptimizer,
                      biases_enabled: bool = True):
        if in_values < 1:
            raise ValueError("in_values can't be less than 1")

        if out_values < 1:
            raise ValueError("out_values can't be less than 1")

        std_dev = out_values ** -0.5
        weights = np.random.normal(0, std_dev, (out_values, in_values)).astype("float32")
        biases = np.zeros((out_values, 1), dtype="float32")
        return DenseLayer(weights, biases, biases_enabled, weights_optimizer, biases_optimizer)
     */
    pub fn random(in_values: usize, out_values: usize) -> Self {
        let std_dev = (out_values as f32).powf(-0.5);
        let dist = Normal::new(0.0, std_dev).unwrap();
        let weights = Array2F::random((out_values, in_values).f(), dist);
        let biases = Array1F::zeros((out_values).f());

        Self::new(weights, biases)
    }
}

impl NNLayer<Ix2, Ix2> for DenseLayer {
    /*
    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        result = self.weights @ np.expand_dims(inputs, -1)
        if self.biases_enabled:
            result += self.biases

        return np.squeeze(result, -1), inputs
     */
    fn forward(&self, inputs: ArrayF<Ix2>, config: &BatchConfig) -> LayerOutput<Ix2> {
        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, self.out_values).f());

        inputs.outer_iter()
            .map(|o| self.weights.dot(&o))
            .map(|o| o + &self.biases)
            .enumerate()
            .for_each(|(index, o)| result.slice_mut(s![index, ..]).assign(&o));


        LayerOutput(result, Box::new(inputs))
    }

    /*
    inputs = cache

        grad = np.expand_dims(grad, -1)
        inputs = np.expand_dims(inputs, -2)

        weights_error = grad @ inputs
        self.weights_grad += weights_error.mean(0)

        if self.biases_enabled:
            self.biases_grad += grad.mean(0)

        return np.squeeze(self.weights.T @ grad, -1)
     */
    fn backward(&mut self, grad: ArrayF<Ix2>, cache: Cache, config: &BatchConfig) -> ArrayF<Ix2> {
        let inputs: &Array2F = cache.downcast_ref().unwrap();
        let mut weights_error = Array2F::default((self.out_values, self.in_values));

        zip(inputs.outer_iter(), grad.outer_iter())
            .map(|(i, g)| {
                let gt = g.insert_axis(Axis(1));
                let it = i.insert_axis(Axis(0));
                gt.dot(&it)
            })
            .for_each(|o| weights_error += &o);

        // self.weights_grad += &weights_error.mean_axis(Axis(0)).unwrap();
        self.biases_grad += &grad.mean_axis(Axis(0)).unwrap();

        let batch_size = inputs.shape()[0];
        let mut result = Array2F::default((batch_size, self.in_values));
        let weights_t = self.weights.t();
        grad.outer_iter()
            // .map(|o| o.insert_axis(Axis(1)))
            .map(|o| weights_t.dot(&o))
            .enumerate()
            .for_each(|(index, o)| result.slice_mut(s![index, ..]).assign(&o));

        result
    }

    /*
    def train(self, config: BatchConfig):
        self.weights += self.weights_optimizer.optimize(self.weights_grad, config)
        self.weights_grad *= 0

        if self.biases_enabled:
            self.biases += self.biases_optimizer.optimize(self.biases_grad, config)
            self.biases_grad *= 0
     */
    fn train(&mut self, config: &BatchConfig) {
        self.weights += &(&self.weights_grad*0.02);
        self.weights_grad.fill(0.0);

        self.biases += &self.biases_grad;
        self.biases_grad.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array};
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use crate::nn::batch_config::DEBUG_CONFIG;
    use crate::nn::layers::dense_layer::DenseLayer;
    use crate::nn::layers::nn_layer::NNLayer;
    use crate::nn::loss::mse_loss::MseLossFunc;
    use crate::utils::{Array2F, ArrayF, arrays_almost_equal};

    #[test]
    fn test_forward() {
        let input = array![
            [1.0, 2.0],
            [2.0, 3.0]
        ];
        let weights = array![
            [0.7, 0.0],
            [0.1, 0.4],
            [0.8, 0.6]
        ];
        let expected = array![
            [0.7, 0.9, 2.0],
            [1.4, 1.4, 3.4]
        ];
        let layer = DenseLayer::new(weights, array![0.0, 0.0, 0.0]);
        let output = layer.forward(input, &DEBUG_CONFIG);

        assert!(arrays_almost_equal(&output.0, &expected));
    }

    #[test]
    fn test_backward() {
        let inputs: Array2F = array![[0.8, 0.7]];
        let grad = array![[0.1, 0.2, 0.3]];
        let weights = array![
            [0.7, 0.0],
            [0.1, 0.4],
            [0.8, 0.6]
        ];

        let mut layer = DenseLayer::new(weights, array![0.0, 0.0, 0.0]);
        layer.backward(grad, Box::new(inputs), &DEBUG_CONFIG);
    }

    #[test]
    fn test_train() {
        let mut layer = DenseLayer::random(20, 30);
        let inputs = Array2F::random((2, 20), Normal::new(0.0, 0.5).unwrap());
        let expected = Array2F::random((2, 1), Normal::new(0.0, 0.5).unwrap()).into_dyn();
        let loss = MseLossFunc {};

        for n in 0..1000 {
            let inputs = inputs.clone();
            let forward = layer.forward(inputs, &DEBUG_CONFIG);
            let outputs = forward.0.into_dyn();
            let error = loss.calc_loss(&expected, &outputs);
            if n % 10 == 0 {
                println!("{}", error.sum());
            }
            let grad = loss.calc_loss_grad(&expected, &outputs);
            layer.backward(grad.into_dimensionality().unwrap(), forward.1, &DEBUG_CONFIG);
            layer.train(&DEBUG_CONFIG);
        }
    }
}


/*

class DenseLayer(NNLayer):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, biases_enabled: bool, weights_optimizer: LrOptimizer,
                 biases_optimizer: LrOptimizer):
        self.weights = weights
        self.biases = biases
        self.weights_grad = weights * 0
        self.biases_grad = biases * 0
        self.biases_enabled = biases_enabled
        self.weights_optimizer = weights_optimizer
        self.biases_optimizer = biases_optimizer

    @staticmethod
    def create_random(in_values: int, out_values: int, weights_optimizer: LrOptimizer, biases_optimizer: LrOptimizer,
                      biases_enabled: bool = True):
        if in_values < 1:
            raise ValueError("in_values can't be less than 1")

        if out_values < 1:
            raise ValueError("out_values can't be less than 1")

        std_dev = out_values ** -0.5
        weights = np.random.normal(0, std_dev, (out_values, in_values)).astype("float32")
        biases = np.zeros((out_values, 1), dtype="float32")
        return DenseLayer(weights, biases, biases_enabled, weights_optimizer, biases_optimizer)

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        result = self.weights @ np.expand_dims(inputs, -1)
        if self.biases_enabled:
            result += self.biases

        return np.squeeze(result, -1), inputs

    def backward(self, grad: np.ndarray, cache: np.ndarray, config: BatchConfig) -> np.ndarray:
        inputs = cache

        grad = np.expand_dims(grad, -1)
        inputs = np.expand_dims(inputs, -2)

        weights_error = grad @ inputs
        self.weights_grad += weights_error.mean(0)

        if self.biases_enabled:
            self.biases_grad += grad.mean(0)

        return np.squeeze(self.weights.T @ grad, -1)


 */