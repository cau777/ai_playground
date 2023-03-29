use std::collections::HashMap;
use crate::gpu::gpu_data::GlobalGpu;
use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::*;
use crate::nn::layers::activation::*;
use crate::utils::{ArrayDynF, GenericResult};
use crate::nn::layers::filtering::convolution::ConvolutionConfig;
use super::expand_dim_layer::ExpandDimConfig;
use crate::nn::layers::filtering::max_pool::MaxPoolConfig;
use crate::nn::layers::stored_array::StoredArray;

/// Enum to represent the layers that create the model and its parameters
#[derive(Clone, Debug)]
pub enum Layer {
    /// Dense Layer: perform matrix multiplication between the input and a weights matrix, and add
    /// a biases matrix.
    /// ### Trainable
    /// * Weights
    /// * Biases
    Dense(dense_layer::DenseConfig),

    /// Simply executes the layers in sequential order, passing the output of a layer as the input
    /// of the next layer. Will probably be the **root** of the model.
    Sequential(sequential_layer::SequentialConfig),

    /// Apply the activation function TanH. Better than Sigmoid for handling negative values.
    /// https://pt.wikipedia.org/wiki/Tangente_hiperb%C3%B3lica
    Tanh,

    /// Apply the sigmoid activation function.
    /// https://en.wikipedia.org/wiki/Sigmoid_function
    Sigmoid,

    /// Apply the Rectified Linear Unit (ReLu) activation function. That means:
    /// * For x >= 0: x
    /// * For x < 0: 0
    Relu,

    /// Just executes the callback functions and passes the data unchanged. Useful for debugging NaNs.
    Debug(debug_layer::DebugLayerConfig),

    /// Apply the convolution operation with 2D filters. That means passing a filter through the last
    /// 2 dimension of the input (usually height and width). In position of the filter, the sum of the
    /// product of those input values and the kernel is computed. Requires a 4 dimensional input (one being
    /// the batch).
    /// ### Trainable
    /// * Kernel
    /// https://en.wikipedia.org/wiki/Convolutional_neural_network
    Convolution(ConvolutionConfig),

    /// Apply MAX operation with 2D filters, That means passing a filter through the last
    /// 2 dimension of the input (usually height and width). In position of the filter, the maximum
    /// value of those input values is computed. Requires a 4 dimensional input (one being the batch).
    /// Use for reducing the size of arrays after **Convolution**.
    /// https://deepai.org/machine-learning-glossary-and-terms/max-pooling
    MaxPool(MaxPoolConfig),

    /// Flattens all dimensions except the batch. The result will always be a 2D array. Useful for
    /// passing **Convolution** results into **Dense** layers.
    Flatten,

    /// Adds a new axis of length 1 in the specified place. By default, it ignores Batch dimension (which is always
    /// the first), so adding an axis in 1 would result: (Batch, Dim_1, 1, Dim_2, Dim_3).
    ExpandDim(ExpandDimConfig),

    /// Randomly nullifies a percentage of the inputs. Useful for avoiding overfitting.
    /// https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
    Dropout(dropout_layer::DropoutConfig),

    /// Feed the same input to all its immediate children and concatenate the result
    /// No extra axis is created
    Concat(concat_layer::ConcatConfig),

    /// Receives 2 inputs A and B, and outputs A - B
    /// It's useful because it makes it easier for the model to produce negative values with ReLu layers
    TwoComplementsTransformer,
}

pub struct InitData<'a> {
    pub assigner: &'a mut KeyAssigner,

    /// Persistent storage to store/update things like weights
    pub storage: &'a mut GenericStorage,
}

pub struct ForwardData<'a> {
    pub inputs: StoredArray,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,

    /// Persistent storage to store things like weights
    pub storage: &'a GenericStorage,

    /// Temporary storage that will be fed to `backward()`.
    pub forward_cache: Option<&'a mut GenericStorage>,

    /// Temporary storage used to reuse computations from the previous iteration, if available 
    pub prev_iteration_cache: Option<&'a mut GenericStorage>,
    pub gpu: Option<GlobalGpu>,
}

pub struct BackwardData<'a> {
    pub grad: ArrayDynF,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    
    /// Persistent storage to store things like weights
    pub storage: &'a GenericStorage,

    /// Temporary storage that comes from `forward()`.
    pub forward_cache: &'a mut GenericStorage,

    /// Temporary storage that will be fed to `train()`.
    pub backward_cache: &'a mut GenericStorage,
    pub gpu: Option<GlobalGpu>,
}

pub struct TrainData<'a> {
    pub batch_config: &'a BatchConfig,
    
    pub assigner: &'a mut KeyAssigner,
    /// Persistent storage to store/update things like weights
    pub storage: &'a mut GenericStorage,
    
    /// Temporary storage that comes from `backward()`.
    pub backward_cache: &'a mut GenericStorage,
}

/// Type alias for a map on which layers store all the needed data.
/// Key: unique string for a layer
/// Value: Vector of NDimensional arrays
/// The purpose of this type is to provide a centralized storage for trainable parameters
/// as opposed to the objected oriented approach where layers are classes that stores parameters as fields.
/// The advantage is that it can be easily serialized, and most of the times, layers can be added or
/// removed without progress loss
pub type GenericStorage = HashMap<String, Vec<ArrayDynF>>;

pub type EmptyLayerResult = GenericResult<()>;
pub type LayerResult = GenericResult<StoredArray>;

pub trait LayerOps<T> {
    fn init(data: InitData, layer_config: &T) -> EmptyLayerResult;

    fn forward(data: ForwardData, layer_config: &T) -> LayerResult;

    fn backward(data: BackwardData, layer_config: &T) -> LayerResult;
}

pub trait TrainableLayerOps<T> {
    fn train(data: TrainData, layer_config: &T) -> EmptyLayerResult;
}

/// Call `ìnit()` in the appropriate layer. Not intended to be called directly.
pub fn init_layer(layer: &Layer, data: InitData) -> EmptyLayerResult {
    use Layer::*;
    use crate::nn::layers::filtering::{convolution, max_pool};
    match layer {
        Dense(c) => dense_layer::DenseLayer::init(data, c),
        Relu => relu_layer::ReluLayer::init(data, &()),
        Tanh => tanh_layer::TanhLayer::init(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::init(data, &()),
        Sequential(c) => sequential_layer::SequentialLayer::init(data, c),
        Debug(c) => debug_layer::DebugLayer::init(data, c),
        Convolution(c) => convolution::ConvolutionLayer::init(data, c),
        MaxPool(c) => max_pool::MaxPoolLayer::init(data, c),
        Flatten => flatten_layer::FlattenLayer::init(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::init(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::init(data, c),
        Concat(c) => concat_layer::ConcatLayer::init(data, c),
        TwoComplementsTransformer => two_complements_transformer_layer::TwoComplementsTransformerLayer::init(data, &()),
    }
}

/// Call `forward()` in the appropriate layer. Not intended to be called directly.
pub fn forward_layer(layer: &Layer, data: ForwardData) -> LayerResult {
    use Layer::*;
    use crate::nn::layers::filtering::{convolution, max_pool};
    match layer {
        Dense(c) => dense_layer::DenseLayer::forward(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::forward(data, c),
        Tanh => tanh_layer::TanhLayer::forward(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::forward(data, &()),
        Relu => relu_layer::ReluLayer::forward(data, &()),
        Debug(c) => debug_layer::DebugLayer::forward(data, c),
        Convolution(c) => convolution::ConvolutionLayer::forward(data, c),
        MaxPool(c) => max_pool::MaxPoolLayer::forward(data, c),
        Flatten => flatten_layer::FlattenLayer::forward(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::forward(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::forward(data, c),
        Concat(c) => concat_layer::ConcatLayer::forward(data, c),
        TwoComplementsTransformer => two_complements_transformer_layer::TwoComplementsTransformerLayer::forward(data, &()),
    }
}

/// Call `backward()` in the appropriate layer. Not intended to be called directly.
pub fn backward_layer(layer: &Layer, data: BackwardData) -> LayerResult {
    use Layer::*;
    use crate::nn::layers::filtering::{convolution, max_pool};
    match layer {
        Dense(c) => dense_layer::DenseLayer::backward(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::backward(data, c),
        Tanh => tanh_layer::TanhLayer::backward(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::backward(data, &()),
        Relu => relu_layer::ReluLayer::backward(data, &()),
        Debug(c) => debug_layer::DebugLayer::backward(data, c),
        Convolution(c) => convolution::ConvolutionLayer::backward(data, c),
        MaxPool(c) => max_pool::MaxPoolLayer::backward(data, c),
        Flatten => flatten_layer::FlattenLayer::backward(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::backward(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::backward(data, c),
        Concat(c) => concat_layer::ConcatLayer::backward(data, c),
        TwoComplementsTransformer => two_complements_transformer_layer::TwoComplementsTransformerLayer::backward(data, &()),
    }
}

/// Call `train()` in the appropriate layer. If the layer doesn't provide an implementation, nothing
/// will happen. Not intended to be called directly.
pub fn train_layer(layer: &Layer, data: TrainData) -> EmptyLayerResult {
    use Layer::*;
    use crate::nn::layers::filtering::convolution;
    match layer {
        Dense(c) => dense_layer::DenseLayer::train(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::train(data, c),
        Convolution(c) => convolution::ConvolutionLayer::train(data, c),
        Concat(c) => concat_layer::ConcatLayer::train(data, c),
        _ => Ok(()),
    }
}
