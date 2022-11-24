use std::collections::HashMap;
use std::error;
use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::*;
use crate::nn::layers::activation::*;
use crate::utils::ArrayDynF;
use super::convolution_layer::ConvolutionConfig;
use super::expand_dim_layer::ExpandDimConfig;
use super::max_pool_layer::MaxPoolConfig;

#[derive(Clone, Debug)]
pub enum Layer {
    Dense(dense_layer::DenseConfig),
    Sequential(sequential_layer::SequentialConfig),
    Tanh,
    Sigmoid,
    Relu,
    Debug(debug_layer::DebugLayerConfig),
    Convolution(ConvolutionConfig),
    MaxPool(MaxPoolConfig),
    Flatten,
    ExpandDim(ExpandDimConfig),
    Dropout(dropout_layer::DropoutConfig)
}

pub struct InitData<'a> {
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage,
}

pub struct ForwardData<'a> {
    pub inputs: ArrayDynF,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a GenericStorage,
    pub forward_cache: &'a mut GenericStorage,
}

pub struct BackwardData<'a> {
    pub grad: ArrayDynF,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a GenericStorage,
    pub forward_cache: &'a mut GenericStorage,
    pub backward_cache: &'a mut GenericStorage,
}

pub struct TrainData<'a> {
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage,
    pub backward_cache: &'a mut GenericStorage,
}

pub type GenericStorage = HashMap<String, Vec<ArrayDynF>>;

pub type LayerError = Box<dyn error::Error>;
pub type EmptyLayerResult = Result<(), LayerError>;
pub type LayerResult = Result<ArrayDynF, LayerError>;

pub trait LayerOps<T> {
    fn init(data: InitData, layer_config: &T) -> EmptyLayerResult;

    fn forward(data: ForwardData, layer_config: &T) -> LayerResult;

    fn backward(data: BackwardData, layer_config: &T) -> LayerResult;
}

pub trait TrainableLayerOps<T> {
    fn train(data: TrainData, layer_config: &T) -> EmptyLayerResult;
}

pub fn init_layer(layer: &Layer, data: InitData) -> EmptyLayerResult {
    use Layer::*;
    match layer {
        Dense(c) => dense_layer::DenseLayer::init(data, c),
        Relu => relu_layer::ReluLayer::init(data, &()),
        Tanh => tanh_layer::TanhLayer::init(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::init(data, &()),
        Sequential(c) => sequential_layer::SequentialLayer::init(data, c),
        Debug(c) => debug_layer::DebugLayer::init(data, c),
        Convolution(c) => convolution_layer::ConvolutionLayer::init(data, c),
        MaxPool(c) => max_pool_layer::MaxPoolLayer::init(data, c),
        Flatten => flatten_layer::FlattenLayer::init(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::init(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::init(data, c),
    }
}

pub fn forward_layer(layer: &Layer, data: ForwardData) -> LayerResult {
    use Layer::*;
    match layer {
        Dense(c) => dense_layer::DenseLayer::forward(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::forward(data, c),
        Tanh => tanh_layer::TanhLayer::forward(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::forward(data, &()),
        Relu => relu_layer::ReluLayer::forward(data, &()),
        Debug(c) => debug_layer::DebugLayer::forward(data, c),
        Convolution(c) => convolution_layer::ConvolutionLayer::forward(data, c),
        MaxPool(c) => max_pool_layer::MaxPoolLayer::forward(data, c),
        Flatten => flatten_layer::FlattenLayer::forward(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::forward(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::forward(data, c),
    }
}

pub fn backward_layer(layer: &Layer, data: BackwardData) -> LayerResult {
    use Layer::*;
    match layer {
        Dense(c) => dense_layer::DenseLayer::backward(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::backward(data, c),
        Tanh => tanh_layer::TanhLayer::backward(data, &()),
        Sigmoid => sigmoid_layer::SigmoidLayer::backward(data, &()),
        Relu => relu_layer::ReluLayer::backward(data, &()),
        Debug(c) => debug_layer::DebugLayer::backward(data, c),
        Convolution(c) => convolution_layer::ConvolutionLayer::backward(data, c),
        MaxPool(c) => max_pool_layer::MaxPoolLayer::backward(data, c),
        Flatten => flatten_layer::FlattenLayer::backward(data, &()),
        ExpandDim(c) => expand_dim_layer::ExpandDimLayer::backward(data, c),
        Dropout(c) => dropout_layer::DropoutLayer::backward(data, c),
    }
}

pub fn train_layer(layer: &Layer, data: TrainData) -> EmptyLayerResult {
    use Layer::*;
    match layer {
        Dense(c) => dense_layer::DenseLayer::train(data, c),
        Sequential(c) => sequential_layer::SequentialLayer::train(data, c),
        Convolution(c) => convolution_layer::ConvolutionLayer::train(data, c),
        _ => Ok(()),
    }
}
