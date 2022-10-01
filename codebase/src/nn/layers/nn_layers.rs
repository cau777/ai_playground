use std::collections::HashMap;
use std::error;
use crate::nn::batch_config::BatchConfig;
use crate::nn::layers::dense_layer::{DenseLayer, DenseLayerConfig};
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::activation::relu_layer::ReluLayer;
use crate::nn::layers::activation::sigmoid_layer::SigmoidLayer;
use crate::nn::layers::activation::tanh_layer::TanhLayer;
use crate::nn::layers::debug_layer::{DebugLayer, DebugLayerConfig};
use crate::nn::layers::sequential_layer::{SequentialLayer, SequentialLayerConfig};
use crate::utils::ArrayDynF;

#[derive(Clone)]
pub enum Layer {
    Dense(DenseLayerConfig),
    Sequential(SequentialLayerConfig),
    Tanh,
    Sigmoid,
    Relu,
    Debug(DebugLayerConfig),
}

pub struct InitData<'a> {
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage,
}

pub struct ForwardData<'a> {
    pub inputs: ArrayDynF,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage,
    pub forward_cache: &'a mut GenericStorage,
}

pub struct BackwardData<'a> {
    pub grad: ArrayDynF,
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage,
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
        Dense(c) => DenseLayer::init(data, c),
        Relu => ReluLayer::init(data, &()),
        Tanh => TanhLayer::init(data, &()),
        Sigmoid => SigmoidLayer::init(data, &()),
        Sequential(c) => SequentialLayer::init(data, c),
        Debug(c) => DebugLayer::init(data, c),
    }
}

pub fn forward_layer(layer: &Layer, data: ForwardData) -> LayerResult {
    use Layer::*;
    match layer {
        Dense(c) => DenseLayer::forward(data, c),
        Sequential(c) => SequentialLayer::forward(data, c),
        Tanh => TanhLayer::forward(data, &()),
        Sigmoid => SigmoidLayer::forward(data, &()),
        Relu => ReluLayer::forward(data, &()),
        Debug(c) => DebugLayer::forward(data, c),
    }
}

pub fn backward_layer(layer: &Layer, data: BackwardData) -> LayerResult {
    use Layer::*;
    match layer {
        Dense(c) => DenseLayer::backward(data, c),
        Sequential(c) => SequentialLayer::backward(data, c),
        Tanh => TanhLayer::backward(data, &()),
        Sigmoid => SigmoidLayer::backward(data, &()),
        Relu => ReluLayer::backward(data, &()),
        Debug(c) => DebugLayer::backward(data, c),
    }
}

pub fn train_layer(layer: &Layer, data: TrainData) -> EmptyLayerResult {
    use Layer::*;
    match layer {
        Dense(c) => DenseLayer::train(data, c),
        Sequential(c) => SequentialLayer::train(data, c),
        _ => Ok(())
    }
}
