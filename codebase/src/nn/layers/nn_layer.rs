use std::any::Any;
use crate::nn::batch_config::BatchConfig;
use crate::utils::ArrayF;

pub type Cache = Box<dyn Any>;

pub struct LayerOutput<D>(pub ArrayF<D>, pub Cache);

pub trait NNLayer<DIn, DOut> {
    fn forward(&self, inputs: ArrayF<DIn>, config: &BatchConfig) -> LayerOutput<DOut>;

    fn backward(&mut self, grad: ArrayF<DOut>, cache: Cache, config: &BatchConfig) -> ArrayF<DIn>;

    fn train(&mut self, config: &BatchConfig);
}