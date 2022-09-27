use crate::nn::layers::nn_layers::{BackwardData, ForwardData, InitData, LayerOps};
use crate::utils::ArrayDynF;

pub struct ReluLayer {}

impl ReluLayer {}

fn gen_name() -> String {
    "relu".to_owned()
}

impl LayerOps<()> for ReluLayer {
    fn init(data: InitData, _: &()) {}

    fn forward(data: ForwardData, _: &()) -> ArrayDynF {
        let ForwardData { assigner, inputs, forward_cache, .. } = data;
        let key = assigner.get_key(gen_name());
        let positives = inputs.mapv(|o| (o > 0.0) as u8 as f32);
        let result = inputs * &positives;
        forward_cache.insert(key, vec![positives]);
        result
    }

    fn backward(data: BackwardData, _: &()) -> ArrayDynF {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let cache = &forward_cache.get(&key).unwrap()[0];
        grad * cache
    }
}