use crate::nn::layers::nn_layers::{BackwardData, ForwardData, InitData, LayerOps};
use crate::utils::ArrayDynF;

pub struct SigmoidLayer {}

fn gen_name() -> String {
    "sigmoid".to_owned()
}

impl LayerOps<()> for SigmoidLayer {
    fn init(data: InitData, _: &()) {}

    fn forward(data: ForwardData, _: &()) -> ArrayDynF {
        let ForwardData { assigner, forward_cache, inputs, .. } = data;
        let key = assigner.get_key(gen_name());
        let result = 1.0 / (1.0 + (-inputs).mapv(f32::exp));
        forward_cache.insert(key, vec![result.clone()]);
        result
    }

    fn backward(data: BackwardData, _: &()) -> ArrayDynF {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let cache: &ArrayDynF = &forward_cache.get(&key).unwrap()[0];
        grad * cache * (1.0 - cache)
    }
}
