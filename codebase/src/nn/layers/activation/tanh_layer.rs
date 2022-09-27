use crate::nn::layers::nn_layers::{BackwardData, ForwardData, InitData, LayerOps};
use crate::utils::ArrayDynF;

pub struct TanhLayer {}

fn gen_name() -> String {
    "tanh".to_owned()
}

impl LayerOps<()> for TanhLayer {
    fn init(data: InitData, _: &()) {}

    fn forward(data: ForwardData, _: &()) -> ArrayDynF {
        let ForwardData { inputs, assigner, forward_cache, .. } = data;

        let result = inputs.mapv(f32::tanh);
        let key = assigner.get_key(gen_name());
        forward_cache.insert(key, vec![result.clone()]);
        result
    }

    fn backward(data: BackwardData, _: &()) -> ArrayDynF {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let cache = &forward_cache.get(&key).unwrap()[0];
        grad * (1.0 - cache * cache)
    }
}
