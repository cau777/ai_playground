use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};

pub struct TanhLayer {}

fn gen_name() -> String {
    "tanh".to_owned()
}

impl LayerOps<()> for TanhLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { inputs, assigner, forward_cache, .. } = data;

        let result = inputs.mapv(f32::tanh);
        let key = assigner.get_key(gen_name());
        forward_cache.insert(key, vec![result.clone()]);
        Ok(result)
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let [cache] = remove_from_storage1(forward_cache, &key);
        let square = &cache * &cache;
        Ok(grad * (1.0 - square))
    }
}
