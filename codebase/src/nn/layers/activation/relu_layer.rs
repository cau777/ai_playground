use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};

pub struct ReluLayer {}

impl ReluLayer {}

fn gen_name() -> String {
    "relu".to_owned()
}

impl LayerOps<()> for ReluLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { assigner, inputs, forward_cache, .. } = data;
        let key = assigner.get_key(gen_name());
        let positives = inputs.mapv(|o| (o > 0.0) as u8 as f32);
        let result = inputs * &positives;
        forward_cache.insert(key, vec![positives]);
        Ok(result)
    }

    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());

        let [cache] = remove_from_storage1(forward_cache, &key);
        println!("g={:?} c={:?}", grad.shape(), cache.shape());
        Ok(grad * cache)
    }
}