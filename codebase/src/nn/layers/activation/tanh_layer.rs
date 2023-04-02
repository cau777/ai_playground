use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};
use crate::nn::layers::stored_array::StoredArray;

/// Apply the activation function TanH. Better than Sigmoid for handling negative values.
/// https://pt.wikipedia.org/wiki/Tangente_hiperb%C3%B3lica
pub struct TanhLayer;

fn gen_name() -> String {
    "tanh".to_owned()
}

impl LayerOps<()> for TanhLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    /// Apply the f32::tanh function
    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { inputs, assigner, forward_cache, .. } = data;
        let inputs = inputs.into_memory()?;

        let result = inputs.mapv(f32::tanh);
        let key = assigner.get_key(gen_name());

        if let Some(forward_cache) = forward_cache {
            forward_cache.insert(key, vec![result.clone()]);
        }
        Ok(StoredArray::Memory {data: result})
    }

    /// The gradient of tanh is:
    /// 1 - tanh(x)²
    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let [cache] = remove_from_storage1(forward_cache, &key);
        let square = &cache * &cache;
        Ok(StoredArray::Memory { data: grad * (1.0 - square) })
    }
}
