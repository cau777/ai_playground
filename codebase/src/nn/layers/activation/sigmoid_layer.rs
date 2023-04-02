use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::*;

/// Apply the sigmoid activation function.
/// https://en.wikipedia.org/wiki/Sigmoid_function
pub struct SigmoidLayer;

fn gen_name() -> String {
    "sigmoid".to_owned()
}

impl LayerOps<()> for SigmoidLayer {
    fn init(_: InitData, _: &()) -> EmptyLayerResult { Ok(()) }

    /// Apply the Sigmoid function:
    ///        1
    /// ----------------
    /// 1 + exp(-inputs)
    fn forward(data: ForwardData, _: &()) -> LayerResult {
        let ForwardData { assigner, forward_cache, inputs, .. } = data;
        let inputs = inputs.into_memory()?;

        let key = assigner.get_key(gen_name());
        let result = 1.0 / (1.0 + (-inputs).mapv(f32::exp));

        if let Some(forward_cache) = forward_cache {
            forward_cache.insert(key, vec![result.clone()]);
        }
        Ok(StoredArray::Memory { data: result })
    }

    /// The gradient of the Sigmoid function is:
    /// Sigmoid(x) * (1 - Sigmoid(x))
    fn backward(data: BackwardData, _: &()) -> LayerResult {
        let BackwardData { assigner, forward_cache, grad, .. } = data;
        let key = assigner.get_key(gen_name());
        let [cache] = remove_from_storage1(forward_cache, &key);
        let diff = 1.0 - &cache;
        Ok(StoredArray::Memory { data: grad * cache * diff })
    }
}
