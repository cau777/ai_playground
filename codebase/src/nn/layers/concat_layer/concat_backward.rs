use ndarray::{Axis, Slice};
use crate::ArrayDynF;
use crate::nn::generic_storage::remove_from_storage1;
use crate::nn::layers::concat_layer::{ConcatConfig, gen_name};
use crate::nn::layers::nn_layers::*;

pub fn backward(data: BackwardData, layer_config: &ConcatConfig) -> LayerResult {
    let key = data.assigner.get_key(gen_name(layer_config));
    let [cache] = remove_from_storage1(data.forward_cache, &key);
    let dim = layer_config.dim + 1;
    let splits: Vec<_> = cache.iter().map(|o| o.round() as usize).collect();

    let mut end = data.grad.shape()[dim];
    let mut result: Option<ArrayDynF> = None;
    let layer_count = layer_config.layers.len();

    for i in 0..layer_count {
        let i = layer_count - i - 1;
        let layer = &layer_config.layers[i];
        let split = splits[i];
        let grad = data.grad.slice_axis(Axis(dim), Slice::from((end - split)..end));

        let layer_result = backward_layer(layer, BackwardData {
            grad: grad.to_owned(),
            batch_config: data.batch_config,
            assigner: data.assigner,
            storage: data.storage,
            gpu: data.gpu.clone(),
            forward_cache: data.forward_cache,
            backward_cache: data.backward_cache,
        })?.into_memory()?;
        // println!("conv_back {:?}", layer_result.shape());

        result = Some(match result {
            Some(v) => v + layer_result,
            None => layer_result,
        });
        end -= split;
    }

    Ok(result.unwrap().into())
}