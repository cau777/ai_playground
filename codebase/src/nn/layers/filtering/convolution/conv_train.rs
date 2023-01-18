use std::ops::AddAssign;
use crate::nn::generic_storage::{get_mut_from_storage, remove_from_storage1};
use crate::nn::layers::filtering::convolution::{ConvolutionConfig, ConvolutionLayer, gen_name};
use crate::nn::layers::nn_layers::{EmptyLayerResult, TrainableLayerOps, TrainData};
use crate::nn::lr_calculators::lr_calculator::{apply_lr_calc, LrCalcData};

impl TrainableLayerOps<ConvolutionConfig> for ConvolutionLayer {
    fn train(data: TrainData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
        let TrainData {
            storage,
            backward_cache,
            assigner,
            batch_config,
        } = data;
        let key = assigner.get_key(gen_name(layer_config));

        let [kernel_grad] = remove_from_storage1(backward_cache, &key);
        let kernel_grad = apply_lr_calc(
            &layer_config.lr_calc,
            kernel_grad,
            LrCalcData {
                batch_config,
                storage,
                assigner,
            },
        )?;

        let kernel = get_mut_from_storage(storage, &key, 0);
        kernel.add_assign(&kernel_grad);
        Ok(())
    }
}