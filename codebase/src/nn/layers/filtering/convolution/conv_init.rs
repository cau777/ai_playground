use ndarray::{ErrorKind, ShapeError};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use crate::Array4F;
use crate::nn::layers::filtering::convolution::{ConvolutionConfig, ConvolutionInitMode, gen_name};
use crate::nn::layers::nn_layers::{EmptyLayerResult, InitData};

pub fn init(data: InitData, layer_config: &ConvolutionConfig) -> EmptyLayerResult {
    let InitData { assigner, storage } = data;
    let ConvolutionConfig { in_channels, out_channels, kernel_size, init_mode, .. } = layer_config.clone();
    let key = assigner.get_key(gen_name(layer_config));

    if let std::collections::hash_map::Entry::Vacant(e) = storage.entry(key) {
        let kernel = match init_mode {
            ConvolutionInitMode::Kernel(k) => {
                let shape = k.shape();
                if shape[0] != out_channels
                    || shape[1] != in_channels
                    || shape[2] != kernel_size
                    || shape[3] != kernel_size
                {
                    return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
                }
                k
            }
            ConvolutionInitMode::HeNormal() => {
                let fan_in = in_channels * kernel_size * kernel_size;
                let std_dev = (2.0 / fan_in as f32).sqrt();
                let dist = Normal::new(0.0, std_dev)?;
                Array4F::random((out_channels, in_channels, kernel_size, kernel_size), dist)
            }
        };

        e.insert(vec![kernel.into_dyn()]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::filtering::convolution::ConvolutionInitMode::HeNormal;
    use crate::nn::layers::nn_layers::{GenericStorage};
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use super::*;

    #[test]
    fn test_init() {
        let config = ConvolutionConfig {
            in_channels: 1,
            out_channels: 4,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            init_mode: HeNormal(),
            lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        };

        let mut assigner = KeyAssigner::new();
        let mut storage = GenericStorage::new();
        let data = InitData {
            assigner: &mut assigner,
            storage: &mut storage,
        };
        init(data, &config).unwrap();
    }
}