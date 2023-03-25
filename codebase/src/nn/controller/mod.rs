mod evaluating;
mod training;
mod testing;

use std::sync::{Arc, Mutex, RwLock};
use crate::gpu::gpu_data::{GlobalGpu, GpuData};
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::*;
use crate::nn::loss::loss_func::LossFunc;
use crate::utils::GenericResult;

/// Main struct to train and use the AI model
/// ```
/// use codebase::nn::controller::NNController;
/// use codebase::nn::train_config::TrainConfig;
///
/// let mut controller = NNController::new(main_layer, loss_func).unwrap();
///
/// println!("Started training");
/// for epoch in 0..10 {
///     // It's better to split the data into random batches
///     let loss = controller.train_batch(data_inputs.clone(), &data_expected ).unwrap();
///     println!("Epoch {} finished with avg loss {}", epoch, loss);
/// }
///
/// println!("Finished training. Started validating");
/// let final_loss = controller.test_batch(data_inputs, &data_expected).unwrap();
/// println!("Validated loss = {}", final_loss);
///
/// ```
pub struct NNController {
    main_layer: Layer,
    storage: GenericStorage,
    loss: LossFunc,
    cached_gpu_age: Mutex<u64>,
    cached_gpu: Arc<RwLock<Option<Option<GlobalGpu>>>>,
}

impl NNController {
    /// Create a controller with an empty storage and init its layers
    pub fn new(main_layer: Layer, loss: LossFunc) -> GenericResult<Self> {
        let mut storage = GenericStorage::new();
        let mut assigner = KeyAssigner::new();
        init_layer(
            &main_layer,
            InitData {
                assigner: &mut assigner,
                storage: &mut storage,
            },
        )?;

        Ok(Self {
            main_layer,
            storage,
            loss,
            cached_gpu: Arc::new(RwLock::new(None)),
            cached_gpu_age: Mutex::new(0),
        })
    }

    /// Create a controller with the provided storage and init its layers
    pub fn load(main_layer: Layer, loss: LossFunc, mut storage: GenericStorage) -> GenericResult<Self> {
        let mut assigner = KeyAssigner::new();
        init_layer(
            &main_layer,
            InitData {
                assigner: &mut assigner,
                storage: &mut storage,
            },
        )?;

        Ok(Self {
            main_layer,
            storage,
            loss,
            cached_gpu: Arc::new(RwLock::new(None)),
            cached_gpu_age: Mutex::new(0),
        })
    }

    /// Return a copy of the inner storage
    pub fn export(&self) -> GenericStorage {
        self.storage.clone()
    }

    fn get_gpu(&self) -> Option<GlobalGpu> {
        let mut gpu_age = self.cached_gpu_age.lock().unwrap();
        *gpu_age += 1;

        if self.cached_gpu.read().unwrap().is_none()
            || *gpu_age > 10_000
        {
            *gpu_age = 0;
            let new_gpu = match GpuData::new_global() {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("{:?}", e);
                    None
                }
            };

            let mut prev =self.cached_gpu.write().unwrap();
            if let Some(prev) = prev.as_ref() {
                if let Some(prev ) = prev.as_ref() {
                    prev.contexts.write().unwrap().clear();
                    prev.cmd_alloc.clear(prev.queue.queue_family_index());
                    prev.descriptor_alloc.clear_all();
                }
            }
            *prev = Some(new_gpu);
        }

        self.cached_gpu.read().unwrap().clone().unwrap()
    }

    fn finish_method(&self) -> GenericResult<()> {
        if let Some(gpu) = self.get_gpu() {
            gpu.reset_fast_mem_alloc()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Read};

    use crate::{
        integration::layers_loading::load_model_xml,
        utils::{Array2F, Array3F},
    };

    use super::*;

    #[test]
    #[ignore]
    fn test_digits_model() {
        let mut file = OpenOptions::new()
            .read(true)
            .open("../temp/digits/config.xml")
            .unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let config = load_model_xml(&bytes).unwrap();
        let mut controller = NNController::new(config.main_layer, config.loss_func).unwrap();
        let result = controller
            .train_batch(
                Array3F::ones((64, 28, 28)).into_dyn(),
                &Array2F::ones((64, 10)).into_dyn(),
            )
            .unwrap();
        println!("{}", result);
    }
}
