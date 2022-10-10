use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
use codebase::integration::proto_loading::save_model_bytes;
use codebase::nn::controller::NNController;
use codebase::nn::layers::dense_layer::{DenseLayerConfig, DenseLayerInit};
use codebase::nn::layers::nn_layers::{GenericStorage, Layer};
use codebase::nn::layers::sequential_layer::SequentialLayerConfig;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use std::{io, fs};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, RwLock};
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::nn::loss::loss_func::LossFunc;
use rocket::State;
use crate::ops::lazy::Lazy;
use crate::ops::model_metadata::ModelMetadata;
use crate::ops::path_utils::{get_model_best_path, get_model_config_path, get_model_path, get_model_version_meta_path, get_model_version_path};

pub type ModelSourcesState = State<RwLock<ModelSources>>;

pub struct ModelSources {
    pub testy: Lazy<ModelSource>,
}

impl ModelSources {
    pub fn new() -> Self {
        Self {
            testy: Lazy::new(|| ModelSource::new("testy").unwrap())
        }
    }
}

fn invalid_data_err(e: impl Into<Box<dyn Error + Send + Sync>>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, e)
}

pub struct ModelSource {
    name: &'static str,
    config: Layer,
    best: Option<u32>,
    to_test: Vec<u32>,
}

impl ModelSource {
    pub fn new(name: &'static str) -> io::Result<Self> {
        let config = Self::load_config(&name).unwrap();

        let mut result = Self {
            name,
            config: config.clone(),
            best: None,
            to_test: Vec::new(),
        };

        let mut versions = Self::load_versions(&name);
        println!("Versions {:?}", versions);

        if versions.is_empty() {
            let controller = NNController::new(config, LossFunc::Mse).unwrap();
            let params = controller.export();
            Self::save_model(&name, 0, &params).unwrap();
        } else {
            versions.retain(|o| !Self::check_version_tested(&name, *o));
        }

        Ok(result)
    }

    pub fn most_recent(&self) -> u32 {
        let versions = Self::load_versions(self.name);
        versions.into_iter().max().unwrap_or(0)
    }

    pub fn load_model_bytes(&self, version: u32) -> io::Result<Vec<u8>> {
        let mut file = OpenOptions::new().read(true).open(get_model_version_path(self.name, version))?;
        let mut result = Vec::new();
        file.read_to_end(&mut result)?;
        decompress_default(&result).map_err(invalid_data_err)
    }

    fn load_config(name: &str) -> io::Result<Layer> {
        let mut file = OpenOptions::new().read(true).open(get_model_config_path(name))?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(Layer::Sequential(SequentialLayerConfig {
            layers: vec![
                Layer::Dense(DenseLayerConfig {
                    in_values: 10,
                    out_values: 15,
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                    init_mode: DenseLayerInit::Random(),
                }),
                Layer::Relu,
                Layer::Dense(DenseLayerConfig {
                    in_values: 15,
                    out_values: 10,
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig { lr: 0.05 }),
                    init_mode: DenseLayerInit::Random(),
                }),
            ]
        }))
    }

    fn load_best(&self) -> Option<u32> {
        let mut file = OpenOptions::new().read(true).open(get_model_best_path(self.name)).ok()?;
        let mut content = String::new();
        file.read_to_string(&mut content).ok()?;
        let version: u32 = content.parse().ok()?;
        Some(version)
    }

    fn save_best(name: &str, version: u32) -> io::Result<()> {
        let mut file = OpenOptions::new().write(true).open(get_model_best_path(name))?;
        file.write_all(version.to_string().as_bytes())
    }

    fn save_model(name: &str, version: u32, data: &GenericStorage) -> io::Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(get_model_version_path(name, version))?;
        let bytes = save_model_bytes(data).map_err(invalid_data_err)?;
        let bytes = compress_default(&bytes)?;
        // TODO: test
        file.write_all(&bytes)?;

        Ok(())
    }

    fn load_model_meta(name: &str, version: u32) -> io::Result<ModelMetadata> {
        let mut file = OpenOptions::new().read(true).open(get_model_version_meta_path(name, version))?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        rocket::serde::json::from_str(&content).map_err(invalid_data_err)
    }

    fn save_model_meta(&mut self, version: u32, meta: &ModelMetadata) -> io::Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(get_model_version_meta_path(&self.name, version))?;
        let content = rocket::serde::json::to_string(meta).map_err(invalid_data_err)?;
        file.write_all(content.as_bytes())?;

        let index = self.to_test.iter().position(|o| *o == version).unwrap();
        self.to_test.remove(index);

        if self.best.is_none() || Self::load_model_meta(&self.name, self.best.unwrap())?.accuracy < meta.accuracy {
            self.best = Some(version);
        }
        Ok(())
    }

    fn load_versions(name: &str) -> Vec<u32> {
        let paths = fs::read_dir(get_model_path(name)).unwrap();
        let mut result = Vec::new();

        for path in paths.into_iter().filter_map(|o| o.ok()) {
            let name = path.file_name();
            let name = name.to_str();
            if name.is_none() { continue; }
            let name = name.unwrap();

            if !name.ends_with(".model") { continue; }
            let name = name.strip_suffix(".model");
            if name.is_none() { continue; }

            let version = name.unwrap().parse();
            if version.is_err() { continue; }
            result.push(version.unwrap());
        }

        result
    }

    fn check_version_tested(name: &str, version: u32) -> bool {
        let path = get_model_version_meta_path(name, version);
        Path::new(&path).exists()
    }
}

#[cfg(test)]
mod tests {
    use crate::ModelSources;

    #[test]
    fn test_test() {
        let mut source = ModelSources::new("testy".to_owned()).unwrap();
    }
}