use crate::data::model_metadata::ModelMetadata;
use crate::data::path_utils;
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::layers_loading::{load_model_xml, ModelXmlConfig};
use codebase::integration::proto_loading::save_model_bytes;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
use std::path::Path;
use std::{fs, io};

fn invalid_data_err(e: impl Into<Box<dyn Error + Send + Sync>>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, e)
}

pub struct ModelSource {
    name: &'static str,
    config: ModelXmlConfig,
    best: Option<u32>,
    most_recent: Option<u32>,
    train_count: u32,
    test_count: u32,
}

impl ModelSource {
    pub fn new(name: &'static str, train_count: u32, test_count: u32) -> io::Result<Self> {
        let config = Self::load_config(&name).map_err(invalid_data_err)?;

        let mut result = Self {
            name,
            config: config.clone(),
            best: None,
            most_recent: None,
            train_count,
            test_count,
        };

        let versions = Self::load_versions(&name);
        println!("Versions {:?}", versions);

        if versions.is_empty() {
            result.create_empty();
        }

        Ok(result)
    }

    pub fn latest(&mut self) -> u32 {
        if self.most_recent.is_none() {
            let versions = Self::load_versions(self.name);
            self.most_recent = versions.into_iter().max();
        }
        self.most_recent.unwrap()
    }

    pub fn best(&mut self) -> u32 {
        if self.best.is_none() {
            let versions = Self::load_versions(self.name);
            let mut best_accuracy = f64::MIN;
            let mut best = 0;
            for version in versions {
                let meta = Self::load_model_meta(self.name, version);
                match meta {
                    Ok(meta) => {
                        if meta.accuracy > best_accuracy {
                            best_accuracy = meta.accuracy;
                            best = version;
                        }
                    }
                    _ => {}
                }
            }
            self.best = Some(best);
        }
        self.best.unwrap()
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn train_count(&self) -> u32 {
        self.train_count
    }
    
    pub fn test_count(&self) -> u32 {
        self.test_count
    }
    
    pub fn versions_to_test(&self) -> Vec<u32> {
        let mut versions = Self::load_versions(self.name);
        versions.retain(|o| !Self::check_version_tested(self.name, *o));
        versions
    }

    pub fn load_model_bytes(&self, version: u32) -> io::Result<Vec<u8>> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path_utils::get_model_version_path(self.name, version))?;
        let mut result = Vec::new();
        file.read_to_end(&mut result)?;
        decompress_default(&result).map_err(invalid_data_err)
    }

    pub fn clear_all(&mut self) -> io::Result<()> {
        self.config = Self::load_config(self.name).map_err(invalid_data_err)?;
        self.most_recent = Some(0);
        self.best = Some(0);
        let versions = Self::load_versions(self.name);
        for v in versions {
            let _ = self.delete_model(v);
        }
        self.create_empty();
        Ok(())
    }

    fn create_empty(&mut self) {
        let controller = NNController::new(
            self.config.main_layer.clone(),
            self.config.loss_func.clone(),
        )
        .unwrap();
        let params = controller.export();
        self.save_model(0, &params).unwrap();
    }

    fn load_config(name: &str) -> io::Result<ModelXmlConfig> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path_utils::get_model_config_path(name))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        load_model_xml(&bytes).map_err(invalid_data_err)
    }

    pub fn save_model(&mut self, version: u32, data: &GenericStorage) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path_utils::get_model_version_path(self.name, version))?;
        let bytes = save_model_bytes(data).map_err(invalid_data_err)?;
        let bytes = compress_default(&bytes)?;
        file.write_all(&bytes)?;
        self.most_recent = Some(self.most_recent.unwrap_or(0).max(version));
        Ok(())
    }

    fn delete_model(&self, version: u32) -> io::Result<()> {
        fs::remove_file(path_utils::get_model_version_path(self.name, version))?;
        fs::remove_file(path_utils::get_model_version_meta_path(self.name, version))?;
        Ok(())
    }

    fn load_model_meta(name: &str, version: u32) -> io::Result<ModelMetadata> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path_utils::get_model_version_meta_path(name, version))?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        serde_json::from_str(&content).map_err(invalid_data_err)
    }

    pub fn save_model_meta(&mut self, version: u32, meta: &ModelMetadata) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path_utils::get_model_version_meta_path(&self.name, version))?;
        let content = serde_json::to_string(meta).map_err(invalid_data_err)?;
        file.write_all(content.as_bytes())?;

        if self.best.is_none() || Self::load_model_meta(&self.name, self.best.unwrap())?.accuracy < meta.accuracy
        {
            self.best = Some(version);
        }
        Ok(())
    }

    fn load_versions(name: &str) -> Vec<u32> {
        let paths = fs::read_dir(path_utils::get_model_path(name)).unwrap();
        let mut result = Vec::new();

        for path in paths.into_iter().filter_map(|o| o.ok()) {
            let name = path.file_name();
            let name = name.to_str();
            if name.is_none() {
                continue;
            }
            let name = name.unwrap();

            if !name.ends_with(".model") {
                continue;
            }
            let name = name.strip_suffix(".model");
            if name.is_none() {
                continue;
            }

            let version = name.unwrap().parse();
            if version.is_err() {
                continue;
            }
            result.push(version.unwrap());
        }

        result
    }

    fn check_version_tested(name: &str, version: u32) -> bool {
        let path = path_utils::get_model_version_meta_path(name, version);
        Path::new(&path).exists()
    }
}
