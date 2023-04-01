use std::{fs, io};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::io::ErrorKind::InvalidData;
use codebase::integration::deserialization::deserialize_storage;
use codebase::integration::layers_loading::{load_model_xml, ModelXmlConfig};
use serde::{Serialize, Deserialize};
use codebase::integration::serialization::serialize_storage;
use codebase::nn::layers::nn_layers::GenericStorage;
use crate::EnvConfigDep;

struct Version {
    id: u32,
    meta: VersionMeta,
}

#[derive(Serialize, Deserialize)]
struct VersionMeta {
    loss: f64,
}

pub struct FileManager {
    versions: Vec<Version>,
    base_path: String,
    config: EnvConfigDep,
}

impl FileManager {
    pub fn new(name: &str, config: EnvConfigDep) -> io::Result<Self> {
        let base_path = format!("{}/{}", config.base_path, name);
        let _ = fs::create_dir_all(&base_path); // Create base directory if not present

        let mut versions = Vec::new();
        for entry in fs::read_dir(&base_path)?
            .filter_map(|o| o.ok()) {
            let name = entry.file_name();
            let name = name.to_str();
            if let Some(name) = name {
                if !name.ends_with(".json") { continue; }
                let id = name.strip_suffix(".json").unwrap();

                let file = OpenOptions::new()
                    .read(true)
                    .open(entry.path())?;
                let meta: VersionMeta = serde_json::from_reader(file)?;

                versions.push(Version {
                    id: id.parse().unwrap_or(0),
                    meta,
                });
            }
        }

        Ok(Self {
            versions,
            base_path,
            config
        })
    }

    pub fn add(&mut self, storage: &GenericStorage, loss: f64) -> io::Result<()> {
        let new_id = self.most_recent() + 1;

        // Remove out-dated versions
        while self.versions.len() >= self.config.keep_versions {
            let worst = self.versions.iter()
                .max_by(|a, b| a.meta.loss.total_cmp(&b.meta.loss)).unwrap();
            self.remove(worst.id)?;
        }

        let bytes = serialize_storage(storage);
        println!("1 {}", bytes.len());
        let mut file = self.open_storage(new_id, OpenOptions::new().create_new(true).write(true))?;
        file.write_all(&bytes)?;
        println!("2");

        let meta = VersionMeta { loss };
        let file = self.open_meta(new_id, OpenOptions::new().create_new(true).write(true))?;
        serde_json::to_writer(file, &meta)?;

        self.versions.push(Version { meta, id: new_id });
        Ok(())
    }

    pub fn get_storage(&self, id: u32) -> io::Result<GenericStorage> {
        let mut file = self.open_storage(id, OpenOptions::new().read(true))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        deserialize_storage(&buffer)
            .map_err(|e| io::Error::new(InvalidData, e))
    }

    pub fn get_config(&self) -> io::Result<ModelXmlConfig> {
        self.get_config_bytes().and_then(|o| load_model_xml(&o)
            .map_err(|e| io::Error::new(InvalidData, e)))
    }

    pub fn get_config_bytes(&self) -> io::Result<Vec<u8>> {
        let mut file = self.open_config(OpenOptions::new().read(true))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    pub fn set_config_bytes(&self, bytes: &[u8]) -> io::Result<()> {
        let mut file = self.open_config(OpenOptions::new().write(true).create(true))?;
        file.write_all(bytes)
    }

    /// Find the most recent version
    pub fn most_recent(&self) -> u32 {
        self.versions.iter().map(|o| o.id).max().unwrap_or(0)
    }

    /// Find the version with the lowest loss
    pub fn best(&self) -> u32 {
        self.versions.iter().min_by(|a, b| a.meta.loss.total_cmp(&b.meta.loss))
            .map(|o| o.id).unwrap_or(0)
    }

    fn remove(&mut self, id: u32) -> io::Result<()> {
        fs::remove_file(format!("{}/{}.model", self.base_path, id))?;
        fs::remove_file(format!("{}/{}.json", self.base_path, id))?;
        let index = self.versions.iter().position(|o| o.id == id).unwrap();
        self.versions.swap_remove(index);
        Ok(())
    }

    fn open_storage(&self, id: u32, options: &mut OpenOptions) -> io::Result<File> {
        options.open(format!("{}/{}.model", self.base_path, id))
    }

    fn open_meta(&self, id: u32, options: &mut OpenOptions) -> io::Result<File> {
        options.open(format!("{}/{}.json", self.base_path, id))
    }

    fn open_config(&self, options: &mut OpenOptions) -> io::Result<File> {
        options.open(format!("{}/config.xml", self.base_path))
    }
}
