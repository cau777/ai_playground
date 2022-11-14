use std::error::Error;
use std::ffi::c_double;
use std::fmt::{Display, Formatter};
use codebase::integration::layers_loading::{load_model_xml, ModelXmlConfig};
use codebase::integration::serialization::serialize_version;
use codebase::nn::layers::nn_layers::GenericStorage;
use http::StatusCode;
use reqwest::blocking::*;
use crate::EnvConfig;

pub struct ServerClient {
    client: Client,
    base_url: String,
}

impl ServerClient {
    pub fn new(config: &EnvConfig) -> Self {
        Self {
            client: Client::new(),
            base_url: config.versions_server_url.clone(),
        }
    }

    pub fn submit(&self, storage: &GenericStorage, loss: f64) {
        println!("Uploading");
        let bytes = serialize_version(storage, loss);

        let response = self.client.post(self.create_url("trainable"))
            .body(bytes)
            .send().unwrap();
        if response.status() != StatusCode::OK {
            panic!("Invalid response from server {:?}", response)
        }
    }

    pub fn load_model_config(&self) -> Result<ModelXmlConfig, Box<dyn Error>> {
        let model_config_response = self.client.get(self.create_url("config")).send()?;
        model_config_response.error_for_status_ref()?;
        Ok(load_model_xml(&model_config_response.bytes()?)?)
    }

    pub fn get(&self, path: &str) -> reqwest::Result<Response> {
        self.client.get(self.create_url(path)).send()
    }

    fn create_url(&self, path: &str) -> String {
        println!("{}/{}", self.base_url, path);
        format!("{}/{}", self.base_url, path)
    }
}