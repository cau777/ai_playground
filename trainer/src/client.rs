use std::error::Error;
use std::ffi::c_double;
use std::fmt::{Display, Formatter};
use std::time::Duration;
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
            client: Client::builder()
                .danger_accept_invalid_certs(true)
                .connect_timeout(Duration::new(15, 0))
                .timeout(Duration::new(15, 0))
                .build()
                .unwrap(),
            base_url: config.versions_server_url.clone(),
        }
    }

    pub fn submit(&self, storage: &GenericStorage, loss: f64, name: &str) {
        println!("Uploading");
        let bytes = serialize_version(storage, loss);

        let response = self.client.post(self.create_url_with_name(name, "trainable"))
            .body(bytes)
            .send().unwrap();
        if response.status() != StatusCode::OK {
            panic!("Invalid response from server {:?}", response)
        }
    }

    pub fn load_model_config(&self, name: &str) -> Result<ModelXmlConfig, Box<dyn Error>> {
        let response = self.client.get(self.create_url_with_name(name, "config")).send()?;
        response.error_for_status_ref()?;
        Ok(load_model_xml(&response.bytes()?)?)
    }

    pub fn get_trainable(&self, name: &str) -> reqwest::Result<Response> {
        self.client.get(self.create_url_with_name(name, "trainable")).send()
    }

    fn create_url(&self, path: &str) -> String {
        println!("{}/{}", self.base_url, path);
        format!("{}/{}", self.base_url, path)
    }

    fn create_url_with_name(&self,name: &str,  path: &str) -> String {
        println!("{}/{}/{}", self.base_url, name, path);
        format!("{}/{}/{}", self.base_url, name, path)
    }
}