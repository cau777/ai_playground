use std::env;
/*
#[cfg(any(debug_assertions, test))]
const BASE_PATH: &'static str = "./temp";

#[cfg(all(not(debug_assertions), not(test)))]
const BASE_PATH: &'static str = "/app/files";
*/
pub fn get_model_path(name: &str) -> String {
    format!("{}/{}", env::var("MODEL_FILE_PATH").unwrap_or_else(|_| "./temp".to_owned()), name)
}

pub fn get_model_version_path(name: &str, version: u32) -> String {
    format!("{}/{}.model", get_model_path(name), version)
}

pub fn get_model_version_meta_path(name: &str, version: u32) -> String {
    format!("{}/{}.json", get_model_path(name), version)
}

pub fn get_model_config_path(name: &str) -> String {
    format!("{}/model.xml", get_model_path(name))
}

pub fn get_model_url(name: &str, version: u32) -> String {
    format!("models/{}/{}.model", name, version)
}