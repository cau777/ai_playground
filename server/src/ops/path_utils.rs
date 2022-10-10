#[cfg(any(debug_assertions, test))]
const BASE_PATH: &'static str = "./temp";

#[cfg(all(not(debug_assertions), not(test)))]
const BASE_PATH: &'static str = "/app/files";

pub fn get_model_path(name: &str) -> String {
    format!("{}/{}", BASE_PATH, name)
}

pub fn get_model_version_path(name: &str, version: u32) -> String {
    format!("{}/{}.model", get_model_path(name), version)
}

pub fn get_model_version_meta_path(name: &str, version: u32) -> String {
    format!("{}/{}.json", get_model_path(name), version)
}

pub fn get_model_config_path(name: &str) -> String {
    format!("{}/config.json", get_model_path(name))
}

pub fn get_model_best_path(name: &str) -> String {
    format!("{}/best.txt", get_model_path(name))
}