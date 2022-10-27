use std::env;

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

pub fn get_model_config_url(name: &str) -> String {
    format!("models/{}/model.xml", name)
}

pub fn get_model_url(name: &str, version: u32) -> String {
    format!("models/{}/{}.model", name, version)
}

pub fn get_train_batch_url(name: &str, num: u32) -> String {
    format!("static/{}/train/train.{}.dat", name, num)
}

pub fn get_test_batch_url(name: &str, num: u32) -> String {
    format!("static/{}/test/test.{}.dat", name, num)
}
