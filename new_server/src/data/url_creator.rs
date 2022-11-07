use super::env_config::EnvConfig;

const AZURE_FS_PREFIX: &'static str = "azure-fs";
const LOCAL_SERVER_PREFIX: &'static str = "local";

pub fn get_model_config(config: &EnvConfig, name: &str) -> String {
    if config.is_local_server {
        format!("{}|{}/model.xml", LOCAL_SERVER_PREFIX, name)
    } else {
        format!(
            "{}|{}/models/{}/model.xml?{}",
            AZURE_FS_PREFIX, config.azure_fs_url, name, config.azure_fs_query
        )
    }
}

pub fn get_model_data(config: &EnvConfig, name: &str, version: u32) -> String {
    if config.is_local_server {
        format!("{}|{}/{}.model", LOCAL_SERVER_PREFIX, name, version)
    } else {
        format!(
            "{}|{}/models/{}/{}.model?{}",
            AZURE_FS_PREFIX, config.azure_fs_url, name, version, config.azure_fs_query
        )
    }
}

pub fn get_train_batch(config: &EnvConfig, name: &str, batch: u32) -> String {
    format!(
        "{}|{}/static/{}/train/train.{}.dat?{}",
        AZURE_FS_PREFIX, config.azure_fs_url, name, batch, config.azure_fs_query
    )
}

pub fn get_validate_batch(config: &EnvConfig, name: &str, batch: u32) -> String {
    format!(
        "{}|{}/static/{}/validate/validate.{}.dat?{}",
        AZURE_FS_PREFIX, config.azure_fs_url, name, batch, config.azure_fs_query
    )
}
