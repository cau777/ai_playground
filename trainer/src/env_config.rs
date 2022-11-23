use std::env::{var, VarError};

pub struct EnvConfig {
    pub versions_server_url: String,
    pub versions: u32,
    pub epochs_per_version: u32,
    pub mounted_path: String,
}

fn get_path(name: &str) -> Result<String, VarError> {
    let value = var(name)?;
    match value.strip_suffix('/') {
        Some(value) => Ok(value.to_owned()),
        None => Ok(value)
    }
}


impl EnvConfig {
    pub fn new() -> Self {
        Self::try_new().unwrap()
    }

    fn try_new() -> Result<Self, VarError> {
        let epochs_per_version = var("EPOCHS_PER_VERSION").unwrap_or_else(|_| "3".to_owned()).parse().unwrap();
        let versions = var("VERSIONS").unwrap_or_else(|_| "10".to_owned()).parse().unwrap();

        Ok(Self {
            versions_server_url: get_path("VERSIONS_SERVER_URL")?,
            mounted_path: get_path("MOUNTED_PATH")?,
            epochs_per_version,
            versions
        })
    }
}