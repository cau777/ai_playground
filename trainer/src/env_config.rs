use std::env::{var, VarError};

pub struct EnvConfig {
    pub versions_server_url: String,
    pub max_epochs: u32,
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
        Ok(Self {
            versions_server_url: get_path("VERSIONS_SERVER_URL")?,
            max_epochs: var("MAX_EPOCHS").ok().and_then(|o| o.parse().ok()).unwrap_or(10), // Optional
            mounted_path: get_path("MOUNTED_PATH")?,
        })
    }
}