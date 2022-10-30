use std::env::var;

pub struct EnvConfig {
    pub is_local_server: bool,
    pub azure_fs_url: String,
    pub azure_fs_query: String,
}

impl EnvConfig {
    pub fn new() -> Self {
        Self {
            is_local_server: var(&"IS_LOCAL_SERVER")
                .map(|o| o.to_ascii_uppercase() == "TRUE")
                .unwrap_or(false),
            azure_fs_url: var(&"AZURE_FS_URL")
                .expect("AZURE_FS_URL must be defined as an env variable"),
            azure_fs_query: var(&"AZURE_FS_QUERY_SAS")
                .expect("AZURE_FS_QUERY_SAS must be defined as an env variable"),
        }
    }
}
