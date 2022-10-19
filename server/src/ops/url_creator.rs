use std::env::var;

const AZURE_FS_PREFIX: &'static str = "azure-fs";
const LOCAL_SERVER_PREFIX: &'static str = "local";

pub struct UrlCreator {
    is_local_server: bool,
    azure_fs_url: String,
    azure_fs_query: String,
    name: &'static str,
}

impl UrlCreator {
    pub fn new(name: &'static str) -> Self {
        Self {
            is_local_server: var(&"IS_LOCAL_SERVER")
                .map(|o| o.to_ascii_uppercase() == "TRUE")
                .unwrap_or(false),
            azure_fs_url: var(&"AZURE_FS_URL")
                .expect("AZURE_FS_URL must be defined as an env variable"),
            azure_fs_query: var(&"AZURE_FS_QUERY_SAS")
                .expect("AZURE_FS_QUERY_SAS must be defined as an env variable"),
            name,
        }
    }

    pub fn gen_model_config(&self) -> String {
        if self.is_local_server {
            format!("{}|{}/model.xml", LOCAL_SERVER_PREFIX, self.name)
        } else {
            format!(
                "{}|{}/models/{}/model.xml?{}",
                AZURE_FS_PREFIX, self.azure_fs_url, self.name, self.azure_fs_query
            )
        }
    }

    pub fn gen_model_data(&self, version: u32) -> String {
        if self.is_local_server {
            format!("{}|{}/{}.model", LOCAL_SERVER_PREFIX, self.name, version)
        } else {
            format!(
                "{}|{}/models/{}/{}.model?{}",
                AZURE_FS_PREFIX, self.azure_fs_url, self.name, version, self.azure_fs_query
            )
        }
    }

    pub fn get_train_batch(&self, batch: u32) -> String {
        format!(
            "{}|{}/static/{}/train/train.{}.dat?{}",
            AZURE_FS_PREFIX, self.azure_fs_url, self.name, batch, self.azure_fs_query
        )
    }

    pub fn get_test_batch(&self, batch: u32) -> String {
        format!(
            "{}|{}/static/{}/test/test.{}.dat?{}",
            AZURE_FS_PREFIX, self.azure_fs_url, self.name, batch, self.azure_fs_query
        )
    }
}
