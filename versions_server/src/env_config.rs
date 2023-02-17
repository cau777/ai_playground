use std::env::var;

#[derive(Debug)]
pub struct EnvConfig {
    pub base_path: String,
    pub host_address: [u8; 4],
    pub port: u16,
    pub keep_versions: usize,
    pub eval_delta_exp: f64,
    pub depth_delta_exp: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvConfig {
    pub fn new() -> Self {
        let base_path = var("MOUNTED_PATH").unwrap_or_else(|_| "../temp".to_owned());
        let host_address = var("HOST_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_owned());
        let host_address: Vec<u8> = host_address.split('.').map(|o| o.parse().unwrap()).collect();
        let port = var("PORT").unwrap_or_else(|_| "8000".to_owned()).parse().unwrap();
        let keep_versions = var("KEEP_VERSIONS").unwrap_or_else(|_| "30".to_owned()).parse().unwrap();
        let eval_delta_exp = var("NEXT_NODE_EVAL_DELTA_EXP").unwrap_or_else(|_| "5".to_owned()).parse().unwrap();
        let depth_delta_exp = var("NEXT_NODE_DEPTH_DELTA_EXP").unwrap_or_else(|_| "0.1".to_owned()).parse().unwrap();

        let result = Self {
            base_path,
            host_address: host_address.try_into().unwrap(),
            port,
            keep_versions,
            eval_delta_exp,
            depth_delta_exp,
        };
        println!("{:?}", result);
        result
    }
}
