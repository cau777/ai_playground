mod rest_handlers;
mod env_config;
mod file_manager;

use std::sync::Arc;
use tokio::sync::RwLock;
use warp::{Filter, path};
use crate::env_config::EnvConfig;
use crate::file_manager::FileManager;

pub type EnvConfigDep = Arc<EnvConfig>;
pub type FileManagerDep = Arc<RwLock<FileManager>>;

#[tokio::main]
async fn main() {
    println!("Running");
    let config = Arc::new(EnvConfig::new());
    let file_manager = FileManager::new("digits", config.clone()).unwrap();
    let file_manager = Arc::new(RwLock::new(file_manager));

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(["GET", "POST", "OPTIONS", "DELETE"])
        .allow_header("content-type")
        .build();

    let post_trainable_route = path!("trainable")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(with_file_manager(&file_manager))
        .and_then(rest_handlers::post_trainable);

    let get_trainable_route = path!("trainable")
        .and(warp::get())
        .and(with_file_manager(&file_manager))
        .and_then(rest_handlers::get_trainable);
    
    let get_config_route = path!("config")
        .and(warp::get())
        .and(with_file_manager(&file_manager))
        .and_then(rest_handlers::get_config);
    
    let set_config_route = path!("config")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(with_file_manager(&file_manager))
        .and_then(rest_handlers::post_config);

    let routes = post_trainable_route
        .or(get_trainable_route)
        .or(get_config_route)
        .or(set_config_route)
        .with(cors);

    warp::serve(routes).run((config.host_address, config.port)).await;
}

macro_rules! dep_filter {
    ($x:ty) => {
        impl Filter<Extract = ($x,), Error = std::convert::Infallible> + Clone
    };
}

fn with_env_config(instance: &EnvConfigDep) -> dep_filter![EnvConfigDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_file_manager(instance: &FileManagerDep) -> dep_filter![FileManagerDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}