extern crate core;

mod rest_handlers;
mod env_config;
mod file_manager;
mod loaded_model;

use std::sync::Arc;
use tokio::sync::RwLock;
use warp::{Filter, path};
use crate::env_config::EnvConfig;
use crate::file_manager::FileManager;
use crate::loaded_model::LoadedModel;

pub type EnvConfigDep = Arc<EnvConfig>;
pub type FileManagerDep = Arc<RwLock<FileManager>>;
pub type LoadedModelDep = Arc<RwLock<LoadedModel>>;

#[tokio::main]
async fn main() {
    println!("Running");
    let config = Arc::new(EnvConfig::new());
    let file_manager = FileManager::new("digits", config.clone()).unwrap();
    let file_manager = Arc::new(RwLock::new(file_manager));
    let loaded_model = LoadedModel::new();
    let loaded_model = Arc::new(RwLock::new(loaded_model));

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

    let post_eval_route = path!("eval")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_file_manager(&file_manager))
        .and(with_loaded_model(&loaded_model))
        .and_then(rest_handlers::post_eval);

    let routes = post_trainable_route
        .or(get_trainable_route)
        .or(get_config_route)
        .or(set_config_route)
        .or(post_eval_route)
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

fn with_loaded_model(instance: &LoadedModelDep) -> dep_filter![LoadedModelDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}