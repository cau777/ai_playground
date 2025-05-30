extern crate core;

mod rest_handlers;
mod env_config;
mod file_manager;
mod loaded_model;
mod utils;
mod endpoint_dict;
mod digits_handler;
mod chess;

use std::sync::Arc;
use tokio::sync::RwLock;
use warp::{Filter, path};
use warp::http::StatusCode;
use crate::chess::chess_games_pool::ChessGamesPool;
use crate::endpoint_dict::EndpointDict;
use crate::env_config::EnvConfig;
use crate::file_manager::FileManager;
use crate::loaded_model::LoadedModel;

pub type EnvConfigDep = Arc<EnvConfig>;
pub type FileManagerDep = Arc<RwLock<FileManager>>;
pub type LoadedModelDep = Arc<RwLock<LoadedModel>>;
pub type ChessGamesPoolDep = Arc<RwLock<ChessGamesPool>>;

pub type FileManagersDep = EndpointDict<Arc<RwLock<FileManager>>>;
pub type LoadedModelsDep = EndpointDict<Arc<RwLock<LoadedModel>>>;

#[tokio::main]
async fn main() {
    println!("Running");
    let config = Arc::new(EnvConfig::new());
    // There's one instance of the FileManager and one for the LoadedModel for each model name
    let file_managers = EndpointDict::new(
        Arc::new(RwLock::new(FileManager::new("digits", config.clone()).unwrap())),
        Arc::new(RwLock::new(FileManager::new("chess", config.clone()).unwrap())),
    );
    let loaded_models = EndpointDict::new(
        Arc::new(RwLock::new(LoadedModel::new())),
        Arc::new(RwLock::new(LoadedModel::new())),
    );
    let chess_games_pool = Arc::new(RwLock::new(ChessGamesPool::new(&config)));
    println!("Finished loading");

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(["GET", "POST", "OPTIONS", "DELETE"])
        .allow_header("content-type")
        .build();

    // Generic routes
    let wake_up_route = path!("wakeup")
        .and(warp::get())
        .map(warp::reply);

    let post_trainable_route = path!(String / "trainable")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(with_file_managers(&file_managers))
        .and_then(rest_handlers::post_trainable);
        // .recover(|err| {
        // 
        // });

    let get_trainable_route = path!(String / "trainable")
        .and(warp::get())
        .and(with_file_managers(&file_managers))
        .and_then(rest_handlers::get_trainable);

    let get_config_route = path!(String / "config")
        .and(warp::get())
        .and(with_file_managers(&file_managers))
        .and_then(rest_handlers::get_config);

    let set_config_route = path!(String / "config")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(with_file_managers(&file_managers))
        .and_then(rest_handlers::post_config);

    // Digits routes
    let digits_eval_route = path!("digits" / "eval")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_file_manager(&file_managers.digits))
        .and(with_loaded_model(&loaded_models.digits))
        .and_then(digits_handler::post_eval);

    // Chess routes
    let chess_start_route = path!("chess" / "start")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_file_manager(&file_managers.chess))
        .and(with_loaded_model(&loaded_models.chess))
        .and(with_chess_games_pool(&chess_games_pool))
        .and(with_env_config(&config))
        .and_then(chess::chess_handlers::post_start);

    let chess_move_route = path!("chess" / "move")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_file_manager(&file_managers.chess))
        .and(with_loaded_model(&loaded_models.chess))
        .and(with_chess_games_pool(&chess_games_pool))
        .and(with_env_config(&config))
        .and_then(chess::chess_handlers::post_move);

    let routes = wake_up_route
        .or(post_trainable_route)
        .or(get_trainable_route)
        .or(get_config_route)
        .or(set_config_route)

        .or(digits_eval_route)

        .or(chess_start_route)
        .or(chess_move_route)

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

fn with_file_managers(instance: &FileManagersDep) -> dep_filter![FileManagersDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_loaded_models(instance: &LoadedModelsDep) -> dep_filter![LoadedModelsDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_chess_games_pool(instance: &ChessGamesPoolDep) -> dep_filter![ChessGamesPoolDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}