mod data;
mod rest_handlers;
pub mod utils;
mod ws_handlers;

use std::{convert::Infallible, sync::Arc};

use data::{
    clients::Clients, current_model::CurrentModel, env_config::EnvConfig,
    model_source::ModelSource, task_manager::TaskManager,
};
use tokio::{self, sync::RwLock};
use warp::{self, Filter};

pub type TaskManagerDep = Arc<RwLock<TaskManager>>;
pub type ModelsSourcesDep = Arc<RwLock<ModelSource>>;
pub type CurrentModelDep = Arc<RwLock<CurrentModel>>;
pub type ClientsDep = Arc<RwLock<Clients>>;
pub type EnvConfigDep = Arc<EnvConfig>;

#[tokio::main]
async fn main() {
    let config = Arc::new(EnvConfig::new());

    let mut models_sources = ModelSource::new("digits", 938, 40).unwrap(); // TODO 938, 40
    let task_manager = TaskManager::new(&models_sources, config.clone());
    let current_model = CurrentModel::new(&mut models_sources);
    let clients = Clients::new();

    let models_sources_dep = Arc::new(RwLock::new(models_sources));
    let task_manager_dep = Arc::new(RwLock::new(task_manager));
    let current_model_dep = Arc::new(RwLock::new(current_model));
    let clients_dep = Arc::new(RwLock::new(clients));

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(["GET", "POST", "OPTIONS", "DELETE"])
        .allow_header("content-type")
        .build();

    let assign_route = warp::path!("assign")
        .and(warp::get())
        .and(with_task_manager(&task_manager_dep))
        .and(with_models_sources(&models_sources_dep))
        .and_then(rest_handlers::assign_handler);

    let submit_train_route = warp::path!("submit_train")
        .and(warp::post())
        .and(with_task_manager(&task_manager_dep))
        .and(with_models_sources(&models_sources_dep))
        .and(with_current_model(&current_model_dep))
        .and(warp::body::bytes())
        .and_then(rest_handlers::submit_train);

    let submit_test_route = warp::path!("submit_test")
        .and(warp::post())
        .and(with_task_manager(&task_manager_dep))
        .and(with_models_sources(&models_sources_dep))
        .and(warp::body::json::<rest_handlers::TaskTestResult>())
        .and_then(rest_handlers::submit_test);

    let recent_route = warp::path!("recent")
        .and(warp::get())
        .and(with_current_model(&current_model_dep))
        .and_then(rest_handlers::recent);

    let register_route = warp::path!("register")
        .and(warp::get())
        .and_then(rest_handlers::register);

    let best_route = warp::path!("best")
        .and(warp::get())
        .and(with_models_sources(&models_sources_dep))
        .and(with_env_config(&config))
        .and_then(rest_handlers::get_best);

    let config_route = warp::path!("config")
        .and(warp::get())
        .and(with_env_config(&config))
        .and_then(rest_handlers::get_config);

    let routes = submit_train_route
        .or(assign_route)
        .or(submit_test_route)
        .or(recent_route)
        .or(register_route)
        .or(best_route)
        .or(config_route)
        .with(cors);

    warp::serve(routes).run(([0, 0, 0, 0], 8000)).await;
}

macro_rules! dep_filter {
    ($x:ty) => {
        impl Filter<Extract = ($x,), Error = Infallible> + Clone
    };
}

fn with_task_manager(instance: &TaskManagerDep) -> dep_filter![TaskManagerDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_models_sources(instance: &ModelsSourcesDep) -> dep_filter![ModelsSourcesDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_current_model(instance: &CurrentModelDep) -> dep_filter![CurrentModelDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_clients(instance: &ClientsDep) -> dep_filter![ClientsDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}

fn with_env_config(instance: &EnvConfigDep) -> dep_filter![EnvConfigDep] {
    let instance = instance.clone();
    warp::any().map(move || instance.clone())
}
