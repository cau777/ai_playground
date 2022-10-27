mod data;
mod rest_handlers;
pub mod utils;
mod ws_handlers;

use std::{convert::Infallible, sync::Arc};

use data::{current_model::CurrentModel, model_source::ModelSource, task_manager::TaskManager, clients::Clients};
use tokio::{self, sync::RwLock};
use warp::{self, Filter};

pub type TaskManagerDep = Arc<RwLock<TaskManager>>;
pub type ModelsSourcesDep = Arc<RwLock<ModelSource>>;
pub type CurrentModelDep = Arc<RwLock<CurrentModel>>;
pub type ClientsDep = Arc<RwLock<Clients>>;

#[tokio::main]
async fn main() {
    let mut models_sources = ModelSource::new("digits", 469, 79).unwrap(); // TODO 469 79
    let task_manager = TaskManager::new(&models_sources);
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

    let ws_train_route = warp::path!("ws_train")
        .and(warp::ws())
        .and(with_current_model(&current_model_dep))
        .and(with_models_sources(&models_sources_dep))
        .and(with_task_manager(&task_manager_dep))
        .and(with_clients(&clients_dep))
        .and_then(ws_handlers::ws_train_handler);

    let assign_route = warp::path!("assign")
        .and(warp::get())
        .and(with_task_manager(&task_manager_dep))
        .and(with_models_sources(&models_sources_dep))
        .and_then(rest_handlers::assign_handler);

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

    let routes = ws_train_route
        .or(assign_route)
        .or(submit_test_route)
        .or(recent_route)
        .or(register_route)
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
