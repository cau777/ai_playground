use crate::data::url_creator;
use codebase::integration::serialization::deserialize_storage;
use serde::{Deserialize, Serialize};
use warp::{reject, reply, Reply, hyper::{StatusCode}};

use crate::{
    utils::EndpointResult, CurrentModelDep, EnvConfigDep, ModelsSourcesDep, TaskManagerDep,
};

pub async fn get_best(
    sources: ModelsSourcesDep,
    config: EnvConfigDep,
) -> EndpointResult<reply::Json> {
    let sources = sources.read().await;
    let best = sources.best();
    Ok(reply::json(&url_creator::get_model_data(
        &config, "digits", best,
    )))
}
pub async fn get_config(config: EnvConfigDep) -> EndpointResult<reply::Json> {
    Ok(reply::json(&url_creator::get_model_config(&config, "digits")))
}

pub async fn assign_handler(
    task_manager: TaskManagerDep,
    sources: ModelsSourcesDep,
) -> EndpointResult<reply::Json> {
    let mut task_manager = task_manager.write().await;
    let sources = sources.read().await;

    let result = task_manager.get_task(&sources);
    Ok(reply::json(&result))
}

pub async fn submit_train(
    task_manager: TaskManagerDep,
    sources: ModelsSourcesDep,
    current_model: CurrentModelDep,
    body: warp::hyper::body::Bytes,
) -> EndpointResult<impl Reply> {
    match deserialize_storage(&body) {
        Ok(deltas) => {
            let mut current_model = current_model.write().await;
            current_model.increment(1, deltas);
            if current_model.should_save() {
                let mut sources = sources.write().await;
                current_model.save_to(&mut sources).unwrap();

                let mut task_manager = task_manager.write().await;
                task_manager.add_to_test(current_model.version(), sources.test_count());
                
                // TODO: broadcast updates
            }

            Ok(StatusCode::OK)
        }
        Err(_) => Ok(StatusCode::BAD_REQUEST),
    }
}

pub async fn submit_test(
    manager: TaskManagerDep,
    sources: ModelsSourcesDep,
    data: TaskTestResult,
) -> EndpointResult<impl Reply> {
    let mut task_manager = manager.write().await;
    let mut sources = sources.write().await;

    match task_manager.complete_test_task(&mut sources, data.version, data.batch, data.accuracy) {
        Some(_) => Ok(reply::reply()),
        None => Err(warp::reject::not_found()),
    }
}

pub async fn register() -> EndpointResult<String> {
    Ok(format!("ws://127.0.0.1:8000/ws_train"))
}

pub async fn recent(current_model: CurrentModelDep) -> EndpointResult<Vec<u8>> {
    let current_model = current_model.read().await;
    Ok(current_model.export())
}

#[derive(Serialize, Deserialize)]
pub struct TaskTestResult {
    version: u32,
    batch: u32,
    accuracy: f64,
}
