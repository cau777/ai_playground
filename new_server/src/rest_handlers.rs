use serde::{Deserialize, Serialize};
use warp::{
    reply::{self},
    Reply
};

use crate::{utils::EndpointResult, CurrentModelDep, ModelsSourcesDep, TaskManagerDep};

pub async fn assign_handler(
    task_manager: TaskManagerDep,
    sources: ModelsSourcesDep,
) -> EndpointResult<reply::Json> {
    let mut task_manager = task_manager.write().await;
    let sources = sources.read().await;

    let result = task_manager.get_task(&sources);
    Ok(warp::reply::json(&result))
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
