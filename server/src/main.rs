mod cors;
mod ops;

#[macro_use]
extern crate rocket;

use crate::cors::CORS;
use crate::ops::current_model::{CurrentModel, CurrentModelState};
use codebase::integration::compression::decompress_default;
use codebase::integration::proto_loading::{load_task_result_from_bytes, TaskResult};
use ops::lazy::{LazyWithoutArgs, LazyWithoutArgsOps};
use ops::lazy_by_module::{ByModule, ModelSourcesState, TaskManagerState};
use ops::model_source::ModelSource;
use ops::task_manager::{Task, TaskManager};
use rocket::serde::json::Json;
use rocket::serde::{Deserialize, Serialize};
use rocket::{Build, Rocket};
use std::sync::RwLock;

#[get("/wakeup")]
fn wakeup() {
    println!("wake up");
}

#[options("/wakeup")]
fn wakeup_options() -> rocket::response::status::NoContent {
    rocket::response::status::NoContent
}

#[get("/best")]
async fn best(sources: &ModelSourcesState) -> String {
    let mut sources = sources.write().unwrap();
    let sources = sources.digits.v_mut();
    format!("{}", sources.best())
}

#[options("/best")]
fn best_options() -> rocket::response::status::NoContent {
    rocket::response::status::NoContent
}

#[derive(Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct AssignRequest {}

#[post("/assign", data = "<body>")]
fn assign(
    body: Json<AssignRequest>,
    source: &ModelSourcesState,
    task_manager: &TaskManagerState,
) -> Json<Task> {
    let mut source = source.write().unwrap();
    let source = source.digits.v_mut();
    let mut task_manager = task_manager.write().unwrap();
    let task_manager = &mut task_manager.digits;

    Json(task_manager.get_task(source))
}

#[options("/assign")]
fn assign_options() -> rocket::response::status::NoContent {
    rocket::response::status::NoContent
}

#[post("/submit", data = "<body>")]
fn submit(
    current_model: &CurrentModelState,
    body: &[u8],
    source: &ModelSourcesState,
    task_manager: &TaskManagerState,
) {
    let body = decompress_default(body).unwrap();
    let task_result = load_task_result_from_bytes(&body).unwrap();
    match task_result {
        TaskResult::Train(deltas) => {
            let mut current_model = current_model.write().unwrap();
            current_model.increment(deltas);
            if current_model.should_save() {
                let mut source = source.write().unwrap();
                let source = source.digits.v_mut();
                current_model.save_to(source).unwrap();

                let mut task_manager = task_manager.write().unwrap();
                let task_manager = &mut task_manager.digits;
                task_manager.add_to_test(current_model.version(), source.test_count());
            }
        }
        TaskResult::Test(version, batch, accuracy) => {
            println!("{} {} {}", version, batch, accuracy);
            let mut source = source.write().unwrap();
            let source = source.digits.v_mut();

            let mut task_manager = task_manager.write().unwrap();
            let task_manager = &mut task_manager.digits;
            
            task_manager.complete_test_task(source, version, batch, accuracy);
        }
    }
}

#[options("/submit")]
fn submit_options() -> rocket::response::status::NoContent {
    rocket::response::status::NoContent
}

#[delete("/clear_all")]
fn clear_all(current_model: &CurrentModelState, sources: &ModelSourcesState) {
    let mut sources = sources.write().unwrap();
    let mut source = sources.digits.v_mut();
    source.clear_all().unwrap();
    current_model.write().unwrap().reload(&mut source);
}

#[options("/clear_all")]
fn clear_all_options() -> rocket::response::status::NoContent {
    rocket::response::status::NoContent
}

#[launch]
fn rocket() -> Rocket<Build> {
    let mut sources = ByModule::new(LazyWithoutArgs::new(|_| {
        ModelSource::new("digits", 469, 2).unwrap() // TODO: 79
    }));
    let model = CurrentModel::new(sources.digits.v_mut());
    let assigners = ByModule::new(TaskManager::new());

    rocket::build()
        .mount(
            "/",
            routes![
                submit,
                submit_options,
                best,
                best_options,
                wakeup,
                wakeup_options,
                assign,
                assign_options,
                clear_all,
                clear_all_options
            ],
        )
        .attach(CORS)
        .manage(RwLock::new(sources))
        .manage(RwLock::new(model))
        .manage(RwLock::new(assigners))
}
