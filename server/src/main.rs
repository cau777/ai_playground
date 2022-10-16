mod ops;
mod cors;

#[macro_use]
extern crate rocket;

use std::sync::RwLock;
use codebase::integration::compression::decompress_default;
use codebase::integration::proto_loading::{load_task_result_from_bytes, TaskResult};
use rocket::serde::json::Json;
use rocket::serde::{Serialize, Deserialize};
use rocket::{Build, Rocket};
use crate::ops::model_source::{ModelSources, ModelSourcesState};
use crate::ops::current_model::{CurrentModel, CurrentModelState};
use crate::cors::CORS;
use crate::ops::path_utils;

#[get("/wakeup")]
fn wakeup() {
    println!("wake up");
}

#[options("/wakeup")]
fn wakeup_options() -> rocket::response::status::NoContent { rocket::response::status::NoContent }

#[get("/best")]
async fn best(sources: &ModelSourcesState) -> String {
    let mut sources = sources.write().unwrap();
    let sources = sources.digits.value_mut();
    format!("{}", sources.best())
}

#[options("/best")]
fn best_options() -> rocket::response::status::NoContent { rocket::response::status::NoContent }

#[derive(Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct AssignRequest {
    
}

#[derive(Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct AssignResponse {
    task_name: String,
    model_config: String,
    model_data: String,
    data_path: String,
}

#[post("/assign", data = "<body>")]
fn assign(body: Json<AssignRequest>, sources: &ModelSourcesState) -> Json<AssignResponse> {
    let mut sources = sources.write().unwrap();
    let sources = sources.digits.value_mut();
    let model_data = path_utils::get_model_url("digits", sources.most_recent());
    
    Json(AssignResponse {
        task_name: "train".to_owned(),
        model_config: "models/digits/model.xml".to_owned(),
        model_data,
        data_path: "static/digits/train/train.0.dat".to_owned(),
    })
}

#[options("/assign")]
fn assign_options() -> rocket::response::status::NoContent { rocket::response::status::NoContent }

#[post("/submit", data = "<body>")]
fn submit(current_model: &CurrentModelState, body: &[u8], sources: &ModelSourcesState) {
    let body = decompress_default(body).unwrap();
    let task_result = load_task_result_from_bytes(&body).unwrap();
    match task_result {
        TaskResult::Train(deltas) => {
            let mut current_model = current_model.write().unwrap();
            current_model.increment(deltas);
            let mut sources = sources.write().unwrap();
            let sources = sources.digits.value_mut();
            current_model.save_to(sources).unwrap();
        }
    }
}

#[options("/submit")]
fn submit_options() -> rocket::response::status::NoContent { rocket::response::status::NoContent }

#[delete("/clear_all")]
fn clear_all(current_model: &CurrentModelState, sources: &ModelSourcesState) {
    let mut sources = sources.write().unwrap();
    let mut source = sources.digits.value_mut();
    source.clear_all().unwrap();
    current_model.write().unwrap().reload(&mut source);
}

#[options("/clear_all")]
fn clear_all_options() -> rocket::response::status::NoContent { rocket::response::status::NoContent }

#[launch]
fn rocket() -> Rocket<Build> {
    let mut sources = ModelSources::new();
    let model = CurrentModel::new(sources.digits.value_mut());
    rocket::build()
    .mount("/", routes![
        submit, submit_options,
        best, best_options,
        wakeup, wakeup_options,
        assign, assign_options,
        clear_all, clear_all_options
    ])
        .attach(CORS)
        .manage(RwLock::new(sources))
        .manage(RwLock::new(model))
}