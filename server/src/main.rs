mod ops;
mod cors;

#[macro_use]
extern crate rocket;

use std::collections::HashMap;
use std::sync::RwLock;
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::proto_loading::{load_model_from_bytes, load_task_result_from_bytes, save_model_bytes, TaskResult};
use codebase::nn::layers::nn_layers::GenericStorage;
use rocket::{Build, Rocket, State};
use rocket::futures::AsyncReadExt;
use rocket::http::Method;
use crate::ops::model_source::{ModelSources, ModelSourcesState};
use crate::ops::current_model::{CurrentModel, CurrentModelState};
use crate::cors::CORS;

// #[get("/")]
// fn index() -> String {
//     let mut controller = NNController::new(Layer::Dense(DenseLayerConfig {
//         out_values: 20,
//         in_values: 20,
//         init_mode: DenseLayerInit::Random(),
//         weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
//         biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
//     })).unwrap();
//
//     let inputs = Array1F::ones(20).into_dyn();
//     let out = controller.eval_one(inputs);
//     format!("{:?}", out)
// }

#[get("/best")]
async fn best(sources: &ModelSourcesState) -> String {
    let mut best = sources.write().unwrap();
    let best = &mut best.testy;
    let best = best.value();
    // format!("https://aiplaygroundmodels.file.core.windows.net/models/testy/{}.model", best.best)
    // format!("")
    "2".to_owned()
}

#[get("/current")]
fn current(current_model: &CurrentModelState) -> Vec<u8> {
    current_model.read().unwrap().export()
}

#[post("/submit", data = "<body>")]
fn submit(current_model: &CurrentModelState, body: &[u8]) {
    let body = decompress_default(body).unwrap();
    let task_result = load_task_result_from_bytes(&body).unwrap();
    match task_result {
        TaskResult::Train(deltas) => {
            let mut current_model = current_model.write().unwrap();
            current_model.increment(deltas);
            //     TODO: save
        }
    }
}

#[launch]
fn rocket() -> Rocket<Build> {
    let mut sources = ModelSources::new();
    let model = CurrentModel::new(&sources.testy.value());
    rocket::build()
        .mount("/", routes![submit, best, current])
        .attach(CORS)
        .manage(RwLock::new(sources))
        .manage(RwLock::new(model))
}