#[macro_use]
extern crate rocket;

use std::fs::{File, OpenOptions};
use std::io::Write;
use codebase::nn::controller::NNController;
use codebase::nn::layers::dense_layer::{DenseLayerConfig, DenseLayerInit};
use codebase::nn::layers::nn_layers::Layer;
use codebase::nn::lr_calculators::constant_lr::ConstantLrConfig;
use codebase::nn::lr_calculators::lr_calculator::LrCalc;
use codebase::utils::Array1F;
use rocket::{Build, Rocket};
use rocket::response::status;
use rocket::serde::{Deserialize, json::Json};

#[get("/")]
fn index() -> String {
    let mut controller = NNController::new(Layer::Dense(DenseLayerConfig {
        out_values: 20,
        in_values: 20,
        init_mode: DenseLayerInit::Random(),
        weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
    })).unwrap();

    let inputs = Array1F::ones(20).into_dyn();
    let out = controller.eval_one(inputs);
    format!("{:?}", out)
}

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct FileWriteRequest {
    name: String,
}

#[post("/write", data = "<data>")]
fn write(data: Json<FileWriteRequest>) -> status::Accepted<()> {
    println!("{}", &data.name);
    let mut a = OpenOptions::new().write(true).open(format!("./files/{}.txt", &data.name)).unwrap();
    a.write_all("123".as_bytes()).unwrap();

    status::Accepted(None)
}

#[launch]
fn rocket() -> Rocket<Build> {
    rocket::build().mount("/", routes![index, write])
}