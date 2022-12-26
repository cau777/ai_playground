use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::integration::serde_utils::Pairs;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use rand::thread_rng;
use crate::{EnvConfig, ServerClient};
use crate::files::load_file_data;

const NAME: &str = "digits";
const BATCH_SIZE: usize = 128;

pub fn train(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig, client: &ServerClient) {
    let mut controller = NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap();
    let train_data = load_file_data("train", NAME, config).unwrap();
    let validate_data = load_file_data("validate", NAME, config).unwrap();
    let mut rng = thread_rng();

    for version in 0..config.versions {
        let mut total_loss = 0.0;

        println!("Start {}", version + 1);
        for epoch in 0..config.epochs_per_version {
            let data = train_data.pick_rand(BATCH_SIZE, &mut rng);
            let loss = controller.train_batch(data.inputs, &data.expected).unwrap();
            total_loss += loss;

            if epoch % 16 == 0 {
                println!("    {} -> loss={}", epoch, loss);
            }
        }

        let avg_loss = total_loss / config.epochs_per_version as f64;
        let tested_loss = validate(&validate_data, &controller);

        println!("    Finished with avg_loss={} and tested_loss={}", avg_loss, tested_loss);
        client.submit(&controller.export(), avg_loss, NAME);
    }
}

fn validate(data: &Pairs, controller: &NNController) -> f64 {
    println!("Started testing");

    let mut total = 0.0;
    let mut count = 0;
    for batch in data.chunks_iter(256) {
        total += controller.test_batch(batch.0.into_owned(), &batch.1.into_owned()).unwrap();
        count += 1;
    }
    total / (count as f64)
}
