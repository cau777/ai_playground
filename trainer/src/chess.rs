use codebase::chess::board::Board;
use codebase::integration::layers_loading::ModelXmlConfig;
use codebase::integration::random_picker::RandomPicker;
use codebase::nn::controller::NNController;
use codebase::nn::layers::nn_layers::GenericStorage;
use codebase::utils::{Array2F};
use codebase::utils::ndarray::{Axis, stack};
use rand::thread_rng;
use crate::{EnvConfig, ServerClient};
use crate::files::{load_file_lines};

const NAME: &str = "chess";
const BATCH_SIZE: usize = 128;

pub fn train(initial: GenericStorage, model_config: ModelXmlConfig, config: &EnvConfig, client: &ServerClient) {
    let mut controller = NNController::load(model_config.main_layer, model_config.loss_func, initial).unwrap();
    let mut endgames_string = String::new();

    let endgames = load_file_lines("endgames", NAME, config, &mut endgames_string).unwrap();
    let mut rng = thread_rng();

    for version in 0..config.versions {
        let mut total_loss = 0.0;

        println!("Start {}", version + 1);
        for epoch in 0..config.epochs_per_version {
            let mut picker = RandomPicker::new(endgames.len());
            let mut all_chosen = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                let chosen = picker.pick(&mut rng);
                let result = &endgames[chosen][0..3];
                let result = if result == "1-0" { 1.0 } else { -1.0 };
                let game = &endgames[chosen][4..];
                all_chosen.push((result, game));
            }

            let chosen_games: Vec<_> = all_chosen
                .iter()
                .map(|(_, o)| *o)
                .map(Board::from_literal)
                .map(|o| o.to_array())
                .collect();
            let chosen_games_views: Vec<_> = chosen_games.iter().map(|o| o.view()).collect();
            let inputs = stack(Axis(0), &chosen_games_views).unwrap();

            let expected = Array2F::from_shape_vec((BATCH_SIZE, 1),
                                                   all_chosen.into_iter().map(|(o, _)| o).collect()).unwrap();

            let loss = controller.train_batch(inputs.into_dyn(), &expected.into_dyn()).unwrap();
            total_loss += loss;

            if epoch % 16 == 0 {
                println!("    {} -> loss={}", epoch, loss);
            }
        }

        let avg_loss = total_loss / config.epochs_per_version as f64;

        println!("    Finished with avg_loss={}", avg_loss);
        client.submit(&controller.export(), avg_loss, NAME);
    }
}