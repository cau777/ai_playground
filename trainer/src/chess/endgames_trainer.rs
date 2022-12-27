use codebase::chess::board::Board;
use codebase::integration::random_picker::RandomPicker;
use codebase::nn::controller::NNController;
use codebase::utils::Array2F;
use codebase::utils::ndarray::{Axis, stack};
use rand::thread_rng;
use crate::chess::{BATCH_SIZE, NAME};
use crate::EnvConfig;
use crate::files::load_file_lines;

pub struct EndgamesTrainer {
    endgames: Vec<String>,
}

#[derive(Debug)]
pub struct EndgamesMetrics {
    pub avg_loss: f64,
}

impl EndgamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        let mut buffer = String::new();
        let endgames = load_file_lines("endgames", NAME, config, &mut buffer).unwrap();
        Self {
            endgames: endgames.into_iter().map(|o| o.to_owned()).collect()
        }
    }

    pub fn train_version(&self, controller: &mut NNController, config: &EnvConfig) -> EndgamesMetrics {
        let mut total_loss = 0.0;
        let mut rng = thread_rng();

        for epoch in 0..config.epochs_per_version {
            let mut picker = RandomPicker::new(self.endgames.len());
            let mut all_chosen = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                let chosen = picker.pick(&mut rng);
                let result = &self.endgames[chosen][0..3];
                let result = if result == "1-0" { 1.0 } else { -1.0 };
                let game = &self.endgames[chosen][4..];
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

            if epoch % 20 == 0 {
                println!("    {} -> loss={}", epoch, loss);
            }
        }

        let avg_loss = total_loss / config.epochs_per_version as f64;
        EndgamesMetrics {avg_loss}
    }
}