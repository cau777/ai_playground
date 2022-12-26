use codebase::chess::board::Board;
use codebase::chess::board_controller::BoardController;
use codebase::chess::game_result::{DrawReason, GameResult};
use codebase::chess::movement::Movement;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F};
use codebase::utils::ndarray::{Axis, stack};
use rand::{Rng, thread_rng};
use crate::chess::{BATCH_SIZE, NAME};
use crate::EnvConfig;

pub struct GamesTrainer {}

#[derive(Debug)]
pub struct GameMetrics {}

impl GamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        Self {}
    }

    pub fn train_version(&self, controller: &mut NNController, config: &EnvConfig) -> GameMetrics {
        let games = self.play_games(controller, config.epochs_per_version as usize);
        for chunk in games.chunks(BATCH_SIZE) {
            let inputs: Vec<_> = chunk.iter().map(|(b, _)| b.to_array()).collect();
            let views: Vec<_> = inputs.iter().map(|o| o.view()).collect();
            let inputs = stack(Axis(0), &views).unwrap();

            let expected = Array2F::from_shape_vec((inputs.len(), 1), chunk.iter().map(|(_, v)| *v).collect()).unwrap();
            controller.train_batch(inputs.into_dyn(), &expected.into_dyn()).unwrap();
        }
        GameMetrics {}
    }

    fn play_games(&self, controller: &NNController, count: usize) -> Vec<(Board, f32)> {
        let mut playing = Vec::with_capacity(count);
        for _ in 0..count {
            playing.push(BoardController::new_start());
        }

        let mut result = Vec::new();
        let mut rng = thread_rng();

        while !playing.is_empty() {
            let side = playing.first().unwrap().side_to_play();

            let mut need_eval: Vec<_> = playing.iter_mut()
                .filter_map(|o| {
                    let continuations = o.get_opening_continuations();
                    if !continuations.is_empty() {
                        let chosen = continuations[rng.gen_range(0..continuations.len())];
                        o.apply_move(chosen);
                        None
                    } else {
                        Some(o)
                    }
                })
                .collect();

            if need_eval.is_empty() {
                continue;
            }

            let moves: Vec<_> = need_eval.iter().enumerate()
                .flat_map(|(i, o)| {
                    o.get_possible_moves(side).into_iter().map(move |o| (i, o))
                })
                .collect();
            let mut best_moves: Vec<Option<(f32, Movement)>> = vec![None; need_eval.len()];

            let chunks = moves.chunks(BATCH_SIZE);

            for chunk in chunks {
                let arrays: Vec<_> = chunk.iter().copied().map(|(i, m)| {
                    //Get the resulting position after playing each move
                    need_eval[i].apply_move(m);
                    let arr = need_eval[i].current().to_array();
                    need_eval[i].revert();
                    arr
                }).collect();

                let views: Vec<_> = arrays.iter().map(|o| o.view()).collect();
                let inputs = stack(Axis(0), &views).unwrap();

                let outputs = controller.eval_for_train(inputs.into_dyn()).unwrap();
                for (i, v) in outputs.output.outer_iter().enumerate() {
                    let (board_index, movement) = chunk[i];
                    let value = *v.first().unwrap();
                    let value = if side { value } else { -value };

                    let current = best_moves[board_index];
                    best_moves[board_index] = Some(match current {
                        Some(prev) => if prev.0 > value { prev } else { (value, movement) },
                        None => (value, movement),
                    });
                }
            }

            best_moves.into_iter()
                .map(|o| o.unwrap()).map(|(_, m)| m)
                .enumerate()
                .for_each(|(i, m)| need_eval[i].apply_move(m));

            let mut i = 0;
            while i < playing.len() {
                let moves = playing[i].get_possible_moves(!side);
                let game_result = playing[i].get_game_result(&moves);

                if let GameResult::Draw(DrawReason::Aborted) = &game_result {
                    println!("Aborted");
                }

                if let Some(game_result) = game_result.value() {
                    let removed = playing.swap_remove(i);
                    for pos in removed.into_non_opening_positions() {
                        result.push((pos, game_result))
                    }
                } else {
                    i += 1;
                }
            }
        }

        result
    }
}

