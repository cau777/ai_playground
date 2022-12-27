use std::borrow::BorrowMut;
use std::cell::{Cell, RefCell};
use std::fs::OpenOptions;
use std::sync::Arc;
use codebase::chess::board::Board;
use codebase::chess::board_controller::BoardController;
use codebase::chess::game_result::{DrawReason, GameResult, WinReason};
use codebase::chess::movement::Movement;
use codebase::chess::openings::openings_tree::OpeningsTree;
use codebase::nn::controller::NNController;
use codebase::utils::{Array2F};
use codebase::utils::ndarray::{AssignElem, Axis, stack};
use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::chess::{BATCH_SIZE};
use crate::EnvConfig;

pub struct GamesTrainer {
    opening_tree: Arc<OpeningsTree>,
}

#[derive(Debug, Default)]
pub struct GameMetrics {
    pub total_games: u32,
    pub aborted_rate: f64,
    pub white_win_rate: f64,
    pub black_win_rate: f64,
    pub draw_rate: f64,
    pub average_len: f64,
}

fn apply_best_move(best_moves: &[Cell<Option<(f32, Movement)>>], board_index: usize, value: f32, m: Movement) {
    let prev = best_moves[board_index].get();
    best_moves[board_index].replace(Some(
        match prev {
            Some(prev) => if prev.0 > value { prev } else { (value, m) },
            None => (value, m),
        })
    );
}

impl GamesTrainer {
    pub fn new(config: &EnvConfig) -> Self {
        let file = OpenOptions::new().read(true).open(format!("{}/chess/openings.dat", config.mounted_path)).unwrap();
        Self {
            opening_tree: Arc::new(OpeningsTree::load_from_file(file).unwrap()),
        }
    }

    pub fn train_version(&self, controller: &mut NNController, config: &EnvConfig) -> GameMetrics {
        let (games, metrics) = self.play_games(controller, 4);
        for chunk in games.chunks(BATCH_SIZE) {
            let inputs: Vec<_> = chunk.iter().map(|(b, _)| b.to_array()).collect();
            let views: Vec<_> = inputs.iter().map(|o| o.view()).collect();
            let inputs = stack(Axis(0), &views).unwrap();

            let expected: Vec<_> = chunk.iter().map(|(_, v)| *v).collect();
            let expected = Array2F::from_shape_vec((expected.len(), 1), expected).unwrap();
            controller.train_batch(inputs.into_dyn(), &expected.into_dyn()).unwrap();
        }
        metrics
    }

    fn play_games(&self, controller: &NNController, count: usize) -> (Vec<(Board, f32)>, GameMetrics) {
        println!();
        let mut metrics = GameMetrics::default();
        let factor = 1.0 / count as f64;
        let mut playing = Vec::with_capacity(count);
        for _ in 0..count {
            let mut controller = BoardController::new_start();
            controller.add_openings_tree(self.opening_tree.clone());
            playing.push(controller);
        }

        let mut playing: Vec<_> = playing
            .into_iter()
            .map(RefCell::new)
            .collect();

        let mut result = Vec::new();
        let mut rng = thread_rng();

        while !playing.is_empty() {
            // The side is the same for all games
            let side = playing[0].borrow_mut().side_to_play();

            let best_moves = vec![Cell::new(None); playing.len()];

            let moves_iter = playing
                .iter()
                .enumerate()
                .filter_map(|(i, o)| {
                    let continuations = o.borrow_mut().get_opening_continuations();
                    if !continuations.is_empty() {
                        let chosen = continuations[rng.gen_range(0..continuations.len())];
                        o.borrow_mut().apply_move(chosen);
                        if i == 0 {
                            print!("{} ", chosen);
                        }
                        None
                    } else {
                        Some((i, o))
                    }
                })
                .flat_map(|(i, o)| {
                    o.borrow_mut().get_possible_moves(side)
                        .into_iter()
                        .map(move |m| (i, m))
                });

            let chunks_iter = moves_iter.filter_map(|(i, m)| {
                let mut o = playing[i].borrow_mut();
                o.apply_move(m);
                let possible = o.get_possible_moves(side);

                match o.get_game_result(&possible) {
                    GameResult::Undefined => {
                        let array = o.current().to_array();
                        o.revert();
                        Some((i, array, m))
                    }
                    GameResult::Draw(_) => {
                        apply_best_move(&best_moves, i, 0.0, m);
                        o.revert();
                        None
                    }
                    GameResult::Win(true, _) => {
                        apply_best_move(&best_moves, i, 1_000_000.0, m);
                        o.revert();
                        None
                    }
                    GameResult::Win(false, _) => {
                        apply_best_move(&best_moves, i, -1_000_000.0, m);
                        o.revert();
                        None
                    }
                }
            }).chunks(BATCH_SIZE);

            for chunk in &chunks_iter {
                let chunk: Vec<_> = chunk.collect();
                let views: Vec<_> = chunk.iter().map(|(_, arr, _)| arr.view()).collect();
                let inputs = stack(Axis(0), &views).unwrap();

                let outputs = controller.eval_for_train(inputs.into_dyn()).unwrap();
                for (i, v) in outputs.output.outer_iter().enumerate() {
                    let (board_index, _, m) = chunk[i];
                    let value = *v.first().unwrap();
                    let value = if side { value } else { -value };

                    apply_best_move(&best_moves, board_index, value, m);
                }
            }

            // Apply chosen moves
            best_moves.into_iter()
                .filter_map(|o| o.into_inner())
                .map(|(_, m)| m)
                .enumerate()
                .for_each(|(i, m)| {
                    if i == 0 {
                        print!("{} ", m);
                    }
                    playing[i].borrow_mut().apply_move(m)
                });

            let mut i = 0;
            while i < playing.len() {
                let moves = playing[i].borrow().get_possible_moves(!side);
                let game_result = playing[i].borrow_mut().get_game_result(&moves);

                if let Some(result_value) = game_result.value() {
                    metrics.total_games += 1;
                    metrics.average_len += playing[i].borrow_mut().half_moves() as f64 * factor;

                    let removed = playing.swap_remove(i);
                    for pos in removed.into_inner().into_non_opening_positions() {
                        result.push((pos, result_value))
                    }

                    match game_result {
                        GameResult::Draw(reason) => {
                            metrics.draw_rate += 1.0 * factor;
                            if let DrawReason::Aborted = reason {
                                metrics.aborted_rate += 1.0 * factor
                            }
                        },
                        GameResult::Win(true, _) => metrics.white_win_rate += 1.0 * factor,
                        GameResult::Win(false, _) => metrics.black_win_rate += 1.0 * factor,
                        GameResult::Undefined => {}
                    }
                } else {
                    i += 1;
                }
            }
            // let mut need_eval: Vec<_> = playing.iter_mut()
            //     .filter_map(|o| {
            //         let continuations = o.get_opening_continuations();
            //         if !continuations.is_empty() {
            //             let chosen = continuations[rng.gen_range(0..continuations.len())];
            //             o.apply_move(chosen);
            //             None
            //         } else {
            //             Some(o)
            //         }
            //     })
            //     .collect();
            //
            // if need_eval.is_empty() {
            //     continue;
            // }
            //
            // let moves: Vec<_> = need_eval.iter().enumerate()
            //     .flat_map(|(i, o)| {
            //         o.get_possible_moves(side).into_iter().map(move |o| (i, o))
            //     })
            //     .collect();
            // let mut best_moves: Vec<Option<(f32, Movement)>> = vec![None; need_eval.len()];
            //
            // let chunks = moves.chunks(BATCH_SIZE);
            //
            // for chunk in chunks {
            //     let arrays: Vec<_> = chunk.iter().copied().map(|(i, m)| {
            //         // Get the resulting position after playing each move
            //         need_eval[i].apply_move(m);
            //         let arr = need_eval[i].current().to_array();
            //         need_eval[i].revert();
            //         arr
            //     }).collect();
            //
            //     let views: Vec<_> = arrays.iter().map(|o| o.view()).collect();
            //     let inputs = stack(Axis(0), &views).unwrap();
            //
            //     let outputs = controller.eval_for_train(inputs.into_dyn()).unwrap();
            //     for (i, v) in outputs.output.outer_iter().enumerate() {
            //         let (board_index, movement) = chunk[i];
            //         let value = *v.first().unwrap();
            //         let value = if side { value } else { -value };
            //
            //         let current = best_moves[board_index];
            //         best_moves[board_index] = Some(match current {
            //             Some(prev) => if prev.0 > value { prev } else { (value, movement) },
            //             None => (value, movement),
            //         });
            //     }
            // }
            //
            // best_moves.into_iter()
            //     .map(|o| o.unwrap()).map(|(_, m)| m)
            //     .enumerate()
            //     .for_each(|(i, m)| need_eval[i].apply_move(m));
            //
            // let mut i = 0;
            // while i < playing.len() {
            //     let moves = playing[i].get_possible_moves(!side);
            //     let game_result = playing[i].get_game_result(&moves);
            //
            //     if let Some(result_value) = game_result.value() {
            //         metrics.total_games += 1;
            //         metrics.average_len += playing[i].half_moves() as f64 * factor;
            //
            //         let removed = playing.swap_remove(i);
            //         for pos in removed.into_non_opening_positions() {
            //             result.push((pos, result_value))
            //         }
            //
            //         match game_result {
            //             GameResult::Draw(DrawReason::Aborted) => metrics.aborted_rate += 1.0 * factor,
            //             GameResult::Draw(_) => metrics.draw_rate += 1.0 * factor,
            //             GameResult::Win(true, _) => metrics.white_win_rate += 1.0 * factor,
            //             GameResult::Win(false, _) => metrics.black_win_rate += 1.0 * factor,
            //             GameResult::Undefined => {}
            //         }
            //     } else {
            //         i += 1;
            //     }
            // }
        }

        (result, metrics)
    }
}

