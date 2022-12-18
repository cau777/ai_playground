mod move_applying;
mod finding_moves;

use crate::chess::board::Board;

#[derive(Eq, PartialEq, Clone)]
pub struct BoardController {
    // TODO: 3-fold repetition
    current: Board,
    half_moves: u16,
    last_50mr_reset: u16,
}

impl BoardController {
    pub fn new_start() -> Self {
        Self { current: Board::new(), half_moves: 0, last_50mr_reset: 0 }
    }

    pub fn new(board: Board, half_moves: u16, last_50mr_reset: u16) -> Self {
        Self { current: board, half_moves, last_50mr_reset }
    }
}