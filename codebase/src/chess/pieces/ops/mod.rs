use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;

mod pawn_ops;
mod knight_ops;

pub trait PieceOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord);
    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool;
}