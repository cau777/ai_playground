use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::PieceOps;

struct PawnOps;

impl PieceOps for PawnOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        todo!()
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        todo!()
    }
}