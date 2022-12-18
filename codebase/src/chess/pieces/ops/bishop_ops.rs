use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::lines::{diagonal_valid, DIAGONALS, find_possible_moves_line, piece_in_diagonal};
use crate::chess::pieces::ops::PieceOps;
use crate::chess::utils::CoordIndexed;

pub struct BishopOps;

impl PieceOps for BishopOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        for [off_row, off_col] in DIAGONALS {
            find_possible_moves_line(result, board, side, from, off_row, off_col);
        }
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        let piece = board.pieces.get_at(to);
        (piece.is_empty() || piece.side != side) && diagonal_valid(from, to) &&
            !piece_in_diagonal(board, from, to)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;

    #[test]
    fn test_find_possible_moves() {
        let board = Board::from_literal("\
        r n b q k b n r\
        _ p p p p p p p\
        _ _ _ _ _ _ _ _\
        p _ _ _ _ _ _ _\
        _ _ _ _ _ B _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N _ Q K B N R");

        let mut result = Vec::new();
        BishopOps::find_possible_moves(&mut result, &board, true, Coord::from_notation("F4"));
        let result = HashSet::from_iter(result.into_iter().map(|o| o.to));
        let expected = HashSet::from(["E3", "G3", "G5", "H6", "E5", "D6", "C7"].map(Coord::from_notation));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_can_move_to() {
        let board = Board::from_literal("\
        r n b q k b n r\
        _ p p p p p p p\
        _ _ _ _ _ _ _ _\
        p _ _ _ _ _ _ _\
        _ _ _ _ _ B _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N _ Q K B N R");
        let from = Coord::from_notation("F4");
        for possible in ["E3", "G3", "G5", "H6", "E5", "D6", "C7"].map(Coord::from_notation) {
            assert!(BishopOps::can_move_to(&board, true, from, possible));
        }

        for impossible in ["F4", "G6", "G4", "H8", "D5", "A1", "E7"].map(Coord::from_notation) {
            assert!(!BishopOps::can_move_to(&board, true, from, impossible));
        }
    }
}