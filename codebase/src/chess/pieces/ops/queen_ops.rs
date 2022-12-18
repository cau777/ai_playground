use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::lines::{diagonal_valid, DIAGONALS, find_possible_moves_line, line_valid, LINES, piece_in_diagonal, piece_in_line};
use crate::chess::pieces::ops::PieceOps;
use crate::chess::utils::CoordIndexed;

struct QueenOps;

impl PieceOps for QueenOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        for [off_row, off_col] in DIAGONALS.into_iter().chain(LINES) {
            find_possible_moves_line(result, board, side, from, off_row, off_col);
        }
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        let piece = board.pieces.get_at(to);
        (piece.is_empty() || piece.side != side) && (
            (diagonal_valid(from, to) && !piece_in_diagonal(board, from, to))
                || (line_valid(from, to) && !piece_in_line(board, from, to))
        )
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
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ p _ Q _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ P _\
        P P P P P P _ P\
        R N B _ K B N R");

        let mut result = Vec::new();
        QueenOps::find_possible_moves(&mut result, &board, true, Coord::from_notation("G5"));
        let result = HashSet::from_iter(result.into_iter().map(|o| o.to));
        let expected = HashSet::from(["G4", "E5", "F5", "G6", "G7", "H5", "F4", "E3", "F6", "E7", "D8", "H6", "H4"].map(Coord::from_notation));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_can_move_to() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ p _ Q _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ P _\
        P P P P P P _ P\
        R N B _ K B N R");

        let from = Coord::from_notation("G5");
        for possible in ["G4", "E5", "F5", "G6", "G7", "H5", "F4", "E3", "F6", "E7", "D8", "H6", "H4"].map(Coord::from_notation) {
            assert!(QueenOps::can_move_to(&board, true, from, possible));
        }

        for impossible in ["G5", "A1", "E4", "G3", "G2"].map(Coord::from_notation) {
            assert!(!QueenOps::can_move_to(&board, true, from, impossible));
        }
    }
}