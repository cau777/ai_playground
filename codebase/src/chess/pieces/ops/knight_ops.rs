use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::PieceOps;
use crate::chess::utils::CoordIndexed;

pub struct KnightOps;

const OFFSETS: [[i8; 2]; 8] = [
    [1, 2],
    [2, 1],
    [-1, 2],
    [-2, 1],
    [-2, -1],
    [-1, -2],
    [1, -2],
    [2, -1],
];

impl PieceOps for KnightOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        for [off_row, off_col] in OFFSETS {
            let option = from.add_checked(off_row, off_col);
            if let Some(to) = option {
                if !to.in_bounds() { continue; }
                let piece = board.pieces.get_at(to);

                if piece.is_empty() || piece.side != side {
                    result.push(Movement::new(from, to))
                }
            }
        }
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        let distance = from.distance_2d(to);
        (distance.row == 1 || distance.col == 1) && (distance.row == 2 || distance.col == 2) && {
            let piece = board.pieces.get_at(to);
            piece.is_empty() || piece.side != side
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;

    #[test]
    fn test_possible_moves() {
        let board = Board::from_literal("\
        r n b q k b n r\
        _ p p p p p p p\
        _ _ _ _ _ _ _ _\
        p _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _\
        P P P P P P P P\
        R _ B Q K B N R");

        let mut result = Vec::new();
        KnightOps::find_possible_moves(&mut result, &board, true, Coord::from_notation("B3"));

        let result = HashSet::from_iter(result.into_iter().map(|o| o.to));
        let expected = HashSet::from(["A5", "C5", "D4"].map(Coord::from_notation));

        assert_eq!(expected, result);
    }

    #[test]
    fn test_can_move_to() {
        let board = Board::from_literal("\
        r n b q k b n r\
        _ p p p p p p p\
        _ _ _ _ _ _ _ _\
        p _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _\
        P P P P P P P P\
        R _ B Q K B N R");
        let from = Coord::from_notation("B3");

        assert!(KnightOps::can_move_to(&board, true, from, Coord::from_notation("A5")));
        assert!(KnightOps::can_move_to(&board, true, from, Coord::from_notation("C5")));
        assert!(KnightOps::can_move_to(&board, true, from, Coord::from_notation("D4")));

        assert!(!KnightOps::can_move_to(&board, true, from, Coord::from_notation("H8")));
        assert!(!KnightOps::can_move_to(&board, true, from, Coord::from_notation("D2")));
        assert!(!KnightOps::can_move_to(&board, true, from, Coord::from_notation("A1")));
        assert!(!KnightOps::can_move_to(&board, true, from, Coord::from_notation("B3")));
        assert!(!KnightOps::can_move_to(&board, true, from, Coord::from_notation("D3")));
    }
}