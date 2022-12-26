use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::PieceOps;
use crate::chess::utils::CoordIndexed;

const SQUARE: [[i8; 2]; 8] = [
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
];

pub struct KingOps;

impl PieceOps for KingOps {
    // Castle is done later
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        for [off_row, off_col] in SQUARE {
            let coord = match from.add_checked(off_row, off_col) {
                Some(v) => v,
                None => continue,
            };
            if !coord.in_bounds() { continue; }
            let piece = board.pieces.get_at(coord);
            if piece.is_empty() || piece.side != side {
                result.push(Movement::new(from, coord));
            }
        }
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        let distance = from.distance_2d(to);
        to.in_bounds() && distance.row <= 1 && distance.col <= 1 && {
            let piece = board.pieces.get_at(to);
            piece.is_empty() || piece.side != side
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::test_utils::assert_same_positions;
    use super::*;

    #[test]
    fn test_possible_moves() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ K\
        _ _ _ _ _ _ P _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P _ P\
        R N B Q _ B N R");
        let mut result = Vec::new();
        KingOps::find_possible_moves(&mut result, &board, true, Coord::from_notation("H6"));
        assert_same_positions(&result, ["G7", "H7", "G6", "H5"]);
    }

    #[test]
    fn test_can_move() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ K\
        _ _ _ _ _ _ P _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P _ P\
        R N B Q _ B N R");

        let from = Coord::from_notation("H6");
        for possible in ["G7", "H7", "G6", "H5"].map(Coord::from_notation) {
            assert!(KingOps::can_move_to(&board, true, from, possible))
        }
        for impossible in ["G5", "H6", "D5", "E7"].map(Coord::from_notation) {
            assert!(!KingOps::can_move_to(&board, true, from, impossible))
        }
    }
}