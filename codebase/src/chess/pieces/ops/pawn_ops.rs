use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::PieceOps;
use crate::chess::utils::CoordIndexed;

struct PawnOps;

impl PieceOps for PawnOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord) {
        let direction = if side { 1 } else { -1 };

        let front1 = match from.add_checked(direction, 0) {
            Some(v) => v,
            None => panic!("Pawns can't be on row 0 (will always promote)"),
        };

        if board.pieces.get_at(front1).is_empty() {
            result.push(Movement::new(from, front1));

            // Pawns can advance 2 squares in their first moves
            // Since they can't go backwards, their first moves will always be on row 2 or 7
            let initial_row = if side { 1 } else { 6 };
            if from.row == initial_row {
                let front2 = Coord::new(if side { 3 } else { 4 }, from.col);
                if board.pieces.get_at(front2).is_empty() {
                    result.push(Movement::new(from, front2));
                }
            }
        }

        // Loop just to avoid repetition
        for x in 0..2 {
            let coord = match from.add_checked(direction, if x == 0 { 1 } else { -1 }) {
                Some(v) => v,
                None => continue,
            };
            if !coord.in_bounds() { continue; }
            let piece = board.pieces.get_at(coord);

            if Some(coord) == board.en_passant_vulnerable || (!piece.is_empty() && piece.side != side) {
                result.push(Movement::new(from, coord));
            }
        }
    }

    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool {
        let off_row = to.row as i8 - from.row as i8;
        let off_col = to.col as i8 - from.col as i8;
        let direction = if side { 1 } else { -1 };

        if off_row == direction * 2 && off_col == 0 {
            let initial_row = if side { 1 } else { 6 };
            from.row == initial_row &&
                board.pieces.get_at(from.add_checked(direction, 0).unwrap()).is_empty() &&
                board.pieces.get_at(to).is_empty()
        } else if off_row == direction && off_col == 0 {
            board.pieces.get_at(to).is_empty()
        } else if off_row == direction && off_col.abs() == 1 {
            let piece = board.pieces.get_at(to);
            board.en_passant_vulnerable == Some(to) || (!piece.is_empty() && piece.side != side)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::test_utils::assert_same_positions;
    use super::*;

    #[test]
    fn test_first_advance_possible() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");
        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E7"));

        assert_same_positions(&result, ["E6", "E5"]);
    }

    #[test]
    fn test_first_advance_blocked_possible() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ N _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");
        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E7"));

        assert_same_positions(&result, ["E6"]);
    }

    #[test]
    fn test_simple_advance_possible() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ p _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");
        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E6"));

        assert_same_positions(&result, ["E5"]);
    }

    #[test]
    fn test_capture_possible() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ p _ _ _\
        _ _ _ N _ n _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");
        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E6"));

        assert_same_positions(&result, ["E5", "D5"]);
    }

    #[test]
    fn test_capture_en_passant_possible() {
        let mut board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ P p _ _ _\
        _ _ _ _ _ _ _ _\
        P P P _ P P P P\
        R N B Q K B N R");
        board.en_passant_vulnerable = Some(Coord::from_notation("D3"));

        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E4"));

        assert_same_positions(&result, ["E3", "D3"]);
    }

    #[test]
    fn test_wrong_capture_en_passant_possible() {
        let mut board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ p _ _ _\
        _ _ _ _ _ _ _ _\
        P P _ P P P P P\
        R N B Q K B N R");
        board.en_passant_vulnerable = Some(Coord::from_notation("C3"));

        let mut result = Vec::new();
        PawnOps::find_possible_moves(&mut result, &board, false, Coord::from_notation("E4"));

        assert_same_positions(&result, ["E3"]);
    }

    // --------------------------------------------------

    #[test]
    fn test_first_advance_check() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");

        let from = Coord::from_notation("E7");
        for possible in ["E6", "E5"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["E4", "D5", "E7"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }

    #[test]
    fn test_first_advance_blocked_check() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ N _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");

        let from = Coord::from_notation("E7");
        for possible in ["E6"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["E5", "E4", "D5", "E7"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }

    #[test]
    fn test_simple_advance_check() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ p _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");

        let from = Coord::from_notation("E6");
        for possible in ["E5"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["E6", "E4", "D5", "E7"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }

    #[test]
    fn test_capture_check() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ p _ _ _\
        _ _ _ N _ n _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R");

        let from = Coord::from_notation("E6");
        for possible in ["E5", "D5"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["E4", "F5", "E7"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }

    #[test]
    fn test_capture_en_passant_check() {
        let mut board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ P p _ _ _\
        _ _ _ _ _ _ _ _\
        P P P _ P P P P\
        R N B Q K B N R");
        board.en_passant_vulnerable = Some(Coord::from_notation("D3"));

        let from = Coord::from_notation("E4");
        for possible in ["E3", "D3"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["E4", "D4", "F3"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }

    #[test]
    fn test_wrong_capture_en_passant_check() {
        let mut board = Board::from_literal("\
        r n b q k b n r\
        p p p p _ p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ p _ _ _\
        _ _ _ _ _ _ _ _\
        P P _ P P P P P\
        R N B Q K B N R");
        board.en_passant_vulnerable = Some(Coord::from_notation("C3"));

        let from = Coord::from_notation("E4");
        for possible in ["E3"].map(Coord::from_notation) {
            assert!(PawnOps::can_move_to(&board, false, from, possible))
        }
        for impossible in ["C3", "C4", "C3"].map(Coord::from_notation) {
            assert!(!PawnOps::can_move_to(&board, false, from, impossible))
        }
    }
}