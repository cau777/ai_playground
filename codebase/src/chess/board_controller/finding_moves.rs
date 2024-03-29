use crate::chess::board_controller::GameController;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::ops::{piece_can_move_to, piece_find_possible_moves};
use crate::chess::utils::CoordIndexed;

fn get_queen_side_path(side: bool) -> [Coord; 3] {
    if side {
        [Coord::from_notation("B1"), Coord::from_notation("C1"), Coord::from_notation("D1")]
    } else {
        [Coord::from_notation("B8"), Coord::from_notation("C8"), Coord::from_notation("D8")]
    }
}

fn get_king_side_path(side: bool) -> [Coord; 2] {
    if side {
        [Coord::from_notation("F1"), Coord::from_notation("G1")]
    } else {
        [Coord::from_notation("F8"), Coord::from_notation("G8")]
    }
}

impl GameController {
    pub fn get_possible_moves(&self, side: bool) -> Vec<Movement> {
        let mut result = Vec::new();
        let board = self.current();

        for coord in Coord::board_coords() {
            let piece = board.pieces.get_at(coord);
            if piece.is_empty() || piece.side != side { continue; }
            piece_find_possible_moves(&mut result, board, piece.ty, piece.side, coord);
        }

        let mut controller = self.clone();
        result.retain(|m| {
            controller.apply_move(*m);
            let in_check = controller.is_in_check(side);
            controller.revert();
            !in_check
        });

        if !self.is_in_check(side) {
            let row = if side { 0 } else { 7 };
            let path = get_queen_side_path(side);
            if board.castle_rights[side].queen_side && self.path_free(&path) && !self.pieces_cover_path(!side, &path) {
                result.push(Movement::new(Coord::new(row, 4), Coord::new(row, 2)));
            }

            let path = get_king_side_path(side);
            if board.castle_rights[side].king_side && self.path_free(&path) && !self.pieces_cover_path(!side, &path) {
                result.push(Movement::new(Coord::new(row, 4), Coord::new(row, 6)));
            }
        }

        result
    }

    pub fn is_in_check(&self, side: bool) -> bool {
        let info = self.current_info();
        let board = &info.board;
        let to = info.kings_coords[side];

        for coord in Coord::board_coords() {
            let piece = board.pieces.get_at(coord);
            if piece.is_empty() || piece.side == side { continue; }
            if piece_can_move_to(board, piece.ty, piece.side, coord, to) {
                return true;
            }
        }

        false
    }

    fn path_free(&self, path: &[Coord]) -> bool {
        let board = self.current();
        for coord in path {
            if !board.pieces.get_at(*coord).is_empty() {
                return false;
            }
        }
        true
    }

    fn pieces_cover_path(&self, side: bool, path: &[Coord]) -> bool {
        let board = self.current();
        for coord in Coord::board_coords() {
            let piece = board.pieces.get_at(coord);
            if piece.is_empty() || piece.side != side { continue; }

            for to in path {
                if piece_can_move_to(board, piece.ty, piece.side, coord, *to) {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::chess::board::Board;
    use super::*;

    #[test]
    fn test_possible_moves() {
        let board = Board::from_literal("\
        _ _ _ _ _ k _ r\
        _ _ _ _ p _ b _\
        _ _ _ _ _ _ _ n\
        _ _ _ _ _ _ _ _\
        P _ _ _ p _ _ _\
        _ P _ _ _ _ _ _\
        Q _ P K _ _ _ _\
        _ N B _ _ _ _ _");
        let controller = GameController::new_from_single(board);
        let result = HashSet::from_iter(controller.get_possible_moves(true).into_iter());
        let expected = HashSet::from([
            ["A2", "A1"], ["A2", "B2"], ["A2", "A3"],
            ["B1", "A3"], ["B1", "C3"],
            ["C1", "B2"], ["C1", "A3"],
            ["D2", "D1"], ["D2", "E1"], ["D2", "E2"], ["D2", "E3"],
            ["C2", "C3"], ["C2", "C4"],
            ["B3", "B4"],
            ["A4", "A5"],
        ].map(|[from, to]| Movement::from_notations(from, to)));
        assert_eq!(result, expected);

        let result = HashSet::from_iter(controller.get_possible_moves(false).into_iter());
        let expected = HashSet::from([
            ["H8", "H7"], ["H8", "G8"],
            ["F8", "E8"], ["F8", "F7"], ["F8", "G8"],
            ["H6", "G4"], ["H6", "F5"], ["H6", "G8"], ["H6", "F7"],
            ["E7", "E6"], ["E7", "E5"],
            ["E4", "E3"],
            ["G7", "F6"], ["G7", "E5"], ["G7", "D4"], ["G7", "C3"], ["G7", "B2"], ["G7", "A1"],
        ].map(|[from, to]| Movement::from_notations(from, to)));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_castle_conditions() {
        let board = Board::from_literal("\
        r _ _ _ k b _ r\
        p B p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ q _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ Q _ _ _ P\
        P P P P P P P _\
        R _ _ _ K _ _ R");
        let mut controller = GameController::new_from_single(board);
        // Move rook to remove king side castle rights
        controller.apply_move(Movement::from_notations("H1", "H2"));

        let result = controller.get_possible_moves(true);
        assert!(result.contains(&Movement::from_notations("E1", "C1")));
        assert!(!result.contains(&Movement::from_notations("E1", "G1")));

        let result = controller.get_possible_moves(false);
        assert!(!result.contains(&Movement::from_notations("E8", "C8")));
        assert!(!result.contains(&Movement::from_notations("E8", "G8")));
    }

    #[test]
    fn test_is_in_check() {
        let board = Board::from_literal("\
        r n b q k b n r\
        p p p _ p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ P p _ _ _\
        _ _ _ _ _ _ _ _\
        P P P K _ P P P\
        R N B Q _ B N R");
        let controller = GameController::new_from_single(board);
        assert!(!controller.is_in_check(true));
        assert!(!controller.is_in_check(false));

        let board = Board::from_literal("\
        r n b _ k b n r\
        p p p _ p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ q p _ _ _\
        _ _ _ _ _ _ _ _\
        P P P K _ P P P\
        R N B Q _ B N R");
        let controller = GameController::new_from_single(board);
        assert!(controller.is_in_check(true));
        assert!(!controller.is_in_check(false));
    }
}