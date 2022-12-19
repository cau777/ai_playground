use crate::chess::board_controller::BoardController;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::utils::CoordIndexed;

impl BoardController {
    pub fn apply_move(&mut self, movement: Movement) {
        let piece_from = self.current.pieces.get_at(movement.from);
        let piece_to = self.current.pieces.get_at(movement.to);
        let distance = movement.from.distance_2d(movement.to);

        self.set_and_update(movement.from, BoardPiece::empty());
        self.set_and_update(movement.to, piece_from);
        self.half_moves += 1;

        // If it is capturing en-passant
        if piece_from.ty == PieceType::Pawn && Some(movement.to) == self.current.en_passant_vulnerable {
            self.set_and_update(Coord::new(if piece_to.side { 3 } else { 4 }, movement.to.col), BoardPiece::empty());
        }

        // The pawn advanced 2 squares, so it's vulnerable to en-passant
        if piece_from.ty == PieceType::Pawn && distance.row == 2 {
            self.current.en_passant_vulnerable = Some(Coord::new(if piece_from.side { 2 } else { 5 }, movement.from.col));
        } else {
            self.current.en_passant_vulnerable = None;
        }

        // Pawn reached the end of the board
        if piece_from.ty == PieceType::Pawn && (movement.to.row == 7 || movement.to.row == 0) {
            // Auto-queen enabled
            self.set_and_update(movement.to, BoardPiece::new(PieceType::Queen, piece_from.side));
        }

        // Update kings positions
        if piece_from.ty == PieceType::King {
            self.kings_coords[piece_from.side] = movement.to;
        }

        // Resets every time a pawn is moved or a piece is captured
        if !piece_to.is_empty() || piece_from.ty == PieceType::Pawn {
            self.last_50mr_reset = self.half_moves;
        }
    }

    fn set_and_update(&mut self, coord: Coord, piece: BoardPiece) {
        let prev = self.current.pieces.get_at(coord);
        if !prev.is_empty() {
            self.piece_counts[prev.side][prev.ty] -= 1;
        }
        self.current.pieces.set_at(coord, piece);
        if !piece.is_empty() {
            self.piece_counts[piece.side][piece.ty] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::board::Board;
    use crate::chess::coord::Coord;
    use crate::chess::pieces::piece_dict::PieceDict;
    use crate::chess::side_dict::SideDict;
    use crate::chess::test_utils::assert_same_board_pieces;
    use super::*;

    #[test]
    fn test_pawn_move() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ P _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");

        let mut controller = BoardController::new(board, 0, 0);
        controller.apply_move(Movement::new(Coord::from_notation("B2"), Coord::from_notation("B4")));

        assert_same_board_pieces(&controller.current, "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ P _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves, 1);
        assert_eq!(controller.last_50mr_reset, 1);
        assert_eq!(controller.current.en_passant_vulnerable, Some(Coord::from_notation("B3")));
    }

    #[test]
    fn test_move_with_capture() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ p _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _");

        let mut controller = BoardController::new(board, 0, 0);
        assert_eq!(controller.piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([1, 0, 0, 0, 0, 0])));

        controller.apply_move(Movement::new(Coord::from_notation("B1"), Coord::from_notation("C3")));
        assert_eq!(controller.piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([0, 0, 0, 0, 0, 0])));

        assert_same_board_pieces(&controller.current, "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves, 1);
        assert_eq!(controller.last_50mr_reset, 1);
    }

    #[test]
    fn test_promotion() {
        let board = Board::from_literal("\
        r n _ _ k _ _ _\
        _ _ _ _ _ _ _ P\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        R N _ _ K _ _ _");

        let mut controller = BoardController::new(board, 0, 0);
        controller.apply_move(Movement::new(Coord::from_notation("H7"), Coord::from_notation("H8")));

        assert_eq!(controller.piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 1, 1, 1]), PieceDict::new([0, 1, 0, 1, 0, 1])));
        assert_same_board_pieces(&controller.current, "\
        r n _ _ k _ _ Q\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        R N _ _ K _ _ _");
    }

    #[test]
    fn test_en_passant() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        _ _ _ _ _ _ _ _");

        let mut controller = BoardController::new(board, 0, 0);
        controller.apply_move(Movement::new(Coord::from_notation("C2"), Coord::from_notation("C4")));
        controller.apply_move(Movement::new(Coord::from_notation("D4"), Coord::from_notation("C3")));

        assert_eq!(controller.piece_counts, SideDict::new(PieceDict::new([0, 0, 0, 0, 0, 0]), PieceDict::new([1, 0, 0, 0, 0, 0])));
        assert_same_board_pieces(&controller.current, "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ p _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
    }

    #[test]
    fn test_move() {
        let board = Board::from_literal("\
        K _ _ _ q q q _\
        K _ _ _ q _ q _\
        K K K _ q q q _\
        K _ K _ _ _ q _\
        K K K _ _ _ q _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _");

        let mut controller = BoardController::new(board, 0, 0);
        controller.apply_move(Movement::new(Coord::from_notation("B1"), Coord::from_notation("C3")));

        assert_same_board_pieces(&controller.current, "\
        K _ _ _ q q q _\
        K _ _ _ q _ q _\
        K K K _ q q q _\
        K _ K _ _ _ q _\
        K K K _ _ _ q _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves, 1);
        assert_eq!(controller.last_50mr_reset, 0);
    }
}
