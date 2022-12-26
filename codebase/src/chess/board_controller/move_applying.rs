use crate::chess::board::castle_rights::CastleRights;
use crate::chess::board_controller::{BoardController, BoardInfo};
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::utils::CoordIndexed;

fn set_and_update(info: &mut BoardInfo, coord: Coord, piece: BoardPiece) {
    let prev = info.board.pieces.get_at(coord);
    if !prev.is_empty() {
        info.piece_counts[prev.side][prev.ty] -= 1;
    }
    info.board.pieces.set_at(coord, piece);
    if !piece.is_empty() {
        info.piece_counts[piece.side][piece.ty] += 1;
    }
}

impl BoardController {
    /// Warning: This method assumes the move is VALID
    pub fn apply_move(&mut self, m: Movement) {
        let mut info = self.current_info().clone();
        let piece_from = info.board.pieces.get_at(m.from);
        let piece_to = info.board.pieces.get_at(m.to);
        let distance = m.from.distance_2d(m.to);
        let side_moving = piece_from.side;

        set_and_update(&mut info, m.from, BoardPiece::empty());
        set_and_update(&mut info, m.to, piece_from);

        // If it is capturing en-passant
        if piece_from.ty == PieceType::Pawn && Some(m.to) == info.board.en_passant_vulnerable {
            set_and_update(&mut info, Coord::new(if side_moving { 4 } else { 3 }, m.to.col), BoardPiece::empty());
        }

        // The pawn advanced 2 squares, so it's vulnerable to en-passant
        if piece_from.ty == PieceType::Pawn && distance.row == 2 {
            info.board.en_passant_vulnerable = Some(Coord::new(if side_moving { 2 } else { 5 }, m.from.col));
        } else {
            info.board.en_passant_vulnerable = None;
        }

        // Pawn reached the end of the board
        if piece_from.ty == PieceType::Pawn && (m.to.row == 7 || m.to.row == 0) {
            // Auto-queen enabled
            set_and_update(&mut info, m.to, BoardPiece::new(PieceType::Queen, side_moving));
        }

        if piece_from.ty == PieceType::King {
        // Update kings positions
            info.kings_coords[side_moving] = m.to;

            // Castle queen side
            if m.from.col == 4 && m.to.col == 2 {
                set_and_update(&mut info, Coord::new(m.from.row, 0), BoardPiece::empty());
                set_and_update(&mut info, Coord::new(m.from.row, 3), BoardPiece::new(PieceType::Rook, side_moving));
            }

            // Castle king side
            if m.from.col == 4 && m.to.col == 6 {
                set_and_update(&mut info, Coord::new(m.from.row, 7), BoardPiece::empty());
                set_and_update(&mut info, Coord::new(m.from.row, 5), BoardPiece::new(PieceType::Rook, side_moving));
            }

            // Castle is illegal after the king moves
            info.board.castle_rights[side_moving] = CastleRights::none();
        }

        // Resets every time a pawn is moved or a piece is captured
        if !piece_to.is_empty() || piece_from.ty == PieceType::Pawn {
            self.last_50mr_reset = self.half_moves() + 1;
        }

        // Castle is illegal if that specific rook moves
        if piece_from.ty == PieceType::Rook {
            if m.from.col == 0 {
                info.board.castle_rights[side_moving].queen_side = false;
            } else if m.from.col == 7 {
                info.board.castle_rights[side_moving].king_side = false;
            }
        }
        
        // Update opening
        info.opening = info.opening.and_then(|current| self.openings.as_ref().and_then(|o| o.find_opening_move(current, m)));

        self.push(info);
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

        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::new(Coord::from_notation("B2"), Coord::from_notation("B4")));

        assert_same_board_pieces(controller.current(), "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ P _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves(), 1);
        assert_eq!(controller.last_50mr_reset, 1);
        assert_eq!(controller.current().en_passant_vulnerable, Some(Coord::from_notation("B3")));
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

        let mut controller = BoardController::new_from_single(board);
        assert_eq!(controller.current_info().piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([1, 0, 0, 0, 0, 0])));

        controller.apply_move(Movement::from_notations("B1", "C3"));
        assert_eq!(controller.current_info().piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([0, 0, 0, 0, 0, 0])));

        assert_same_board_pieces(controller.current(), "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves(), 1);
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

        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("H7", "H8"));

        assert_eq!(controller.current_info().piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 1, 1, 1]), PieceDict::new([0, 1, 0, 1, 0, 1])));
        assert_same_board_pieces(controller.current(), "\
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

        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("C2", "C4"));
        controller.apply_move(Movement::from_notations("D4", "C3"));

        assert_eq!(controller.current_info().piece_counts, SideDict::new(PieceDict::new([0, 0, 0, 0, 0, 0]), PieceDict::new([1, 0, 0, 0, 0, 0])));
        assert_same_board_pieces(controller.current(), "\
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
    fn test_castle_king_side() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        R _ _ _ K _ _ R");
        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("E1", "G1"));
        println!("{}", controller.current());
        assert_same_board_pieces(controller.current(), "\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        R _ _ _ _ R K _");
    }

    #[test]
    fn test_castle_queen_side() {
        let board = Board::from_literal("\
        r _ _ _ k _ _ r\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("E8", "C8"));
        println!("{}", controller.current());
        assert_same_board_pieces(controller.current(), "\
        _ _ k r _ _ _ r\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        _ _ _ _ _ _ _ _");
    }

    #[test]
    fn test_castle_rook_moved() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        R _ _ _ K _ _ R");
        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("A1", "C1"));
        assert!(!controller.current().castle_rights[true].queen_side);
        assert!(controller.current().castle_rights[true].king_side);
    }

    #[test]
    fn test_castle_king_moved() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ p _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ P _ _ _ _ _\
        R _ _ _ K _ _ R");
        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("E1", "D1"));
        assert!(!controller.current().castle_rights[true].queen_side);
        assert!(!controller.current().castle_rights[true].king_side);
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

        let mut controller = BoardController::new_from_single(board);
        controller.apply_move(Movement::from_notations("B1", "C3"));

        assert_same_board_pieces(controller.current(), "\
        K _ _ _ q q q _\
        K _ _ _ q _ q _\
        K K K _ q q q _\
        K _ K _ _ _ q _\
        K K K _ _ _ q _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(controller.half_moves(), 1);
        assert_eq!(controller.last_50mr_reset, 0);
    }
}
