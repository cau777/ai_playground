use crate::chess::board::Board;
use crate::chess::movement::Movement;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::utils::CoordIndexed;

impl Board {
    pub fn apply_move(&mut self, movement: Movement) {
        let piece_from = self.pieces.get_at(movement.from);
        let piece_to = self.pieces.get_at(movement.to);

        self.pieces.set_at(movement.from, BoardPiece::empty());
        self.pieces.set_at(movement.to, piece_from);
        self.half_moves += 1;

        // TODO: en passant and promotion
        // Reduce the captured piece's count
        if !piece_to.is_empty() {
            self.piece_counts[piece_to.side][piece_to.ty] -= 1;
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
}

#[cfg(test)]
mod tests {
    use crate::chess::coord::Coord;
    use crate::chess::pieces::piece_dict::PieceDict;
    use crate::chess::side_dict::SideDict;
    use super::*;

    #[test]
    fn test_pawn_move() {
        let mut board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ P _ _ _ _ _ _\
        _ _ _ _ _ _ _ _", 0, 0);

        board.apply_move(Movement::new(Coord::from_notation("B2"), Coord::from_notation("B4")));

        assert_eq!(board, Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ P _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _", 1, 1))
    }

    #[test]
    fn test_move_with_capture() {
        let mut board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ p _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _", 0, 0);

        assert_eq!(board.piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([1, 0, 0, 0, 0, 0])));
        board.apply_move(Movement::new(Coord::from_notation("B1"), Coord::from_notation("C3")));
        assert_eq!(board.piece_counts, SideDict::new(PieceDict::new([0, 1, 0, 0, 0, 0]), PieceDict::new([0, 0, 0, 0, 0, 0])));

        assert_eq!(board, Board::from_literal("\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _", 1, 1))
    }

    #[test]
    fn test_move() {
        let mut board = Board::from_literal("\
        K _ _ _ q q q _\
        K _ _ _ q _ q _\
        K K K _ q q q _\
        K _ K _ _ _ q _\
        K K K _ _ _ q _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ N _ _ _ _ _ _", 0, 0);

        board.apply_move(Movement::new(Coord::from_notation("B1"), Coord::from_notation("C3")));

        assert_eq!(board, Board::from_literal("\
        K _ _ _ q q q _\
        K _ _ _ q _ q _\
        K K K _ q q q _\
        K _ K _ _ _ q _\
        K K K _ _ _ q _\
        _ _ N _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _", 1, 0));
    }
}
