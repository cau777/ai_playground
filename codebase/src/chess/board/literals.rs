use crate::chess::board::Board;
use crate::chess::board::castle_rights::CastleRights;
use crate::chess::coord::Coord;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::side_dict::SideDict;
use crate::chess::utils::CoordIndexed;

impl Board {
    /// Literal is a representation of the game from the white's perspective
    pub fn from_literal(literal: &str) -> Self {
        let mut pieces = [[BoardPiece::empty(); 8]; 8];
        let mut index = 0;

        for c in literal.chars() {
            let c: char = c;
            if c == ' ' || c == '\r' || c == '\n' { continue; }
            let piece= BoardPiece::try_from_notation(&c.to_string()).unwrap();
            pieces[7 - (index / 8)][index % 8] =piece;

            index += 1;
        }

        let piece = pieces.get_at(Coord::from_notation("E1"));
        let white_king_in_start = piece.side && piece.ty == PieceType::King;

        let piece = pieces.get_at(Coord::from_notation("E8"));
        let black_king_in_start = !piece.side && piece.ty == PieceType::King;

        let mut castle_rights = SideDict::new(
            if white_king_in_start {CastleRights::full()} else {CastleRights::none()},
            if black_king_in_start {CastleRights::full()} else {CastleRights::none()},
        );

        // Rooks must also be in position
        let piece = pieces.get_at(Coord::from_notation("A1"));
        castle_rights[true].queen_side &= piece.side && piece.ty == PieceType::Rook;
        let piece = pieces.get_at(Coord::from_notation("H1"));
        castle_rights[true].king_side &= piece.side && piece.ty == PieceType::Rook;

        let piece = pieces.get_at(Coord::from_notation("A8"));
        castle_rights[false].queen_side &= !piece.side && piece.ty == PieceType::Rook;
        let piece = pieces.get_at(Coord::from_notation("H8"));
        castle_rights[false].king_side &= !piece.side && piece.ty == PieceType::Rook;

        Self {
            pieces,
            en_passant_vulnerable: None,
            castle_rights,
        }
    }
}
