mod literals;
pub mod castle_rights;

use std::fmt::{Debug, Display, Formatter};
use crate::chess::board::castle_rights::CastleRights;
use crate::chess::coord::Coord;
use crate::chess::game_result::GameResult;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_dict::PieceDict;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::side_dict::SideDict;
use crate::chess::utils::{BoardArray};
use crate::utils::Array3F;

#[derive(Eq, PartialEq, Clone)]
pub struct Board {
    pub pieces: BoardArray<BoardPiece>,
    pub en_passant_vulnerable: Option<Coord>,
    pub castle_rights: SideDict<CastleRights>,
}

impl Board {
    pub fn new() -> Self {
        use PieceType::*;
        use BoardPiece as bp;
        let pieces = [
            [bp::white(Rook), bp::white(Knight), BoardPiece::white(Bishop), bp::white(Queen), bp::white(King), bp::white(Bishop), bp::white(Knight), bp::white(Rook)],
            [bp::white(Pawn), bp::white(Pawn), bp::white(Pawn), bp::white(Pawn), bp::white(Pawn), bp::white(Pawn), bp::white(Pawn), bp::white(Pawn)],
            [bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty()],
            [bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty()],
            [bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty()],
            [bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty(), bp::empty()],
            [bp::black(Pawn), bp::black(Pawn), bp::black(Pawn), bp::black(Pawn), bp::black(Pawn), bp::black(Pawn), bp::black(Pawn), bp::black(Pawn)],
            [bp::black(Rook), bp::black(Knight), BoardPiece::black(Bishop), bp::black(Queen), bp::black(King), bp::black(Bishop), bp::black(Knight), bp::black(Rook)],
        ];

        Self {
            pieces,
            en_passant_vulnerable: None,
            castle_rights: SideDict::new(CastleRights::full(), CastleRights::full()),
        }
    }

    pub fn to_array(&self) -> Array3F {
        let mut result = Array3F::zeros((6, 8, 8));
        for row in 0..8 {
            for col in 0..8 {
                let piece = self.pieces[row][col];
                if !piece.is_empty() {
                    result[(piece.ty as usize - 1, row, col)] = if piece.side { 1.0 } else { -1.0 };
                }
            }
        }
        result
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..8 {
            for col in 0..8 {
                write!(f, "{} ", self.pieces[7 - row][col])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_board() {
        assert_eq!(Board::new(), Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R"));
    }
}
