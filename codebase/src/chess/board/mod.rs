mod move_applying;
mod literals;

use std::fmt::{Debug, Display, Formatter};
use crate::chess::coord::Coord;
use crate::chess::game_result::GameResult;
use crate::chess::movement::Movement;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_dict::PieceDict;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::side_dict::SideDict;
use crate::chess::utils::{BoardArray, CoordIndexed};

#[derive(Eq, PartialEq, Clone)]
pub struct Board {
    pub pieces: BoardArray<BoardPiece>,
    // Moves until 50-move rule
    pub piece_counts: SideDict<PieceDict<u8>>,
    pub kings_coords: SideDict<Coord>,
    // TODO: 3 fold repetition
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
            piece_counts: SideDict::new(PieceDict::new([8, 2, 2, 2, 1, 1]), PieceDict::new([8, 2, 2, 2, 1, 1])),
            kings_coords: SideDict::new(Coord::from_notation("E1"), Coord::from_notation("E8")),
        }
    }



    pub fn is_valid(&self) -> bool {
        unimplemented!()
    }

    pub fn get_possible_moves(&self) -> Vec<Movement> {
        unimplemented!()
    }

    pub fn get_game_result(&self) -> GameResult {
        unimplemented!()
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
        write!(f, "{} piece_counts={:?}, kings_coords={:?}", self, self.piece_counts, self.kings_coords)
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
