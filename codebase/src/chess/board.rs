use std::fmt::{Debug, Display, Formatter};
use crate::chess::game_result::GameResult;
use crate::chess::movement::Movement;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::utils::BoardArray;

#[derive(Eq, PartialEq, Clone)]
pub struct Board {
    pieces: BoardArray<BoardPiece>,
    half_moves: u16,
    moves_no_captures: u16, // Used for the 50-move rule
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
            half_moves: 0,
            moves_no_captures: 0,
        }
    }

    /// Literal is a representation of the game from the white's perspective
    pub fn from_literal(literal: &str, half_moves: u16, moves_no_captures: u16) -> Self {
        let mut pieces = [[BoardPiece::empty(); 8]; 8];
        let mut index = 0;

        for c in literal.chars() {
            let c: char = c;
            if c == ' ' || c == '\r' || c == '\n' { continue; }
            pieces[7 - (index / 8)][index % 8] = BoardPiece::try_from_notation(&c.to_string()).unwrap();
            index += 1;
        }
        Self { pieces, half_moves, moves_no_captures }
    }

    pub fn get_possible_moves(&self) -> Vec<Movement> {
        unimplemented!()
    }

    pub fn apply_move(&mut self) {
        self.half_moves += 1;
        unimplemented!()
    }

    pub fn get_game_result() -> GameResult {
        unimplemented!()
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..8 {
            for col in 0..8 {
                write!(f, "{} ", self.pieces[7 - row][col])?;
            }
            write!(f, "\n")?;
        }

        Ok(())
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}half_moves={}, moves_no_captures={}", self, self.half_moves, self.moves_no_captures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_board() {
        println!("{:?}", Board::new().pieces);
        println!("{:?}", Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R", 0, 0).pieces);

        assert_eq!(Board::new(), Board::from_literal("\
        r n b q k b n r\
        p p p p p p p p\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        P P P P P P P P\
        R N B Q K B N R", 0, 0));
    }
}
