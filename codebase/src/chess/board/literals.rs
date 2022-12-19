use crate::chess::board::Board;
use crate::chess::pieces::board_piece::BoardPiece;

impl Board {
    /// Literal is a representation of the game from the white's perspective
    pub fn from_literal(literal: &str) -> Self {
        let mut pieces = [[BoardPiece::empty(); 8]; 8];
        let mut index = 0;

        for c in literal.chars() {
            let c: char = c;
            if c == ' ' || c == '\r' || c == '\n' { continue; }
            pieces[7 - (index / 8)][index % 8] = BoardPiece::try_from_notation(&c.to_string()).unwrap();
            index += 1;
        }
        
        Self { pieces, en_passant_vulnerable: None }
    }
}
