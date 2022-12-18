use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_dict::PieceDict;
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
            pieces[7 - (index / 8)][index % 8] = BoardPiece::try_from_notation(&c.to_string()).unwrap();
            index += 1;
        }

        let mut counts = SideDict::new(PieceDict::default(), PieceDict::default());
        let mut kings_coords = SideDict::new(Coord::default(), Coord::default());
        for coord in Coord::board_coords() {
            let piece = pieces.get_at(coord);
            if piece.ty != PieceType::Empty {
                counts[piece.side][piece.ty] += 1;
                if piece.ty == PieceType::King {
                    kings_coords[piece.side] = coord;
                }
            }
        }
        Self { pieces, piece_counts: counts, kings_coords }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_count() {
        let board = Board::from_literal("\
        K k _ _ _ _ _ _\
        Q q _ _ _ _ _ _\
        R r r _ _ _ _ _\
        N N _ _ _ _ _ _\
        B B B b _ _ _ _\
        p p _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        assert_eq!(board.piece_counts, SideDict::new(
            PieceDict::new([0, 2, 3, 1, 1, 1]),
            PieceDict::new([2, 0, 1, 2, 1, 1])
        ));
    }

    #[test]
    fn test_kings_pos() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        Q q _ _ _ _ _ _\
        R r r _ _ _ _ _\
        N N _ _ _ _ _ _\
        B B B b _ k _ _\
        p p _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ K _ _ _ _ _ _");
        assert_eq!(board.kings_coords, SideDict::new(
            Coord::from_notation("B1"),
            Coord::from_notation("F4")
        ));
    }
}