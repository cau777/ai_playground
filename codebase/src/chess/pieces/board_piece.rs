use std::fmt::{Display, Formatter};
use crate::chess::pieces::piece_type::PieceType;

#[derive(Eq, PartialEq, Debug, Copy, Clone, Hash)]
pub struct BoardPiece {
    pub side: bool,
    // True for white and False for black
    pub ty: PieceType,
}

impl Display for BoardPiece {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PieceType::*;
        let letter = match self.ty {
            Empty => '_',
            Pawn => 'p',
            Knight => 'n',
            Bishop => 'b',
            Rook => 'r',
            Queen => 'q',
            King => 'k',
        };

        write!(f, "{}", if self.side { letter.to_ascii_uppercase() } else { letter })
    }
}

impl BoardPiece {
    pub fn new(ty: PieceType, side: bool) -> Self {
        Self {
            ty,
            side: ty == PieceType::Empty || side, // For consistency and equality, makes sure that empty is always considered white
        }
    }

    pub fn white(ty: PieceType) -> Self {
        Self::new(ty, true)
    }

    pub fn black(ty: PieceType) -> Self {
        Self::new(ty, false)
    }

    pub fn empty() -> Self {
        // Side value doesn't matter
        Self::new(PieceType::Empty, true)
    }

    pub fn from_notation(notation: &str) -> Self {
        Self::try_from_notation(notation)
            .unwrap_or_else(|| panic!("Could not parse {} as BoardPiece", notation))
    }

    pub fn try_from_notation(notation: &str) -> Option<Self> {
        Some(BoardPiece::new(PieceType::try_from_notation(notation)?, notation.chars().next()?.is_ascii_uppercase()))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ty == PieceType::Empty
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::pieces::piece_type::PieceType::*;
    use super::*;

    #[test]
    fn test_try_from_notation() {
        assert_eq!(BoardPiece::try_from_notation("K"), Some(BoardPiece::white(King)));
        assert_eq!(BoardPiece::try_from_notation("n"), Some(BoardPiece::black(Knight)));
        assert_eq!(BoardPiece::try_from_notation("p"), Some(BoardPiece::black(Pawn)));

        assert_eq!(BoardPiece::try_from_notation("nothing"), None);
        assert_eq!(BoardPiece::try_from_notation("U"), None);
    }
}