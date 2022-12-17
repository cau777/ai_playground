use std::fmt::Display;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum PieceType {
    Empty,
    Pawn,
    Knight, // Or horsey
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub fn from_notation(notation: &str) -> Self {
        Self::try_from_notation(notation).expect(&format!("Could not parse {} as PieceType", notation))
    }

    pub fn try_from_notation(notation: &str) -> Option<Self> {
        if notation.len() == 0 {
            // Because when no letter is specified, it's implicitly a Pawn
            // Example: "E4" means Pawn to A4
            // "Empty" is used for other purposes
            return Some(PieceType::Pawn);
        }

        if notation.len() != 1 {
            return None;
        }

        let mut chars = notation.chars();
        let c: char = chars.next()?;
        use PieceType::*;
        match c.to_ascii_uppercase() {
            '_' => Some(Empty),
            'P' => Some(Pawn), // Allow for explicit notation
            'N' => Some(Knight),
            'B' => Some(Bishop),
            'R' => Some(Rook),
            'Q' => Some(Queen),
            'K' => Some(King),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem() {
        assert_eq!(std::mem::size_of::<PieceType>(), 1)
    }

    #[test]
    fn test_try_from_notation() {
        use PieceType::*;

        assert_eq!(PieceType::try_from_notation("K"), Some(King));
        assert_eq!(PieceType::try_from_notation("N"), Some(Knight));
        assert_eq!(PieceType::try_from_notation(""), Some(Pawn));

        assert_eq!(PieceType::try_from_notation("nothing"), None);
        assert_eq!(PieceType::try_from_notation("U"), None);
    }
}