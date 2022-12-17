use std::fmt::{Debug, Display, Formatter};

#[derive(Eq, PartialEq)]
pub struct Coord {
    row: u8,
    col: u8,
}

impl Coord {
    pub fn new(row: u8, col: u8) -> Self {
        Self { row, col }
    }

    pub fn from_notation(notation: &str) -> Self {
        Self::try_from_notation(notation).expect(&format!("Could not parse {} as Coord", notation))
    }

    pub fn try_from_notation(notation: &str) -> Option<Self> {
        if notation.len() != 2 {
            return None;
        }
        
        let mut chars = notation.chars();
        let col_char: char = chars.next()?;
        let row_char: char = chars.next()?;

        // Subtracting the first char as u16 makes A=0, B=1, C=2, D=3, ...
        let col = col_char.to_ascii_uppercase() as i16 - 'A' as i16;
        if col < 0 || col > 7 {
            return None;
        }

        let row = row_char as i16 - '1' as i16;
        if  row < 0 || row > 7 {
            return None;
        }

        Some(Self { row: row as u8, col: col as u8 })
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (self.col as u8 + 'A' as u8) as char, self.row)
    }
}

impl Debug for Coord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (self.col as u8 + 'A' as u8) as char, self.row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_from_notation() {
        assert_eq!(Coord::try_from_notation("A1"), Some(Coord::new(0, 0)));
        assert_eq!(Coord::try_from_notation("B1"), Some(Coord::new(0, 1)));
        assert_eq!(Coord::try_from_notation("H6"), Some(Coord::new(5, 7)));
        assert_eq!(Coord::try_from_notation("C5"), Some(Coord::new(4, 2)));

        assert_eq!(Coord::try_from_notation("/1"), None);
        assert_eq!(Coord::try_from_notation("A9"), None);
        assert_eq!(Coord::try_from_notation("A0"), None);
    }
}