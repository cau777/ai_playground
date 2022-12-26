use crate::chess::coord::Coord;

#[derive(Eq, PartialEq, Debug, Copy, Clone, Hash)]
pub struct Movement {
    pub from: Coord,
    pub to: Coord,
}

impl Movement {
    pub fn new(from: Coord, to: Coord) -> Self {
        Self { from, to }
    }

    pub fn from_notations(from: &str, to: &str) -> Self {
        Self::new(Coord::from_notation(from), Coord::from_notation(to))
    }

    pub fn try_from_notations(from: &str, to: &str) -> Option<Self> {
        Some(Self::new(Coord::try_from_notation(from)?,
                       Coord::try_from_notation(to)?))
    }
}