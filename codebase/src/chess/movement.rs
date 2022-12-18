use crate::chess::coord::Coord;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Movement {
    pub from: Coord,
    pub to: Coord,
}

impl Movement {
    pub fn new(from: Coord, to: Coord) -> Self {
        Self { from, to }
    }
}