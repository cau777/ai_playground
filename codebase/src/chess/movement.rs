use crate::chess::coord::Coord;

#[derive(Eq, PartialEq, Debug)]
pub struct Movement {
    from: Coord,
    to: Coord,
}

impl Movement {
    pub fn new(from: Coord, to: Coord) -> Self {
        Self { from, to }
    }
}