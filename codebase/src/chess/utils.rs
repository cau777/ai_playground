use crate::chess::coord::Coord;

pub type BoardArray<T> = [[T; 8]; 8];

pub trait CoordIndexed<T> {
    fn get_at(&self, coord: Coord) -> T;
    fn set_at(&mut self, coord: Coord, val: T);
}

impl<T: Copy> CoordIndexed<T> for BoardArray<T> {
    fn get_at(&self, coord: Coord) -> T {
        self[coord.row as usize][coord.col as usize]
    }

    fn set_at(&mut self, coord: Coord, val: T) {
        self[coord.row as usize][coord.col as usize] = val;
    }
}
