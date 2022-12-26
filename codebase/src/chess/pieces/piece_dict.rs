use std::ops::{Index, IndexMut};
use crate::chess::pieces::piece_type::PieceType;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PieceDict<T> {
    dict: [T; 6],
}

impl<T> PieceDict<T> {
    pub fn new(dict: [T; 6]) -> Self {
        Self {dict}
    }
}

impl<T: Default> PieceDict<T> {
    pub fn default() -> Self{
        Self {dict: [T::default(), T::default(), T::default(), T::default(), T::default(), T::default()]}
    }
}

impl<T> Index<PieceType> for PieceDict<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: PieceType) -> &Self::Output {
        &self.dict[index as usize - 1]
    }
}

impl<T> IndexMut<PieceType> for PieceDict<T> {
    #[inline]
    fn index_mut(&mut self, index: PieceType) -> &mut Self::Output {
        &mut self.dict[index as usize - 1]
    }
}