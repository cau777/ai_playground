use std::ops::{Index, IndexMut};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SideDict<T> {
    white: T,
    black: T,
}

impl<T> SideDict<T> {
    pub fn new(white: T, black: T) -> Self {
        Self { white, black }
    }
}

impl<T: Default> SideDict<T> {
    pub fn default() -> Self {
        Self::new(T::default(), T::default())
    }
}


impl<T> Index<bool> for SideDict<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: bool) -> &Self::Output {
        if index { &self.white } else { &self.black }
    }
}

impl<T> IndexMut<bool> for SideDict<T> {
    #[inline]
    fn index_mut(&mut self, index: bool) -> &mut Self::Output {
        if index { &mut self.white } else { &mut self.black }
    }
}