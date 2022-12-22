use ndarray_rand::rand;
use ndarray_rand::rand::Rng;

pub struct RandomPicker {
    possible: Vec<usize>,
}

impl RandomPicker {
    pub fn new(len: usize) -> Self{
        Self{
            possible: (0..len).collect()
        }
    }
    
    pub fn pick(&mut self, rng:  &mut impl rand::RngCore) -> usize {
        let chosen = rng.gen_range(0..self.possible.len());
        self.possible.swap_remove(chosen);
        self.possible[chosen]
    }
}