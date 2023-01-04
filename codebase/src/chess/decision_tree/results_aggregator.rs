use std::iter::zip;
use crate::ArrayDynF;
use crate::chess::decision_tree::NodeExtraInfo;
use crate::chess::movement::Movement;

pub struct ResultsAggregator {
    pub owner: usize,
    count: usize,
    target_count: usize,
    moves: Vec<Movement>,
    arrays: Vec<Option<ArrayDynF>>,
    buffer: Vec<(f32, NodeExtraInfo)>
}

impl ResultsAggregator {
    pub fn new(owner: usize, target_count: usize) -> Self {
        Self{
            owner,
            count: 0,
            target_count,
            moves: Vec::with_capacity(target_count),
            arrays: Vec::with_capacity(target_count),
            buffer: Vec::with_capacity(target_count),
        }
    }

    pub fn push(&mut self, m: Movement, arr: Option<ArrayDynF>) -> usize {
        self.moves.push(m);
        self.arrays.push(arr);
        self.buffer.push(Default::default());
        self.moves.len() - 1
    }

    pub fn requests_to_eval(&self) -> impl Iterator<Item=(usize, &ArrayDynF)> {
        self.arrays.iter()
            .enumerate()
            .filter_map(|(i, o)| o.as_ref().map(|o| (i, o)))
    }

    pub fn submit(&mut self, index: usize, value: f32, is_ending: bool, is_opening: bool) {
        self.count += 1;
        self.arrays[index] = None;
        self.buffer[index] = (value, NodeExtraInfo{is_ending, is_opening });
    }

    pub fn is_ready(&self) -> bool {
        self.count >= self.target_count
    }

    pub fn arrange(&self) -> Vec<(Movement, f32, NodeExtraInfo)> {
        zip(&self.moves, &self.buffer)
            .map(|(m, (e, info))| (*m, *e, *info))
            .collect()
    }
}