use std::collections::HashMap;
use nohash_hasher::{BuildNoHashHasher};
use crate::chess::board_controller::board_hashable::BoardHashable;

#[derive(Clone, Debug)]
pub struct BoardRepetitions {
    map: HashMap<BoardHashable, u32, BuildNoHashHasher<BoardHashable>>,
}

impl BoardRepetitions {
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(BuildNoHashHasher::default()),
        }
    }

    #[inline]
    pub fn increment_rep(&mut self, record: BoardHashable) {
        match self.map.get_mut(&record) {
            Some(v) => *v += 1,
            None => {
                self.map.insert(record, 1);
            }
        }
    }

    #[inline]
    pub fn decrease_rep(&mut self, record: &BoardHashable) {
        let prev = self.map.get_mut(record).unwrap();
        if *prev == 1 {
            self.map.remove(record);
        } else {
            *prev -= 1;
        }
    }

    #[inline]
    pub fn has_repetitions(&self) -> bool {
        self.map.iter().any(|(_, v)| *v >= 3)
    }
}