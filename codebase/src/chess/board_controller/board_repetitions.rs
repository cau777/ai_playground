use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use nohash_hasher::{BuildNoHashHasher, IsEnabled};
use crate::chess::board_controller::board_hashable::BoardHashable;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::utils::BoardArray;

use crate::chess::pieces::piece_type::PieceType::*;

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