use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::Wrapping;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::utils::BoardArray;

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct BoardRecord {
    pub pieces: BoardArray<BoardPiece>,
}

#[inline]
fn hash_row(row: &[BoardPiece; 8]) -> u8 {
    (
        Wrapping(row[0].ty as u8) ^
        Wrapping(row[1].ty as u8) << 1 ^
        Wrapping(row[2].ty as u8) << 2 ^
        Wrapping(row[3].ty as u8) << 3 ^
        Wrapping(row[4].ty as u8) << 4 ^
        Wrapping(row[5].ty as u8) << 5 ^
        Wrapping(row[6].ty as u8) << 6 ^
        Wrapping(row[7].ty as u8) << 7
    ).0
}

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for BoardRecord {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // This is definitely not the most efficient hashing algorithm but it's very fast
        // TODO: better algorithm
        state.write_u64(
            (hash_row(&self.pieces[0]) as u64) |
                (hash_row(&self.pieces[1]) as u64) << 8 |
                (hash_row(&self.pieces[2]) as u64) << 16 |
                (hash_row(&self.pieces[3]) as u64) << 24 |
                (hash_row(&self.pieces[4]) as u64) << 32 |
                (hash_row(&self.pieces[5]) as u64) << 40 |
                (hash_row(&self.pieces[6]) as u64) << 48 |
                (hash_row(&self.pieces[7]) as u64) << 56
        );
    }
}

#[derive(Clone, Debug)]
pub struct BoardRepetitions {
    map: HashMap<BoardRecord, u32>,
}

impl BoardRepetitions {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    #[inline]
    pub fn increment_rep(&mut self, record: BoardRecord) {
        match self.map.get_mut(&record) {
            Some(v) => *v += 1,
            None => {
                self.map.insert(record, 1);
            }
        }
    }

    #[inline]
    pub fn decrease_rep(&mut self, record: &BoardRecord) {
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