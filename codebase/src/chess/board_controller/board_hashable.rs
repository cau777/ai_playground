use std::hash::{Hash, Hasher};
use nohash_hasher::IsEnabled;
use crate::chess::pieces::board_piece::BoardPiece;
use crate::chess::pieces::piece_type::PieceType::{Bishop, King, Pawn, Queen, Rook};
use crate::chess::utils::BoardArray;

#[derive(Clone, Debug)]
pub struct BoardHashable {
    pieces: BoardArray<BoardPiece>,
    pieces_count: u16,
}

impl BoardHashable {
    #[inline]
    pub fn new(pieces: BoardArray<BoardPiece>) -> Self {
        let mut count = 0;
        for row in &pieces {
            for piece in row {
                count += (!piece.is_empty()) as u16;
            }
        }

        Self {
            pieces,
            pieces_count: count,
        }
    }
}

impl PartialEq for BoardHashable {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.pieces_count == other.pieces_count &&
            self.pieces == other.pieces
    }
}

impl Eq for BoardHashable {}

#[inline]
fn hash1(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[7][6].is_empty()) as u8) |
        ((pieces[0][6].is_empty()) as u8) << 1 |
        ((pieces[6][1].ty == Pawn && !pieces[6][1].side) as u8) << 2 |
        ((pieces[7][5].is_empty()) as u8) << 3 |
        ((pieces[0][0].ty == Rook && pieces[0][0].side) as u8) << 4 |
        ((pieces[7][0].ty == Rook && !pieces[7][0].side) as u8) << 5 |
        ((pieces[0][0].is_empty()) as u8) << 6 |
        ((pieces[6][6].ty == Pawn && !pieces[6][6].side) as u8) << 7
}

#[inline]
fn hash2(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[7][3].is_empty()) as u8) |
        ((pieces[0][3].is_empty()) as u8) << 1 |
        ((pieces[1][2].is_empty()) as u8) << 2 |
        ((pieces[7][0].is_empty()) as u8) << 3 |
        ((pieces[1][1].ty == Pawn && pieces[1][1].side) as u8) << 4 |
        ((pieces[0][5].is_empty()) as u8) << 5 |
        ((pieces[1][5].ty == Pawn && pieces[1][5].side) as u8) << 6 |
        ((pieces[6][0].ty == Pawn && !pieces[6][0].side) as u8) << 7
}

#[inline]
fn hash3(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[1][7].ty == Pawn && pieces[1][7].side) as u8) |
        ((pieces[7][7].is_empty()) as u8) << 1 |
        ((pieces[7][4].is_empty()) as u8) << 2 |
        ((pieces[0][4].is_empty()) as u8) << 3 |
        ((pieces[6][2].is_empty()) as u8) << 4 |
        ((pieces[6][5].ty == Pawn && !pieces[6][5].side) as u8) << 5 |
        ((pieces[2][5].is_empty()) as u8) << 6 |
        ((pieces[6][7].ty == Pawn && !pieces[6][7].side) as u8) << 7
}

#[inline]
fn hash4(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[0][7].is_empty()) as u8) |
        ((pieces[7][2].is_empty()) as u8) << 1 |
        ((pieces[6][0].is_empty()) as u8) << 2 |
        ((pieces[1][6].ty == Pawn && pieces[1][6].side) as u8) << 3 |
        ((pieces[2][2].is_empty()) as u8) << 4 |
        ((pieces[6][1].is_empty()) as u8) << 5 |
        ((pieces[1][0].ty == Pawn && pieces[1][0].side) as u8) << 6 |
        ((pieces[3][3].is_empty()) as u8) << 7
}

#[inline]
fn hash5(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[7][7].ty == Rook) as u8) |
        ((pieces[3][4].is_empty()) as u8) << 1 |
        ((pieces[0][2].is_empty()) as u8) << 2 |
        ((pieces[1][7].is_empty()) as u8) << 3 |
        ((pieces[1][5].is_empty()) as u8) << 4 |
        ((pieces[1][2].ty == Pawn) as u8) << 5 |
        ((pieces[5][5].is_empty()) as u8) << 6 |
        ((pieces[1][1].is_empty()) as u8) << 7
}

#[inline]
fn hash6(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[5][2].is_empty()) as u8) |
        ((pieces[1][0].is_empty()) as u8) << 1 |
        ((pieces[0][7].ty == Rook) as u8) << 2 |
        ((pieces[6][7].is_empty()) as u8) << 3 |
        ((pieces[4][3].is_empty()) as u8) << 4 |
        ((pieces[4][4].is_empty()) as u8) << 5 |
        ((pieces[7][3].ty == Queen) as u8) << 6 |
        ((pieces[6][5].is_empty()) as u8) << 7
}

#[inline]
fn hash7(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[0][6].ty == King) as u8) |
        ((pieces[6][2].ty == Pawn) as u8) << 1 |
        ((pieces[0][3].ty == Queen) as u8) << 2 |
        ((pieces[5][4].is_empty()) as u8) << 3 |
        ((pieces[7][4].ty == King) as u8) << 4 |
        ((pieces[6][4].is_empty()) as u8) << 5 |
        ((pieces[7][6].ty == King) as u8) << 6 |
        ((pieces[0][4].ty == King) as u8) << 7
}

#[inline]
fn hash8(pieces: &BoardArray<BoardPiece>) -> u8 {
    ((pieces[7][2].ty == Bishop) as u8) |
        ((pieces[6][6].is_empty()) as u8) << 1 |
        ((pieces[3][3].ty == Pawn) as u8) << 2 |
        ((pieces[3][4].ty == Pawn) as u8) << 3 |
        ((pieces[6][3].is_empty()) as u8) << 4 |
        ((pieces[1][6].is_empty()) as u8) << 5 |
        ((pieces[5][6].is_empty()) as u8) << 6 |
        ((pieces[5][3].is_empty()) as u8) << 7
}

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for BoardHashable {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(
            (hash1(&self.pieces) as u64)  |
            (hash2(&self.pieces) as u64) << 8 |
            (hash3(&self.pieces) as u64) << 16 |
            (hash4(&self.pieces) as u64) << 24 |
            (hash5(&self.pieces) as u64) << 32 |
            (hash6(&self.pieces) as u64) << 40 |
            (hash7(&self.pieces) as u64) << 48 |
            (hash8(&self.pieces) as u64) << 56
        );
    }
}

// Required marker trait to assert that BoardRecord implements a hash method that only calls write once
impl IsEnabled for BoardHashable {}
