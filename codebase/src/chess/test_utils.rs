use std::collections::HashSet;
use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;

pub fn assert_same_positions<const S: usize>(result: &[Movement], expected: [&str; S]) {
    let result = HashSet::from_iter(result.iter().map(|o| o.to));
    let expected = HashSet::from(expected.map(Coord::from_notation));
    assert_eq!(result, expected);
}

pub fn assert_same_board_pieces(board: &Board, literal: &str) {
    assert_eq!(board.pieces, Board::from_literal(literal).pieces);
}