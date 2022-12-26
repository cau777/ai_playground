use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::utils::CoordIndexed;

pub const DIAGONALS: [[i8; 2]; 4] = [
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1],
];

pub const LINES: [[i8; 2]; 4] = [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
];

pub fn diagonal_valid(from: Coord, to: Coord) -> bool {
    let distance = from.distance_2d(to);
    distance.row == distance.col
}

pub fn line_valid(from: Coord, to: Coord) -> bool {
    from.row == to.row || from.col == to.col
}

/// Checks if there is any piece between 2 points in a diagonal (exclusive on both ends)
pub fn piece_in_line(board: &Board, from: Coord, to: Coord) -> bool {
    // Checks are necessary to avoid an infinite loop
    if !from.in_bounds() { panic!("'from' not in bounds") }
    if !to.in_bounds() { panic!("'to' not in bounds") }
    if !line_valid(from, to) { panic!("'from' and 'to' don't form a line") }

    let inc_row = if from.row == to.row {0} else if from.row < to.row { 1 } else { -1 };
    let inc_col = if from.col == to.col {0} else if from.col < to.col { 1 } else { -1 };
    
    let mut current = from.add_checked(inc_row, inc_col);
    while let Some(value) = current {
        // Will break before checking 'to' (exclusive)
        if value == to { break; }
        let piece = board.pieces.get_at(value);
        if !piece.is_empty() { return true; }
        current = value.add_checked(inc_row, inc_col);
    }
    false
}

/// Checks if there is any piece between 2 points in a line (exclusive on both ends)
pub fn piece_in_diagonal(board: &Board, from: Coord, to: Coord) -> bool {
    // Checks are necessary to avoid an infinite loop
    if !from.in_bounds() { panic!("'from' not in bounds") }
    if !to.in_bounds() { panic!("'to' not in bounds") }
    if !diagonal_valid(from, to) { panic!("'from' and 'to' don't form a diagonal") }

    let inc_row = if from.row < to.row { 1 } else { -1 };
    let inc_col = if from.col < to.col { 1 } else { -1 };
    let mut current = from.add_checked(inc_row, inc_col);
    while let Some(value) = current {
        // Will break before checking 'to' (exclusive)
        if value == to { break; }
        let piece = board.pieces.get_at(value);
        if !piece.is_empty() { return true; }
        current = value.add_checked(inc_row, inc_col);
    }
    false
}

pub fn find_possible_moves_line(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord, off_row: i8, off_col: i8) {
    let mut current = from.add_checked(off_row, off_col);
    while let Some(value) = current {
        if !value.in_bounds() { break; }

        let piece = board.pieces.get_at(value);
        if piece.is_empty() {
            result.push(Movement::new(from, value));
        } else {
            // Can capture last piece before stopping
            if piece.side != side {
                result.push(Movement::new(from, value));
            }
            break;
        }
        current = value.add_checked(off_row, off_col);
    }
}