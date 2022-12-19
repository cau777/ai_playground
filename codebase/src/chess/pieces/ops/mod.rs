use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::pieces::piece_type::PieceType;

mod pawn_ops;
mod knight_ops;
mod bishop_ops;
mod lines;
mod rook_ops;
mod queen_ops;
mod king_ops;

pub trait PieceOps {
    fn find_possible_moves(result: &mut Vec<Movement>, board: &Board, side: bool, from: Coord);
    fn can_move_to(board: &Board, side: bool, from: Coord, to: Coord) -> bool;
}

pub fn piece_find_possible_moves(result: &mut Vec<Movement>, board: &Board, ty: PieceType, side: bool, from: Coord) {
    match ty {
        PieceType::Empty => {}
        PieceType::Pawn => pawn_ops::PawnOps::find_possible_moves(result, board, side, from),
        PieceType::Knight => knight_ops::KnightOps::find_possible_moves(result, board, side, from),
        PieceType::Bishop => bishop_ops::BishopOps::find_possible_moves(result, board, side, from),
        PieceType::Rook => rook_ops::RookOps::find_possible_moves(result, board, side, from),
        PieceType::Queen => queen_ops::QueenOps::find_possible_moves(result, board, side, from),
        PieceType::King => king_ops::KingOps::find_possible_moves(result, board, side, from),
    }
}

pub fn piece_can_move_to(board: &Board, ty: PieceType, side: bool, from: Coord, to: Coord) -> bool {
    match ty {
        PieceType::Empty => false,
        PieceType::Pawn => pawn_ops::PawnOps::can_move_to(board, side, from, to),
        PieceType::Knight => knight_ops::KnightOps::can_move_to(board, side, from, to),
        PieceType::Bishop => bishop_ops::BishopOps::can_move_to(board, side, from, to),
        PieceType::Rook => rook_ops::RookOps::can_move_to(board, side, from, to),
        PieceType::Queen => queen_ops::QueenOps::can_move_to(board, side, from, to),
        PieceType::King => king_ops::KingOps::can_move_to(board, side, from, to),
    }
}