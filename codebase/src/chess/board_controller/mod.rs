mod move_applying;
mod finding_moves;

use crate::chess::board::Board;
use crate::chess::coord::Coord;
use crate::chess::game_result::GameResult;
use crate::chess::pieces::piece_dict::PieceDict;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::side_dict::SideDict;
use crate::chess::utils::CoordIndexed;

#[derive(Eq, PartialEq, Clone)]
struct BoardInfo {
    board: Board,
    piece_counts: SideDict<PieceDict<u8>>,
    kings_coords: SideDict<Coord>,
}

#[derive(Eq, PartialEq, Clone)]
pub struct BoardController {
    // TODO: 3-fold repetition
    boards: Vec<BoardInfo>,
    last_50mr_reset: usize,
}

impl BoardController {
    pub fn new_start() -> Self {
        let mut result = Self {
            last_50mr_reset: 0,
            boards: Vec::new(),
        };
        result.push(BoardInfo {
            board: Board::new(),
            piece_counts: SideDict::new(PieceDict::new([8, 2, 2, 2, 1, 1]), PieceDict::new([8, 2, 2, 2, 1, 1])),
            kings_coords: SideDict::new(Coord::from_notation("E1"), Coord::from_notation("E8")),
        });
        result
    }

    pub fn new_from_single(board: Board) -> Self {
        let mut counts = SideDict::new(PieceDict::default(), PieceDict::default());
        let mut kings_coords = SideDict::new(Coord::default(), Coord::default());
        for coord in Coord::board_coords() {
            let piece = board.pieces.get_at(coord);
            if piece.ty != PieceType::Empty {
                counts[piece.side][piece.ty] += 1;
                if piece.ty == PieceType::King {
                    kings_coords[piece.side] = coord;
                }
            }
        }

        let mut result = Self { boards: Vec::new(), last_50mr_reset: 0 };
        result.push(BoardInfo {
            board,
            piece_counts: counts,
            kings_coords,
        });
        result
    }

    pub fn get_game_result(&self) -> GameResult {
        todo!()
    }

    fn push(&mut self, info: BoardInfo) {
        self.boards.push(info);
    }

    pub fn half_moves(&self) -> usize {
        self.boards.len() - 1
    }

    pub fn current(&self) -> &Board {
        &self.boards.last().as_ref().unwrap().board
    }

    fn current_info(&self) -> &BoardInfo {
        self.boards.last().unwrap()
    }

    pub fn revert(&mut self) -> Option<Board> {
        if self.boards.len() > 1 {
            self.boards.pop().map(|o| o.board)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_count() {
        let board = Board::from_literal("\
        K k _ _ _ _ _ _\
        Q q _ _ _ _ _ _\
        R r r _ _ _ _ _\
        N N _ _ _ _ _ _\
        B B B b _ _ _ _\
        p p _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ _ _ _ _ _ _ _");
        let controller = BoardController::new_from_single(board);

        assert_eq!(controller.current_info().piece_counts, SideDict::new(
            PieceDict::new([0, 2, 3, 1, 1, 1]),
            PieceDict::new([2, 0, 1, 2, 1, 1]),
        ));
    }

    #[test]
    fn test_kings_pos() {
        let board = Board::from_literal("\
        _ _ _ _ _ _ _ _\
        Q q _ _ _ _ _ _\
        R r r _ _ _ _ _\
        N N _ _ _ _ _ _\
        B B B b _ k _ _\
        p p _ _ _ _ _ _\
        _ _ _ _ _ _ _ _\
        _ K _ _ _ _ _ _");
        let controller = BoardController::new_from_single(board);

        assert_eq!(controller.current_info().kings_coords, SideDict::new(
            Coord::from_notation("B1"),
            Coord::from_notation("F4"),
        ));
    }
}