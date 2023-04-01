mod move_applying;
mod finding_moves;
mod game_result;
mod board_repetitions;
pub mod board_hashable;

use std::sync::Arc;
use board_hashable::BoardHashable;
use crate::chess::board::Board;
use crate::chess::board_controller::board_repetitions::BoardRepetitions;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;
use crate::chess::openings::openings_tree::OpeningsTree;
use crate::chess::pieces::piece_dict::PieceDict;
use crate::chess::pieces::piece_type::PieceType;
use crate::chess::side_dict::SideDict;
use crate::chess::utils::CoordIndexed;

#[derive(Eq, PartialEq, Clone, Debug)]
struct BoardInfo {
    board: Board,
    // Opening node index in the book
    opening: Option<usize>,
    // For optimization. Counts how many of each piece are present
    piece_counts: SideDict<PieceDict<u8>>,
    // For optimization. Stores the kings positions for quicker check lookup
    kings_coords: SideDict<Coord>,
    // Stores whether the 50-move rule was just reset
    reset_50mr: bool,
}

/// Responsible for keeping track of all moves of a game, as well as apply the 50-move and 3-fold
/// repetition rules.
#[derive(Clone, Debug)]
pub struct GameController {
    board_repetitions: BoardRepetitions,
    boards: Vec<BoardInfo>,
    openings: Option<Arc<OpeningsTree>>,
}

impl GameController {
    /// Start a new game. It doesn't come with an openings book.
    pub fn new_start() -> Self {
        let mut result = Self {
            boards: Vec::new(),
            board_repetitions: BoardRepetitions::new(),
            openings: None,
        };

        result.push(BoardInfo {
            board: Board::new(),
            piece_counts: SideDict::new(PieceDict::new([8, 2, 2, 2, 1, 1]), PieceDict::new([8, 2, 2, 2, 1, 1])),
            kings_coords: SideDict::new(Coord::from_notation("E1"), Coord::from_notation("E8")),
            opening: Some(0),
            reset_50mr: true,
        });
        result
    }

    pub fn set_openings_book(&mut self, openings: Arc<OpeningsTree>) {
        self.openings = Some(openings);
    }

    /// For the current position, search the openings book for known continuations
    pub fn get_opening_continuations(&self) -> Vec<Movement> {
        self.current_info().opening.and_then(|current| {
            self.openings.as_ref()
                .map(|o| o.get_opening_continuations(current))
        }).unwrap_or_default()
    }

    /// For the current position, search the openings book for the current opening name.
    /// Return an empty string if no opening matches.
    pub fn get_opening_name(&self) -> &str {
        self.openings.as_ref().and_then(|openings| {
            // Because an opening's name is only stored in the first move that defines the opening,
            // we search for names for all played positions in reverse
            self.boards.iter().rev().filter_map(|o| o.opening).map(|o| openings.get_opening_name(o)).next()
        }).unwrap_or_default()
    }

    /// In chess, a "move" comprises both white's and black's turns.
    /// This return the total times a piece moved.
    pub fn half_moves(&self) -> usize {
        // The initial position doesn't count
        self.boards.len() - 1
    }

    /// Return the Board representing the current position
    pub fn current(&self) -> &Board {
        // We can unwrap because revert() always keeps at least one element in boards
        &self.boards.last().as_ref().unwrap().board
    }

    fn current_info(&self) -> &BoardInfo {
        // We can unwrap because revert() always keeps at least one element in boards
        self.boards.last().unwrap()
    }

    pub fn revert(&mut self) -> Option<Board> {
        if self.boards.len() > 1 {
            let removed = self.boards.pop().map(|o| o.board).unwrap();

            let record = BoardHashable::new(removed.pieces);
            self.board_repetitions.decrease_rep(&record);

            Some(removed)
        } else {
            None
        }
    }

    pub fn half_moves_since_50mr_reset(&self) -> usize {
        self.boards.iter().rev().take_while(|o| !o.reset_50mr).count()
    }

    pub fn side_to_play(&self) -> bool {
        self.half_moves() % 2 == 0
    }

    fn push(&mut self, info: BoardInfo) {
        let record = BoardHashable::new(info.board.pieces);
        self.board_repetitions.increment_rep(record);

        self.boards.push(info);
    }

    /// For testing purposes, create an instance from a position.
    #[cfg(test)]
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

        let mut result = Self {
            boards: Vec::new(),
            board_repetitions: BoardRepetitions::new(),
            openings: None,
        };
        result.push(BoardInfo {
            board,
            piece_counts: counts,
            kings_coords,
            opening: None,
            reset_50mr: true,
        });
        result
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
        let controller = GameController::new_from_single(board);

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
        let controller = GameController::new_from_single(board);

        assert_eq!(controller.current_info().kings_coords, SideDict::new(
            Coord::from_notation("B1"),
            Coord::from_notation("F4"),
        ));
    }

    #[test]
    fn test_openings() {
        let openings = OpeningsTree::load_from_string(
            "||1,2,4,11
A01 Nimzovich-Larsen Attack|b2b3|
A02 Bird's Opening|f2f4|3
A03 Bird's Opening|d7d5|
A04 Reti Opening|g1f3|5,6
A05 Reti Opening|g8f6|
A06 Reti Opening|d7d5|7,10
A07 King's Indian Attack|g2g3|8
|c7c5|9
A08 King's Indian Attack|f1g2|
A09 Reti Opening|c2c4|
A10 English|c2c4|12
A11 English, Caro-Kann Defensive System|c7c6|13
|g1f3|14
|d7d5|15
A12 English with b3|b2b3|").unwrap();
        let mut controller = GameController::new_start();
        controller.set_openings_book(Arc::new(openings));
        controller.apply_move(Movement::from_notations("C2", "C4"));
        controller.apply_move(Movement::from_notations("C7", "C6"));
        controller.apply_move(Movement::from_notations("G1", "F3"));
        controller.apply_move(Movement::from_notations("D7", "D5"));
        controller.apply_move(Movement::from_notations("B2", "B3"));
        assert_eq!(controller.current_info().opening, Some(15));
    }
}