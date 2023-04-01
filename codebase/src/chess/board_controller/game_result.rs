use crate::chess::board_controller::GameController;
use crate::chess::game_result::{DrawReason, GameResult, WinReason};
use crate::chess::movement::Movement;
use crate::chess::pieces::piece_dict::PieceDict;
use crate::chess::pieces::piece_type::PieceType;

fn is_insufficient_material(counts: &PieceDict<u8>) -> bool {
    if !(counts[PieceType::Pawn] == 0 &&
        counts[PieceType::Queen] == 0 &&
        counts[PieceType::Rook] == 0) {
        false
    } else if counts[PieceType::Bishop] == 0 {
        counts[PieceType::Knight] < 3
    } else if counts[PieceType::Knight] == 0 {
        counts[PieceType::Bishop] < 2
    } else {
        counts[PieceType::Knight] + counts[PieceType::Bishop] < 2
    }
}

impl GameController {
    pub fn get_game_result(&self, possible_moves: &Vec<Movement>) -> GameResult {
        let side = self.side_to_play();
        let info = self.current_info();

        if possible_moves.is_empty() {
            if self.is_in_check(side) {
                GameResult::Win(!side, WinReason::Checkmate)
            } else {
                GameResult::Draw(DrawReason::Stalemate)
            }
        } else if is_insufficient_material(&info.piece_counts[true]) &&
            is_insufficient_material(&info.piece_counts[false]) {
            GameResult::Draw(DrawReason::InsufficientMaterial)
            // 50-move rule corresponds to 50 FULL moves
        } else if self.half_moves_since_50mr_reset() >= 100 {
            GameResult::Draw(DrawReason::FiftyMoveRule)
            // Check if a position was repeated 3 times 
        } else if self.board_repetitions.has_repetitions() {
            GameResult::Draw(DrawReason::Repetition)
        } else if self.half_moves() > 400 {
            // Internal rule: a game can't last more than 200 FULL moves
            GameResult::Draw(DrawReason::Aborted)
        } else {
            // Game will play on
            GameResult::Undefined
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_insufficient_material() {
        // If there is a pawn, rook or queen, the game continues
        assert!(!is_insufficient_material(&PieceDict::new([1, 0, 0, 0, 0, 1])));
        assert!(!is_insufficient_material(&PieceDict::new([0, 0, 0, 0, 1, 1])));
        assert!(!is_insufficient_material(&PieceDict::new([0, 0, 0, 1, 0, 1])));

        // A lone king is a draw
        assert!(is_insufficient_material(&PieceDict::new([0, 0, 0, 0, 0, 1])));

        // 2 knights is a draw (for simplicity)
        assert!(is_insufficient_material(&PieceDict::new([0, 2, 0, 0, 0, 1])));

        // A single bishop/knight is a draw
        assert!(is_insufficient_material(&PieceDict::new([0, 1, 0, 0, 0, 1])));
        assert!(is_insufficient_material(&PieceDict::new([0, 0, 1, 0, 0, 1])));

        // Bishop + Knight is a win
        assert!(!is_insufficient_material(&PieceDict::new([0, 1, 1, 0, 0, 1])));
        assert!(!is_insufficient_material(&PieceDict::new([0, 2, 1, 0, 0, 1])));
    }
}