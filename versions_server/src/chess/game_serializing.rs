use codebase::chess::game_result::{DrawReason, GameResult, WinReason};
use crate::ChessGamesPoolDep;

pub struct SerializedGame {
    pub board: String,
    pub possible: Vec<[String; 2]>,
    pub game_state: String,
    pub opening: String,
}

/// Return all the relevant information about a board to be sent to the user
pub async fn serialize_game(pool: &ChessGamesPoolDep, id: &str) -> Option<SerializedGame> {
    let pool = pool.read().await;
    let controller = pool.get_controller(id);
    controller.map(|controller| {
        let side = controller.side_to_play();
        let possible = controller.get_possible_moves(side);
        let result = controller.get_game_result(&possible);

        SerializedGame {
            board: format!("{}", controller.current()),
            possible: possible.into_iter()
                .map(|o| [format!("{}", o.from), format!("{}", o.to)])
                .collect(),
            game_state: game_state_to_string(result),
            opening: controller.get_opening_name().to_owned(),
        }
    })
}

pub fn game_state_to_string(state: GameResult) -> String {
    // These strings area also used for translation in the client side
    match state {
        GameResult::Undefined => "gameResultUndefined",
        GameResult::Draw(reason) => match reason {
            DrawReason::Aborted => "gameResultAborted",
            DrawReason::FiftyMoveRule => "gameResultFiftyMoveRule",
            DrawReason::Repetition => "gameResultRepetition",
            DrawReason::Stalemate => "gameResultStalemate",
            DrawReason::InsufficientMaterial => "gameResultInsufficientMaterial",
        },
        GameResult::Win(side, reason) => match reason {
            WinReason::Checkmate => match side {
                true => "gameResultCheckmateWhite",
                false => "gameResultCheckmateBlack",
            },
            WinReason::Timeout => match side {
                true => "gameResultTimeoutWhite",
                false => "gameResultTimeoutBlack",
            },
        }
    }.to_owned()
}