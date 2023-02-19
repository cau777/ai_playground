use codebase::chess::game_result::{GameResult};
use codebase::chess::movement::Movement;
use rand::{Rng, thread_rng};
use warp::{Reply, reply};
use crate::utils::{data_err_proc, EndpointResult};
use serde::{Serialize, Deserialize};
use crate::{ChessGamesPoolDep, EnvConfig, EnvConfigDep, FileManagerDep, LoadedModelDep, StatusCode};
use crate::chess::decision_making::decide_and_apply;
use crate::chess::game_serializing::{game_state_to_string, serialize_game};

#[derive(Deserialize)]
pub struct StartRequest {
    side: Option<bool>,
    openings_book: String,
}

#[derive(Serialize)]
struct StartResponse {
    board: String,
    player_side: bool,
    game_id: String,
    possible: Vec<[String; 2]>,
    game_state: String,
    opening: String,
}

pub async fn post_start(body: StartRequest, file_manager: FileManagerDep, loaded: LoadedModelDep,
                        pool: ChessGamesPoolDep, config: EnvConfigDep) -> EndpointResult<impl Reply> {
    let id = {
        let mut pool = pool.write().await;
        pool.clear_expired();
        pool.start(&body.openings_book)
    };
    let id = match id {
        Some(val) => val,
        None => return Ok(reply::with_status(reply::json(&body.openings_book), StatusCode::NOT_FOUND))
    };

    // If the side is not specified, it's random
    let player_side = body.side.unwrap_or_else(|| thread_rng().gen_bool(0.5));

    // If the AI starts as white, it makes a move
    if !player_side {
        match decide_and_apply(file_manager, loaded, &pool, &id, &config).await {
            Ok(_) => {}
            Err(e) => return data_err_proc(e, reply::json(&"")),
        }
    }

    let serialized = match serialize_game(&pool, &id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&StartResponse {
        player_side,
        game_id: id,
        possible: serialized.possible,
        board: serialized.board,
        game_state: serialized.game_state,
        opening: serialized.opening,
    }), StatusCode::OK))
}

#[derive(Deserialize)]
pub struct MoveRequest {
    game_id: String,
    from: String,
    to: String,
}

#[derive(Serialize)]
struct MoveResponse {
    board: String,
    possible: Vec<[String; 2]>,
    game_state: String,
    opening: String,
}

pub async fn post_move(body: MoveRequest, file_manager: FileManagerDep, loaded: LoadedModelDep,
                       pool: ChessGamesPoolDep, config: EnvConfigDep) -> EndpointResult<impl Reply> {
    let movement = match Movement::try_from_notations(&body.from, &body.to) {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::BAD_REQUEST)),
    };

    // Apply the player's move and return if the game is over
    {
        let mut pool = pool.write().await;
        let controller = match pool.get_controller_mut(&body.game_id) {
            Some(v) => v,
            None => return Ok(reply::with_status(reply::json(&"Board not found"), StatusCode::NOT_FOUND)),
        };
        controller.apply_move(movement);

        let possible = controller.get_possible_moves(controller.side_to_play());
        let result = controller.get_game_result(&possible);
        match result {
            GameResult::Undefined => {}
            _ => {
                return Ok(reply::with_status(reply::json(&MoveResponse {
                    possible: vec![],
                    board: format!("{}", controller.current()),
                    game_state: game_state_to_string(result),
                    opening: "".to_owned(), // This is safe to do because no opening in the database ends in checkmate/draw
                }), StatusCode::OK));
            }
        }
    }

    // AI makes a move
    match decide_and_apply(file_manager, loaded, &pool, &body.game_id, &config).await {
        Ok(_) => {}
        Err(e) => return data_err_proc(e, reply::json(&"Layers error")),
    };

    let serialized = match serialize_game(&pool, &body.game_id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&MoveResponse {
        possible: serialized.possible,
        board: serialized.board,
        game_state: serialized.game_state,
        opening: serialized.opening,
    }), StatusCode::OK))
}
