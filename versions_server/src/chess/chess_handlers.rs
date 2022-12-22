use codebase::chess::game_result::{GameResult, DrawReason, WinReason};
use codebase::chess::movement::Movement;
use codebase::utils::{Array4F, Array5F, GenericResult};
use codebase::utils::ndarray::{Axis, stack};
use rand::{Rng, thread_rng};
use warp::{Reply, reply};
use crate::utils::{data_err_proc, EndpointResult};
use serde::{Serialize, Deserialize};
use crate::{ChessGamesPoolDep, FileManagerDep, LoadedModelDep, StatusCode};

#[derive(Deserialize)]
pub struct StartRequest {
    side: Option<bool>,
}

#[derive(Serialize)]
struct StartResponse {
    board: String,
    player_side: bool,
    game_id: String,
    possible: Vec<[String; 2]>,
    game_state: String,
}

pub async fn post_start(body: StartRequest, file_manager: FileManagerDep, loaded: LoadedModelDep, pool: ChessGamesPoolDep) -> EndpointResult<impl Reply> {
    let id = {
        let mut pool = pool.write().await;
        pool.start()
    };

    // If the side is not specified, it's random
    let player_side = body.side.unwrap_or_else(|| thread_rng().gen_bool(0.5));

    if !player_side {
        match decide_and_apply(file_manager, loaded, &pool, &id).await {
            Ok(_) => {}
            Err(e) => return data_err_proc(e, reply::json(&"")),
        }
    }

    let (board, possible, result) = match get_board_info(&pool, &id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&StartResponse {
        board,
        player_side,
        game_id: id,
        possible,
        game_state: result,
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
}

pub async fn post_move(body: MoveRequest, file_manager: FileManagerDep, loaded: LoadedModelDep, pool: ChessGamesPoolDep) -> EndpointResult<impl Reply> {
    let movement = match Movement::try_from_notations(&body.from, &body.to) {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::BAD_REQUEST)),
    };

    {
        let mut pool = pool.write().await;
        let controller = match pool.get_controller_mut(&body.game_id) {
            Some(v) => v,
            None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
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
                }), StatusCode::OK));
            }
        }
    }

    // AI makes a move
    match decide_and_apply(file_manager, loaded, &pool, &body.game_id).await {
        Ok(_) => {}
        Err(e) => return data_err_proc(e, reply::json(&"")),
    };

    let (board, possible, result) = match get_board_info(&pool, &body.game_id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&MoveResponse {
        possible,
        board,
        game_state: result,
    }), StatusCode::OK))
}

async fn decide_and_apply(file_manager: FileManagerDep, loaded: LoadedModelDep, pool: &ChessGamesPoolDep, id: &str) -> GenericResult<()> {
    let (moves, boards, side) = {
        let pool = pool.read().await;
        let controller = pool.get_controller(id).ok_or("Game not found")?;

        let mut controller = controller.clone();
        let side = controller.side_to_play();
        let possible = controller.get_possible_moves(side);

        let boards: Vec<_> = possible.iter()
            .copied()
            .map(|m| {
                controller.apply_move(m);
                let array = controller.current().to_array();
                controller.revert();
                array
            })
            .collect();
        let views: Vec<_> = boards.iter().map(|o| o.view()).collect();
        (possible, stack(Axis(0), &views)?, controller.side_to_play())
    };

    let chosen = decide_move(file_manager, loaded, boards, side).await?;
    let chosen = moves[chosen];
    {
        let mut pool = pool.write().await;
        let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
        controller.apply_move(chosen);
    }
    Ok(())
}

async fn get_board_info(pool: &ChessGamesPoolDep, id: &str) -> Option<(String, Vec<[String; 2]>, String)> {
    let pool = pool.read().await;
    let controller = pool.get_controller(id);
    controller.map(|controller| {
        let side = controller.side_to_play();
        let possible = controller.get_possible_moves(side);
        let result = controller.get_game_result(&possible);

        (
            format!("{}", controller.current()),
            possible.into_iter()
                .map(|o| [format!("{}", o.from), format!("{}", o.to)])
                .collect(),
            game_state_to_string(result),
        )
    })
}

async fn decide_move(file_manager: FileManagerDep, loaded: LoadedModelDep, options: Array4F, side: bool) -> GenericResult<usize> {
    {
        // Code block to free write lock asap
        let file_manager = file_manager.read().await;
        let target = file_manager.best();
        let mut loaded = loaded.write().await;
        loaded.assert_loaded(target, &file_manager)?;
    }

    let loaded = loaded.read().await;
    let controller = loaded.get_loaded().unwrap();
    let result = controller.eval_batch(options.into_dyn())?;

    Ok(result.into_iter()
        .map(|o| if side {o} else {-o})
        .enumerate()
        .max_by(|(_, v1), (_, v2)| v1.total_cmp(v2))
        .map(|(index, _)| index)
        .unwrap())
}

fn game_state_to_string(state: GameResult) -> String {
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