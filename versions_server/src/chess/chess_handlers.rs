use codebase::chess::movement::Movement;
use codebase::utils::GenericResult;
use rand::{Rng, thread_rng};
use warp::{Reply, reply};
use crate::utils::{data_err_proc, EndpointResult};
use serde::{Serialize, Deserialize};
use crate::{ChessGamesPoolDep, StatusCode};

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
}

pub async fn post_start(body: StartRequest, pool: ChessGamesPoolDep) -> EndpointResult<impl Reply> {
    let id = {
        let mut pool = pool.write().await;
        pool.start()
    };

    // If the side is not specified, it's random
    let player_side = body.side.unwrap_or_else(|| thread_rng().gen_bool(0.5));

    if !player_side {
        match decide_and_apply(&pool, &id).await {
            Ok(_) => {}
            Err(e) => return data_err_proc(e, reply::json(&"")),
        }
    }

    let (board, possible) = match get_board_info(&pool, &id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&StartResponse {
        board,
        player_side,
        game_id: id,
        possible,
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
}

pub async fn post_move(body: MoveRequest, pool: ChessGamesPoolDep) -> EndpointResult<impl Reply> {
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
    }

    // AI makes a move
    match decide_and_apply(&pool, &body.game_id).await {
        Ok(_) => {}
        Err(e) => return data_err_proc(e, reply::json(&"")),
    };

    let (board, possible) = match get_board_info(&pool, &body.game_id).await {
        Some(v) => v,
        None => return Ok(reply::with_status(reply::json(&""), StatusCode::NOT_FOUND)),
    };

    Ok(reply::with_status(reply::json(&MoveResponse {
        possible,
        board,
    }), StatusCode::OK))
}

async fn decide_and_apply(pool: &ChessGamesPoolDep, id: &str) -> GenericResult<()> {
    let possible = {
        let pool = pool.read().await;
        let controller = pool.get_controller(id).ok_or("Game not found")?;

        let side = controller.half_moves() % 2 == 0;
        controller.get_possible_moves(side)
    };

    let chosen = decide_move(possible);
    {
        let mut pool = pool.write().await;
        let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
        controller.apply_move(chosen);
    }
    Ok(())
}

async fn get_board_info(pool: &ChessGamesPoolDep, id: &str) -> Option<(String, Vec<[String; 2]>)> {
    let pool = pool.read().await;
    let controller = pool.get_controller(id);
    controller.map(|controller| {
        let side = controller.half_moves() % 2 == 0;
        (
            format!("{}", controller.current()),
            controller.get_possible_moves(side).into_iter()
                .map(|o| [format!("{}", o.from), format!("{}", o.to)])
                .collect()
        )
    })
}

fn decide_move(possible: Vec<Movement>) -> Movement {
    possible[0]
}