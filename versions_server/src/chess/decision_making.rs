use codebase::chess::movement::Movement;
use codebase::utils::{Array4F};
use codebase::utils::ndarray::{Axis, stack};
use rand::{Rng, thread_rng};
use crate::{ChessGamesPoolDep, FileManagerDep, LoadedModelDep};

type Res<T> = Result<T, String>;

// TODO: improve like training
// TODO: best ai
pub async fn decide_and_apply(file_manager: FileManagerDep, loaded: LoadedModelDep, pool: &ChessGamesPoolDep, id: &str) -> Res<()> {
    let decision_data = gather_decision_data(pool, id).await;
    let chosen = match decision_data.map_err(|e| format!("{:?}", e))? {
        DecisionData::Theory(moves) => {
            moves[thread_rng().gen_range(0..moves.len())]
        }
        DecisionData::NotTheory(moves, boards, side) => {
            let chosen_index = ai_decide(file_manager, loaded, boards, side).await?;
            moves[chosen_index]
        }
    };

    {
        let mut pool = pool.write().await;
        let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
        controller.apply_move(chosen);
    }

    Ok(())
}

async fn ai_decide(file_manager: FileManagerDep, loaded: LoadedModelDep, options: Array4F, side: bool) -> Res<usize>
{
    {
        // Code block to free write lock asap
        let file_manager = file_manager.read().await;
        let target = file_manager.best();
        let mut loaded = loaded.write().await;
        loaded.assert_loaded(target, &file_manager).map_err(|e| format!("{:?}", e))?;
    }

    let loaded = loaded.read().await;
    let controller = loaded.get_loaded().unwrap();
    let result = controller.eval_batch(options.into_dyn())
        .map_err(|e| format!("{:?}", e))?;

    Ok(result.into_iter()
        .map(|o| if side { o } else { -o })
        .enumerate()
        .max_by(|(_, v1), (_, v2)| v1.total_cmp(v2))
        .map(|(index, _)| index)
        .unwrap())
}

enum DecisionData {
    Theory(Vec<Movement>),
    NotTheory(Vec<Movement>, Array4F, bool),
}

async fn gather_decision_data(pool: &ChessGamesPoolDep, id: &str) -> Res<DecisionData> {
    let pool = pool.read().await;
    let controller = pool.get_controller(id).ok_or("Game not found")?;

    let continuations = controller.get_opening_continuations();
    if !continuations.is_empty() {
        return Ok(DecisionData::Theory(continuations));
    }

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
    Ok(DecisionData::NotTheory(possible,
                               stack(Axis(0), &views).map_err(|e| format!("{:?}", e))?,
                               controller.side_to_play()))
}