use std::io::Write;
use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::building::{DecisionTreesBuilder, NextNodeStrategy};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::game_result::GameResult;
use codebase::chess::movement::Movement;
use crate::{ChessGamesPoolDep, FileManagerDep, LoadedModelDep};

type Res<T> = Result<T, String>;

pub async fn decide_and_apply(file_manager: FileManagerDep, loaded: LoadedModelDep, pool: &ChessGamesPoolDep, id: &str) -> Res<()> {
    println!("1");
    let controller = {
        let pool = pool.read().await;
        let controller = pool.get_controller(id).ok_or("Game not found")?;
        controller.clone()
    };
    println!("2");

    let chosen = decide(file_manager, loaded, controller).await?;

    println!("3");
    {
        let mut pool = pool.write().await;
        let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
        controller.apply_move(chosen);
    }
    println!("4");
    Ok(())
}

async fn decide(file_manager: FileManagerDep, loaded: LoadedModelDep, controller: BoardController) -> Res<Movement> {
    let file_manager = file_manager.read().await;
    // For now, the only criteria to evaluate the model's performance is the time spent training
    let target = file_manager.most_recent();
    let mut loaded = loaded.write().await;
    loaded.assert_loaded(target, &file_manager).map_err(|e| format!("{:?}", e))?;

    // let storage = loaded.get_loaded().unwrap().export();
    // for (k, v) in storage {
    //     print!("{}: {:?}", k, v.iter().take(20).collect::<Vec<_>>());
    // }

    let builder = DecisionTreesBuilder::new(
        vec![DecisionTree::new(controller.side_to_play())],
        vec![TreeCursor::new(controller)],
        NextNodeStrategy::BreadthFirst { min_nodes_explored: 10 },
        64,
    );
    let (mut trees, _) = builder.build(loaded.get_loaded().unwrap(), |_| {});
    let tree = trees.pop().unwrap();
    // println!("{}", tree.to_svg());
    // std::fs::OpenOptions::new().write(true).create(true).open(
    //     format!("../out/{}.svg",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("Time went backwards").as_secs()
    // )).unwrap().write_all(tree.to_svg().as_bytes()).unwrap();

    let m = tree.best_path_moves().copied().next();
    m.ok_or_else(|| "No move generated".to_owned())
}

// pub async fn decide_and_apply_old(file_manager: FileManagerDep, loaded: LoadedModelDep, pool: &ChessGamesPoolDep, id: &str) -> Res<()> {
//     let decision_data = gather_decision_data(pool, id).await;
//     let chosen = match decision_data.map_err(|e| format!("{:?}", e))? {
//         DecisionData::Theory(moves) => {
//             moves[thread_rng().gen_range(0..moves.len())]
//         }
//         DecisionData::NotTheory(moves, boards, side) => {
//             let chosen_index = ai_decide(file_manager, loaded, boards, side).await?;
//             moves[chosen_index]
//         }
//     };
//
//     {
//         let mut pool = pool.write().await;
//         let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
//         controller.apply_move(chosen);
//     }
//
//     Ok(())
// }
//
// async fn ai_decide(file_manager: FileManagerDep, loaded: LoadedModelDep, options: Array4F, side: bool) -> Res<usize>
// {
//     {
//         // Code block to free write lock asap
//         let file_manager = file_manager.read().await;
//         // For now, the only criteria to evaluate the model's performance is the time spent training
//         let target = file_manager.most_recent();
//         let mut loaded = loaded.write().await;
//         loaded.assert_loaded(target, &file_manager).map_err(|e| format!("{:?}", e))?;
//     }
//
//     let loaded = loaded.read().await;
//     let controller = loaded.get_loaded().unwrap();
//     let result = controller.eval_batch(options.into_dyn())
//         .map_err(|e| format!("{:?}", e))?;
//
//     Ok(result.into_iter()
//         .map(|o| if side { o } else { -o })
//         .enumerate()
//         .max_by(|(_, v1), (_, v2)| v1.total_cmp(v2))
//         .map(|(index, _)| index)
//         .unwrap())
// }
//
// enum DecisionData {
//     Theory(Vec<Movement>),
//     NotTheory(Vec<Movement>, Array4F, bool),
// }
//
// async fn gather_decision_data(pool: &ChessGamesPoolDep, id: &str) -> Res<DecisionData> {
//     let pool = pool.read().await;
//     let controller = pool.get_controller(id).ok_or("Game not found")?;
//
//     let continuations = controller.get_opening_continuations();
//     if !continuations.is_empty() {
//         return Ok(DecisionData::Theory(continuations));
//     }
//
//     let mut controller = controller.clone();
//
//     let side = controller.side_to_play();
//     let possible = controller.get_possible_moves(side);
//
//     let boards: Vec<_> = possible.iter()
//         .copied()
//         .map(|m| {
//             controller.apply_move(m);
//             let array = controller.current().to_array();
//             controller.revert();
//             array
//         })
//         .collect();
//     let views: Vec<_> = boards.iter().map(|o| o.view()).collect();
//     Ok(DecisionData::NotTheory(possible,
//                                stack(Axis(0), &views).map_err(|e| format!("{:?}", e))?,
//                                controller.side_to_play()))
// }