use codebase::chess::board_controller::BoardController;
use codebase::chess::decision_tree::building_exp_2::{BuilderOptions, DecisionTreesBuilder, LimiterFactors, NextNodeStrategy};
use codebase::chess::decision_tree::cursor::TreeCursor;
use codebase::chess::decision_tree::DecisionTree;
use codebase::chess::movement::Movement;
use crate::{ChessGamesPoolDep, FileManagerDep, LoadedModelDep};
use crate::loaded_model::assert_model_loaded;

type Res<T> = Result<T, String>;

pub async fn decide_and_apply(file_manager: FileManagerDep, loaded: LoadedModelDep, pool: &ChessGamesPoolDep, id: &str) -> Res<()> {
    let controller = {
        let pool = pool.read().await;
        let controller = pool.get_controller(id).ok_or("Game not found")?;
        controller.clone()
    };

    let chosen = decide(file_manager, loaded, controller).await?;

    {
        let mut pool = pool.write().await;
        let controller = pool.get_controller_mut(id).ok_or("Game not found")?;
        controller.apply_move(chosen);
    }
    Ok(())
}

async fn decide(file_manager: FileManagerDep, loaded: LoadedModelDep, controller: BoardController) -> Res<Movement> {
    let file_manager = file_manager.read().await;

    // For now, the only criteria to evaluate the model's performance is the time spent training
    let target = file_manager.most_recent();

    assert_model_loaded(&loaded, target, &file_manager).await
        .map_err(|e| format!("{:?}", e))?;

    let loaded = loaded.read().await;

    let builder = DecisionTreesBuilder::new(
        vec![DecisionTree::new(controller.side_to_play())],
        vec![TreeCursor::new(controller)],
        BuilderOptions {
            batch_size: 64,
            max_cache: 1_000,
            next_node_strategy: NextNodeStrategy::BestNode,
            limits: LimiterFactors {
                max_explored_nodes: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }
    );

    let (mut trees, _) = builder.build(loaded.get_loaded().unwrap());
    let tree = trees.pop().unwrap();

    // use std::io::Write;
    // std::fs::OpenOptions::new().write(true).create(true).open(
    //     format!("../out/{}.svg", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("Time went backwards").as_secs()
    //     )).unwrap().write_all(tree.to_svg().as_bytes()).unwrap();

    let m = tree.best_path_moves().copied().next();
    m.ok_or_else(|| "No move generated".to_owned())
}
