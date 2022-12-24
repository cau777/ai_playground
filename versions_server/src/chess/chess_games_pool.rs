use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use codebase::chess::board_controller::BoardController;
use codebase::chess::openings::openings_tree::OpeningsTree;
use uuid::Uuid;
use crate::EnvConfig;

pub struct ChessGamesPool {
    games: HashMap<String, BoardController>,
    openings: Arc<OpeningsTree>,
}

// TODO: expire games
impl ChessGamesPool {
    pub fn new(config: &EnvConfig) -> Self {
        let openings = OpeningsTree::load_from_file(
            OpenOptions::new().read(true).open(format!("{}/chess/openings.dat", config.base_path)).unwrap()
        ).unwrap();

        Self {
            games: HashMap::new(),
            openings: Arc::new(openings),
        }
    }

    pub fn start(&mut self) -> String {
        let uuid = Uuid::new_v4().to_string();
        let mut controller = BoardController::new_start();
        controller.add_openings_tree(self.openings.clone());
        self.games.insert(uuid.clone(), controller);
        uuid
    }
    
    pub fn get_controller(&self, id: &str) -> Option<&BoardController> {
        self.games.get(id)
    }

    pub fn get_controller_mut(&mut self, id: &str) -> Option<&mut BoardController> {
        self.games.get_mut(id)
    }
}