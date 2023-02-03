use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::time::Duration;
use codebase::chess::board_controller::BoardController;
use codebase::chess::openings::openings_tree::OpeningsTree;
use tokio::time::Instant;
use uuid::Uuid;
use crate::EnvConfig;

struct GameInProgress {
    start: Instant,
    controller: BoardController,
}

pub struct ChessGamesPool {
    games: HashMap<String, GameInProgress>,
    openings: Arc<OpeningsTree>,
}

const EXPIRE_AFTER: Duration = Duration::from_secs(3600 * 2);

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

        self.games.insert(uuid.clone(), GameInProgress {
            controller,
            start: Instant::now(),
        });

        uuid
    }
    
    pub fn get_controller(&self, id: &str) -> Option<&BoardController> {
        self.games.get(id).map(|o| &o.controller)
    }

    pub fn get_controller_mut(&mut self, id: &str) -> Option<&mut BoardController> {
        self.games.get_mut(id).map(|o| &mut o.controller)
    }

    pub fn clear_expired(&mut self) {
        let now = Instant::now();
        let keys: Vec<_> = self.games.keys().cloned().collect();
        for key in keys {
            if now.duration_since(self.games[&key].start) > EXPIRE_AFTER {
                self.games.remove(&key);
            }
        }
    }
}