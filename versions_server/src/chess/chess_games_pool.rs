use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::time::Duration;
use codebase::chess::board_controller::GameController;
use codebase::chess::openings::openings_tree::OpeningsTree;
use tokio::time::Instant;
use uuid::Uuid;
use crate::EnvConfig;

struct GameInProgress {
    start: Instant,
    controller: GameController,
}

/// Struct that keeps track of all the games in progress
pub struct ChessGamesPool {
    games: HashMap<String, GameInProgress>,
    openings_books: HashMap<&'static str, Arc<OpeningsTree>>,
}

// Games expire in 2 hours
const EXPIRE_AFTER: Duration = Duration::from_secs(3600 * 2);

impl ChessGamesPool {
    pub fn new(config: &EnvConfig) -> Self {
        let mut openings = HashMap::new();

        fn add_from_path(openings: &mut HashMap<&'static str, Arc<OpeningsTree>>, name: &'static str,
                         file_name: &str, config: &EnvConfig)  {
            openings.insert(name,
                Arc::new(
                    OpeningsTree::load_from_file(
                        OpenOptions::new().read(true).open(format!("{}/chess/{}.dat", config.base_path, file_name)).unwrap()
                    ).unwrap()
                )
            );
        }

        openings.insert("none", Arc::new(OpeningsTree::load_from_string("||").unwrap()));
        // Load all available openings books from disk
        add_from_path(&mut openings, "complete", "openings", config);
        add_from_path(&mut openings, "gambits", "openings_gambits", config);
        add_from_path(&mut openings, "e4", "openings_e4", config);
        add_from_path(&mut openings, "d4", "openings_d4", config);
        add_from_path(&mut openings, "mainlines", "openings_main", config);

        Self {
            games: HashMap::new(),
            openings_books: openings,
        }
    }

    pub fn start(&mut self, book_name: &str) -> Option<String> {
        let uuid = Uuid::new_v4().to_string();
        let mut controller = GameController::new_start();
        let book = self.openings_books.get(book_name)?.clone();
        controller.set_openings_book(book); // Add only the specified openings book

        self.games.insert(uuid.clone(), GameInProgress {
            controller,
            start: Instant::now(),
        });

        Some(uuid)
    }
    
    pub fn get_controller(&self, id: &str) -> Option<&GameController> {
        self.games.get(id).map(|o| &o.controller)
    }

    pub fn get_controller_mut(&mut self, id: &str) -> Option<&mut GameController> {
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