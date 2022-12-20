use std::collections::HashMap;
use codebase::chess::board_controller::BoardController;
use codebase::chess::movement::Movement;
use uuid::Uuid;

pub struct ChessGamesPool {
    games: HashMap<String, BoardController>,
}

// TODO: expire games
impl ChessGamesPool {
    pub fn new() -> Self {
        Self {
            games: HashMap::new(),
        }
    }

    pub fn start(&mut self) -> String {
        let uuid = Uuid::new_v4().to_string();
        self.games.insert(uuid.clone(), BoardController::new_start());
        uuid
    }

    // pub fn apply_move(&mut self, id: &str, movement: Movement) -> Option<()> {
    //     match self.games.get_mut(id) {
    //         Some(game) => {
    //             game.apply_move(movement);
    //             Some(())
    //         }
    //         None => None,
    //     }
    // }
    // 
    // pub fn generate_possible(&self, id: &str, side: bool) -> Option<Vec<Movement>> {
    //     self.games.get(id).map(|o| o.get_possible_moves(side))
    // }
    // 
    // pub fn get_board(&self, id: &str) -> Option<String> {
    //     self.games.get(id).map(|o| format!("{}", o.current()))
    // }
    
    pub fn get_controller(&self, id: &str) -> Option<&BoardController> {
        self.games.get(id)
    }

    pub fn get_controller_mut(&mut self, id: &str) -> Option<&mut BoardController> {
        self.games.get_mut(id)
    }
}