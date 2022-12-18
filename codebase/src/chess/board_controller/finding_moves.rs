use crate::chess::board_controller::BoardController;
use crate::chess::coord::Coord;
use crate::chess::movement::Movement;

impl BoardController {
    pub fn get_possible_moves(&self) -> Vec<Movement> {
        let mut result = Vec::new();
        for coord in Coord::board_coords() {

        }

        // result.retain() TODO: remove moves that result in checks
        result
    }
}