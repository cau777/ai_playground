pub enum WinReason {
    Checkmate,
    Timeout,
}

pub enum DrawReason {
    Stalemate,
    InsufficientMaterial,
    FiftyMoveRule,
    Aborted,
    Repetition,
}

pub enum GameResult {
    Undefined,
    Win(bool, WinReason),
    Draw(DrawReason),
}

impl GameResult {
    pub fn value(&self) -> Option<f32> {
        use GameResult::*;
        match self {
            Undefined => None,
            Win(side, _) => Some(if *side {1.0} else {-1.0}),
            Draw(_) => Some(0.0),
        }
    }
}