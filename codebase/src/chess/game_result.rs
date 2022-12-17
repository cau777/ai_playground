pub enum WinReason {
    Checkmate,
    Timeout,
}

pub enum DrawReason {
    Stalemate,
    InsufficientMaterial,
    FiftyMoveRule,
    Aborted,
}

pub enum GameResult {
    Undefined,
    BlackWon(WinReason),
    WhiteWon(WinReason),
    Draw(DrawReason),
}

impl GameResult {
    pub fn value(&self) -> Option<f32> {
        use GameResult::*;
        match self {
            Undefined => None,
            BlackWon(_) => Some(-1.0),
            WhiteWon(_) => Some(1.0),
            Draw(_) => Some(0.0),
        }
    }
}