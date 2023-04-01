/// Stores whether a side can castle king and/or queen side
#[derive(Eq, PartialEq, Clone)]
pub struct CastleRights {
    pub queen_side: bool,
    pub king_side: bool,
}

impl CastleRights {
    pub fn full() -> Self {
        Self {
            queen_side: true,
            king_side: true,
        }
    }

    pub fn none() -> Self {
        Self {
            queen_side: false,
            king_side: false,
        }
    }
}