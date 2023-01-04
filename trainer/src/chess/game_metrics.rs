#[derive(Debug, Default, Clone)]
pub struct GameMetrics {
    pub total_branches: u64,
    pub total_nodes: u64,

    pub aborted_rate: f64,
    pub repetition_rate: f64,
    pub draw_50mr_rate: f64,
    pub stalemate_rate: f64,
    pub insuff_material_rate: f64,

    pub white_win_rate: f64,
    pub black_win_rate: f64,
    pub draw_rate: f64,
    pub average_len: f64,
}

impl GameMetrics {
    pub fn rescale_by_branches(&mut self) {
        let factor = 1.0 / self.total_branches as f64;
        self.rescale(factor);
    }

    pub fn rescale(&mut self, factor: f64) {
        self.aborted_rate *= factor;
        self.repetition_rate *= factor;
        self.draw_50mr_rate *= factor;
        self.stalemate_rate *= factor;
        self.insuff_material_rate *= factor;
        self.white_win_rate *= factor;
        self.black_win_rate *= factor;
        self.draw_rate *= factor;
        self.average_len *= factor;
    }

    pub fn add(&mut self, rhs: &GameMetrics) {
        self.aborted_rate += rhs.aborted_rate;
        self.repetition_rate += rhs.repetition_rate;
        self.draw_50mr_rate += rhs.draw_50mr_rate;
        self.stalemate_rate += rhs.stalemate_rate;
        self.insuff_material_rate += rhs.insuff_material_rate;
        self.white_win_rate += rhs.white_win_rate;
        self.black_win_rate += rhs.black_win_rate;
        self.draw_rate += rhs.draw_rate;
        self.average_len += rhs.average_len;
    }
}