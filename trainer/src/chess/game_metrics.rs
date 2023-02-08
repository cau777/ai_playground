#[derive(Debug, Default, Clone)]
pub struct MetricsByBranch {
    pub aborted_rate: f64,
    pub repetition_rate: f64,
    pub draw_50mr_rate: f64,
    pub stalemate_rate: f64,
    pub insuff_material_rate: f64,

    pub white_win_rate: f64,
    pub black_win_rate: f64,
    pub draw_rate: f64,
    pub average_branch_depth: f64,
}

impl MetricsByBranch {
    pub fn scale(&mut self, factor: f64) {
        self.aborted_rate *= factor;
        self.repetition_rate *= factor;
        self.draw_50mr_rate *= factor;
        self.stalemate_rate *= factor;
        self.insuff_material_rate *= factor;
        self.white_win_rate *= factor;
        self.black_win_rate *= factor;
        self.draw_rate *= factor;
        self.average_branch_depth *= factor;
    }

    pub fn add(&mut self, rhs: &MetricsByBranch) {
        self.aborted_rate += rhs.aborted_rate;
        self.repetition_rate += rhs.repetition_rate;
        self.draw_50mr_rate += rhs.draw_50mr_rate;
        self.stalemate_rate += rhs.stalemate_rate;
        self.insuff_material_rate += rhs.insuff_material_rate;
        self.white_win_rate += rhs.white_win_rate;
        self.black_win_rate += rhs.black_win_rate;
        self.draw_rate += rhs.draw_rate;
        self.average_branch_depth += rhs.average_branch_depth;
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetricsByNode {
    pub avg_confidence: f64,
    pub avg_children: f64,
}

impl MetricsByNode {
    pub fn scale(&mut self, factor: f64) {
        self.avg_confidence *= factor;
        self.avg_children *= factor;
    }

    pub fn add(&mut self, rhs: &MetricsByNode) {
        self.avg_confidence += rhs.avg_confidence;
        self.avg_children += rhs.avg_children;
    }
}

#[derive(Debug, Default, Clone)]
pub struct GameMetrics {
    pub total_branches: u64,
    pub total_nodes: u64,
    pub branches: MetricsByBranch,
    pub nodes: MetricsByNode,
}

impl GameMetrics {
    pub fn add(&mut self, rhs: &GameMetrics) {
        self.branches.add(&rhs.branches);
        self.nodes.add(&rhs.nodes);
        self.total_nodes += rhs.total_nodes;
        self.total_branches += rhs.total_branches;
    }
}