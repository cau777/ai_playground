use crate::utils::ArrayDynF;

pub struct MseLossFunc {}

impl MseLossFunc {
    pub fn calc_loss(&self, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        (expected - actual).mapv(|o: f32| o * o)
    }

    pub fn calc_loss_grad(&self, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        expected - actual
    }
}
