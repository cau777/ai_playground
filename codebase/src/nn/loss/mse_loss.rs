use crate::nn::loss::loss_func::LossFuncOps;
use crate::utils::ArrayDynF;

pub struct MseLossFunc {}

impl LossFuncOps for MseLossFunc {
    fn calc_loss(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        (expected - actual).mapv(|o: f32| o * o)
    }

    fn calc_loss_grad(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        expected - actual
    }
}
