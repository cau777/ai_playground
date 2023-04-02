use crate::nn::loss::loss_func::LossFuncOps;
use crate::utils::ArrayDynF;

/// One of the simplest loss functions. Calculates the square of the difference between the actual
/// value and the expected one
pub struct MseLoss;

impl LossFuncOps for MseLoss {
    fn calc_loss(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        (expected - actual).mapv(|o: f32| o * o)
    }

    fn calc_loss_grad(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        expected - actual
    }
}
