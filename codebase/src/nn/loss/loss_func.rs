use crate::nn::loss::mse_loss::MseLossFunc;
use crate::utils::ArrayDynF;

pub trait LossFuncOps {
    fn calc_loss(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF;
    fn calc_loss_grad(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF;
}

#[derive(Clone, Debug)]
pub enum LossFunc {
    Mse
}

pub fn calc_loss(layer: &LossFunc, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
    use LossFunc::*;
    match layer {
        Mse => MseLossFunc::calc_loss(expected, actual)
    }
}

pub fn calc_loss_grad(layer: &LossFunc, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
    use LossFunc::*;
    match layer {
        Mse => MseLossFunc::calc_loss_grad(expected, actual)
    }
}