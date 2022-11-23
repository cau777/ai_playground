use crate::nn::loss::cross_entropy_loss::CrossEntropyLoss;
use crate::nn::loss::mse_loss::MseLoss;
use crate::utils::ArrayDynF;

pub trait LossFuncOps {
    fn calc_loss(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF;
    fn calc_loss_grad(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF;
}

#[derive(Clone, Debug)]
pub enum LossFunc {
    Mse,
    CrossEntropy
}

pub fn calc_loss(layer: &LossFunc, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
    use LossFunc::*;
    match layer {
        Mse => MseLoss::calc_loss(expected, actual),
        CrossEntropy => CrossEntropyLoss::calc_loss(expected, actual),
    }
}

pub fn calc_loss_grad(layer: &LossFunc, expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
    use LossFunc::*;
    match layer {
        Mse => MseLoss::calc_loss_grad(expected, actual),
        CrossEntropy => CrossEntropyLoss::calc_loss_grad(expected, actual),
    }
}