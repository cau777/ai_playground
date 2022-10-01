use crate::nn::batch_config::BatchConfig;
use crate::nn::key_assigner::KeyAssigner;
use crate::nn::layers::nn_layers::{GenericStorage, LayerResult, TrainData};
use crate::nn::lr_calculators::adam_lr::{AdamConfig, AdamLrCalc};
use crate::nn::lr_calculators::constant_lr::{ConstantLr, ConstantLrConfig};
use crate::utils::ArrayDynF;

pub struct LrCalcData<'a> {
    pub batch_config: &'a BatchConfig,
    pub assigner: &'a mut KeyAssigner,
    pub storage: &'a mut GenericStorage
}

impl<'a> LrCalcData<'a> {
    pub fn from_train_data(data: &'a mut TrainData) -> Self {
        Self {
            storage: data.storage,
            batch_config: data.batch_config,
            assigner: data.assigner
        }
    }
}

#[derive(Clone)]
pub enum LrCalc {
    Constant(ConstantLrConfig),
    Adam(AdamConfig)
}

pub trait LrCalcOps<T> {
    fn apply(target: ArrayDynF, data: LrCalcData, config: &T) -> LayerResult;
}

pub fn apply_lr_calc(calc: &LrCalc, target: ArrayDynF, data: LrCalcData) -> LayerResult {
    match calc {
        LrCalc::Constant(c) => ConstantLr::apply(target, data, c),
        LrCalc::Adam(c) => AdamLrCalc::apply(target, data, c)
    }
}