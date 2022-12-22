use std::fmt::Debug;

use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};

pub struct DebugLayer;

#[derive(Clone)]
pub enum DebugAction {
    PrintShape,
    Call(fn(tag: &str, data: &InitData, name: &str), fn(tag: &str, data: &ForwardData, name: &str), fn(tag: &str, data: &BackwardData, name: &str)),
}

#[derive(Clone)]
pub struct DebugLayerConfig {
    pub tag: String,
    pub action: DebugAction,
}

impl Debug for DebugLayerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tag)?;
        Ok(())
    }
}

impl LayerOps<DebugLayerConfig> for DebugLayer {
    fn init(data: InitData, layer_config: &DebugLayerConfig) -> EmptyLayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.action {
            DebugAction::PrintShape => {}
            DebugAction::Call(f, _, _) => {
                f(&layer_config.tag, &data, &key)
            }
        }
        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &DebugLayerConfig) -> LayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.action {
            DebugAction::PrintShape => {
                println!("{}:Inputs_shape={:?}", layer_config.tag, data.inputs.shape())
            }
            DebugAction::Call(_, f, _) => {
                f(&layer_config.tag, &data, &key);
            }
        }
        Ok(data.inputs)
    }

    fn backward(data: BackwardData, layer_config: &DebugLayerConfig) -> LayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.action {
            DebugAction::PrintShape => {
                println!("{}:Grad_shape={:?}", layer_config.tag, data.grad.shape())
            }
            DebugAction::Call(_, _, f) => {
                f(&layer_config.tag, &data, &key);
            }
        }
        Ok(data.grad)
    }
}