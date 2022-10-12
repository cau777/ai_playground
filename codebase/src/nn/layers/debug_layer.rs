use std::fmt::Debug;

use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};

pub struct DebugLayer{}

#[derive(Clone)]
pub struct DebugLayerConfig {
    pub tag: String,
    pub init_callback: Option<fn(tag: &str, data: &InitData, name: &str)>,
    pub forward_callback: Option<fn(tag: &str, data: &ForwardData, name: &str)>,
    pub backward_callback: Option<fn(tag: &str, data: &BackwardData, name: &str)>
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
        match layer_config.init_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &DebugLayerConfig) -> LayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.forward_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
        Ok(data.inputs)
    }

    fn backward(data: BackwardData, layer_config: &DebugLayerConfig) -> LayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.backward_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
        Ok(data.grad)
    }
}