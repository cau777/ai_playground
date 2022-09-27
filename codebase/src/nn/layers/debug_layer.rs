use crate::nn::layers::nn_layers::{BackwardData, ForwardData, InitData, LayerOps};
use crate::utils::ArrayDynF;

pub struct DebugLayer{}

#[derive(Clone)]
pub struct DebugLayerConfig {
    pub tag: String,
    pub init_callback: Option<fn(tag: &str, data: &InitData, name: &str)>,
    pub forward_callback: Option<fn(tag: &str, data: &ForwardData, name: &str)>,
    pub backward_callback: Option<fn(tag: &str, data: &BackwardData, name: &str)>
}

impl LayerOps<DebugLayerConfig> for DebugLayer {
    fn init(data: InitData, layer_config: &DebugLayerConfig) {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.init_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
    }

    fn forward(data: ForwardData, layer_config: &DebugLayerConfig) -> ArrayDynF {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.forward_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
        data.inputs
    }

    fn backward(data: BackwardData, layer_config: &DebugLayerConfig) -> ArrayDynF {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.backward_callback {
            Some(func) => func(&layer_config.tag, &data, &key),
            None => {}
        }
        data.grad
    }
}