use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
use std::time::Instant;

use crate::nn::layers::nn_layers::{BackwardData, EmptyLayerResult, ForwardData, InitData, LayerOps, LayerResult};

pub struct DebugLayer;

lazy_static! {
    static ref START_TIME: Instant = Instant::now();
}

lazy_static! {
    static ref PREVIOUS_INSTANT: Arc<Mutex<Instant>> = Arc::new(Mutex::new(Instant::now()));
}

#[derive(Clone)]
pub enum DebugAction {
    PrintShape,
    PrintTime,
    Call(fn(tag: &str, data: &InitData, name: &str), fn(tag: &str, data: &ForwardData, name: &str), fn(tag: &str, data: &BackwardData, name: &str)),
    PrintElapsed,
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
            DebugAction::PrintTime => {}
            DebugAction::Call(f, _, _) => {
                f(&layer_config.tag, &data, &key)
            }
            DebugAction::PrintElapsed => {}
        }
        Ok(())
    }

    fn forward(data: ForwardData, layer_config: &DebugLayerConfig) -> LayerResult {
        let key = data.assigner.get_key("debug".to_owned());
        match layer_config.action {
            DebugAction::PrintShape => {
                println!("Forward:{}:Inputs_shape={:?}", layer_config.tag, data.inputs.shape())
            }
            DebugAction::PrintTime => {
                println!("Forward:{}:time={:?}", layer_config.tag, START_TIME.elapsed().as_millis())
            }
            DebugAction::PrintElapsed => {
                let mut prev = PREVIOUS_INSTANT.lock().unwrap();
                println!("Forward:{}:elapsed={:?}", layer_config.tag, prev.elapsed().as_millis());
                *prev = Instant::now();
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
                println!("Backward:{}:Grad_shape={:?}", layer_config.tag, data.grad.shape())
            }
            DebugAction::PrintTime => {
                println!("Backward:{}:time={:?}", layer_config.tag, START_TIME.elapsed().as_millis())
            }
            DebugAction::PrintElapsed => {
                let mut prev = PREVIOUS_INSTANT.lock().unwrap();
                println!("Backward:{}:time={:?}", layer_config.tag, prev.elapsed().as_millis());
                *prev = Instant::now();
            }
            DebugAction::Call(_, _, f) => {
                f(&layer_config.tag, &data, &key);
            }
        }
        Ok(data.grad.into())
    }
}