use crate::nn::layers::nn_layers::{backward_layer, BackwardData, forward_layer, ForwardData, init_layer, InitData, Layer, LayerOps, train_layer, TrainableLayerOps, TrainData};
use crate::utils::{ArrayDynF};

#[derive(Clone)]
pub struct SequentialLayerConfig {
    pub layers: Vec<Layer>,
}

pub struct SequentialLayer {}

impl LayerOps<SequentialLayerConfig> for SequentialLayer {
    fn init(data: InitData, layer_config: &SequentialLayerConfig) {
        for layer in layer_config.layers.iter() {
            init_layer(layer, InitData {
                assigner: data.assigner,
                storage: data.storage,
            })
        }
    }

    fn forward(data: ForwardData, layer_config: &SequentialLayerConfig) -> ArrayDynF {
        let mut inputs = data.inputs;
        for layer in layer_config.layers.iter() {
            let data = ForwardData {
                inputs,
                assigner: data.assigner,
                forward_cache: data.forward_cache,
                storage: data.storage,
                batch_config: data.batch_config,
            };
            inputs = forward_layer(layer, data);
        }
        inputs
    }

    fn backward(data: BackwardData, layer_config: &SequentialLayerConfig) -> ArrayDynF {
        let mut grad = data.grad;
        for layer in layer_config.layers.iter().rev() {
            let data = BackwardData {
                grad,
                assigner: data.assigner,
                forward_cache: data.forward_cache,
                backward_cache: data.backward_cache,
                batch_config: data.batch_config,
                storage: data.storage,
            };
            grad = backward_layer(layer, data)
        }
        grad
    }
}

impl TrainableLayerOps<SequentialLayerConfig> for SequentialLayer {
    fn train(data: TrainData, layer_config: &SequentialLayerConfig) {
        for layer in layer_config.layers.iter() {
            let train_data = TrainData { storage: data.storage, batch_config: data.batch_config, assigner: data.assigner, backward_cache: data.backward_cache };
            train_layer(layer, train_data);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;
    use std::sync::Mutex;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::debug_layer::{DebugLayerConfig};
    use crate::nn::layers::nn_layers::{BackwardData, ForwardData, GenericStorage, InitData, Layer, LayerOps};
    use crate::nn::layers::sequential_layer::{SequentialLayer, SequentialLayerConfig};

    use lazy_static::lazy_static;

    lazy_static! {
        static ref INIT_COUNTER: Mutex<Vec<String>> = Mutex::new(Vec::new());
        static ref FORWARD_COUNTER: Mutex<Vec<String>> = Mutex::new(Vec::new());
        static ref BACKWARD_COUNTER: Mutex<Vec<String>> = Mutex::new(Vec::new());
    }

    fn debug_vec(init: Option<fn(tag: &str, data: &InitData, name: &str)>,
                 forward: Option<fn(tag: &str, data: &ForwardData, name: &str)>,
                 backward: Option<fn(tag: &str, data: &BackwardData, name: &str)>) -> Vec<Layer> {
        let mut result = Vec::new();
        for i in 0..4 {
            result.push(Layer::Debug(DebugLayerConfig {
                tag: format!("debug_{}", i),
                init_callback: init,
                forward_callback: forward,
                backward_callback: backward,
            }))
        }

        result
    }

    #[test]
    fn test_init() {
        let data = InitData {
            assigner: &mut KeyAssigner::new(),
            storage: &mut GenericStorage::new(),
        };

        INIT_COUNTER.lock().unwrap().clear();

        let config = SequentialLayerConfig {
            layers: debug_vec(Some(|name, _, _| INIT_COUNTER.lock().unwrap().push(name.to_owned())), None, None)
        };

        SequentialLayer::init(data, &config);
        assert_eq!(INIT_COUNTER.lock().unwrap().deref(), &vec!["debug_0", "debug_1", "debug_2", "debug_3"]);
    }

    #[test]
    fn test_forward() {
        let data = ForwardData {
            inputs: Default::default(),
            batch_config: &BatchConfig { epoch: 1 },
            assigner: &mut KeyAssigner::new(),
            storage: &mut GenericStorage::new(),
            forward_cache: &mut GenericStorage::new(),
        };

        FORWARD_COUNTER.lock().unwrap().clear();

        let config = SequentialLayerConfig {
            layers: debug_vec(None, Some(|name, _, _| FORWARD_COUNTER.lock().unwrap().push(name.to_owned())), None)
        };

        SequentialLayer::forward(data, &config);

        assert_eq!(FORWARD_COUNTER.lock().unwrap().deref(), &vec!["debug_0", "debug_1", "debug_2", "debug_3"]);
    }

    #[test]
    fn test_backward() {
        let data = BackwardData {
            grad: Default::default(),
            batch_config: &BatchConfig { epoch: 1 },
            assigner: &mut KeyAssigner::new(),
            storage: &mut Default::default(),
            forward_cache: &mut Default::default(),
            backward_cache: &mut Default::default(),
        };

        BACKWARD_COUNTER.lock().unwrap().clear();

        let config = SequentialLayerConfig {
            layers: debug_vec(None, None, Some(|name, _, _| BACKWARD_COUNTER.lock().unwrap().push(name.to_owned())))
        };

        SequentialLayer::backward(data, &config);

        assert_eq!(BACKWARD_COUNTER.lock().unwrap().deref(), &vec!["debug_3", "debug_2", "debug_1", "debug_0"]);
    }
}
