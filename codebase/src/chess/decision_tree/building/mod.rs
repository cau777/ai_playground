mod options;
mod games_producer;
mod request;
mod nodes_in_progress_set;
mod limiting_factors;
mod request_storage;
mod cache;
mod building_error;

use std::cell::RefCell;
use std::iter::zip;
use ndarray::{Axis, stack};
use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::{DecisionTree, NodeExtraInfo};

pub use options::*;
pub use limiting_factors::*;
use crate::chess::decision_tree::building::building_error::BuildingError;
use crate::chess::decision_tree::building::cache::Cache;
use crate::chess::decision_tree::building::games_producer::GamesProducer;
use crate::chess::decision_tree::building::request::{RequestPart};
use crate::chess::decision_tree::building::request_storage::RequestStorage;
use crate::chess::game_result::GameResult;
use crate::nn::controller::NNController;
use crate::nn::generic_storage::{combine_storages, split_storages};
use crate::nn::layers::nn_layers::GenericStorage;

type OnGameResultFn = Box<dyn Fn((GameResult, usize))>;

fn scale_output(x: f32) -> f32 {
    if x < 2.0 && x > -2.0 {
        x
    } else {
        4.5 / (1.0 + f32::exp(-1.4 * x)) - 2.25
    }
}

pub struct DecisionTreesBuilder {
    initial_trees: Vec<DecisionTree>,
    initial_cursors: Vec<TreeCursor>,
    options: BuilderOptions,
}

impl DecisionTreesBuilder {
    pub fn new(initial_trees: Vec<DecisionTree>, initial_cursors: Vec<TreeCursor>, options: BuilderOptions) -> Self {
        Self {
            initial_trees,
            initial_cursors,
            options,
        }
    }

    pub fn build(&self, controller: &NNController) -> (Vec<DecisionTree>, Vec<TreeCursor>) {
        let trees: Vec<_> = self.initial_trees.iter().cloned().map(RefCell::new).collect();
        let cursors: Vec<_> = self.initial_cursors.iter().cloned().map(RefCell::new).collect();
        let mut producer = GamesProducer::new(&self.initial_trees, &self.initial_cursors,
                                              &trees, &cursors, &self.options);
        let mut caches = vec![Cache::new(self.options.max_cache_bytes / (trees.len() as u64)); self.initial_trees.len()];

        let mut requests = RequestStorage::new();

        let mut current_factors = CurrentLimitingFactors::default();
        let mut should_break = false;
        let mut comb = (0, 0);

        loop {
            should_break |= !self.options.limits.should_continue(current_factors.clone());
            if !should_break {
                match self.fill_requests(&mut requests, &mut producer) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{:?}", e);
                        break;
                    }
                }
            }

            let parts = Self::take_parts(&requests, self.options.batch_size);

            if !parts.is_empty() {
                // {
                //     let r = requests.get(parts[0].owner());
                //     println!("{:?} {:?}", r.game_index, r.node_index);
                // println!("{:?}", &trees[r.game_index].borrow().nodes[r.node_index]);
                // }

                let inputs: Vec<_> = parts.iter()
                    .map(|o| {
                        if let RequestPart::Pending { array, .. } = o { array.view() } else { panic!("RequestPart should be Pending") }
                    })
                    .collect();
                let inputs = stack(Axis(0), &inputs).unwrap();

                let combined = Self::prepare_cache(&mut caches, &parts, &requests);
                match &combined {
                    Some(_) => comb.0 += 1,
                    None => comb.1 += 1,
                }

                let (output, storage) = controller.eval_with_cache(inputs, combined).unwrap();
                let split = split_storages(storage, parts.len())
                    .expect("Should split cache");

                let completed_requests: Vec<_> = zip(
                    output.outer_iter()
                        // Get the only value in the array
                        .map(|o| *o.first().unwrap())
                        .map(scale_output),
                    split.into_iter(),
                )
                    .enumerate()
                    .map(|(i, (val, split))| (i, val, split))
                    .map(|(i, val, split)| {
                        let part = &parts[i];
                        let index_in_owner = part.index_in_owner();
                        let owner = part.owner();

                        RequestPart::Completed {
                            m: part.movement(),
                            owner,
                            index_in_owner,
                            info: NodeExtraInfo { is_opening: false, is_ending: false },
                            eval: val,
                            cache: Some(split),
                        }
                    }).collect();

                for c in completed_requests {
                    let index_in_owner = c.index_in_owner();
                    let target = requests.get_mut(c.owner());

                    target.parts[index_in_owner] = c;
                }
            }

            let prev_len = requests.len();
            requests = self.remove_completed(requests, &trees, &mut caches);

            let completed = prev_len - requests.len();
            current_factors.curr_explored_nodes += completed;
            current_factors.curr_iterations += 1;

            // Do it every 3 iterations just to save some resources
            if current_factors.curr_iterations % 3 == 0 {
                current_factors.curr_full_paths_explored = 0;

                for (i, tree) in trees.iter().enumerate() {
                    let tree = tree.borrow();

                    current_factors.curr_full_paths_explored += tree.nodes.iter()
                        .filter(|o| o.info.is_ending)
                        .count();

                    caches[i].remove_excess(&tree, &self.options.next_node_strategy);
                }
            }

            if should_break && requests.is_empty() {
                break;
            }
        }

        // println!("Some={} None={}, len={:?}, count={:?}, ls={:?}, size={:?}", comb.0, comb.1, caches[0].len(), caches[0].count, caches[0].last_searched, caches[0].current_bytes);
        (trees.into_iter().map(|o| o.into_inner()).collect(),
         cursors.into_iter().map(|o| o.into_inner()).collect())
    }

    fn prepare_cache(caches: &mut [Cache], parts: &[&RequestPart], requests: &RequestStorage) -> Option<GenericStorage> {
        let storages: Vec<_> = parts.iter()
            .map(|o| o.owner())
            .map(|o| {
                let o = requests.get(o);
                caches[o.game_index].get(o.node_index)
            })
            .collect();

        // Return None if the cache covers les than 60%
        let found_count = storages.iter().filter(|o| o.is_some()).count();
        if found_count < storages.len() * 60 / 100 {
            return None;
        }

        let replacement = storages.iter()
            .filter_map(|o| o.clone().cloned())
            .map(|mut o| {
                for arr in o.values_mut() {
                    for item in arr {
                        item.mapv_inplace(|_| 0.0);
                    }
                }
                o
            })
            .next()?;

        let replaced: Vec<_> = storages.into_iter()
            .map(|o| o.unwrap_or(&replacement))
            .collect();

        combine_storages(&replaced)
    }

    fn fill_requests(&self, requests: &mut RequestStorage, producer: &mut GamesProducer) -> Result<(), BuildingError> {
        let mut parts_len: usize = requests.iter().map(|o| o.parts.len()).sum();

        while parts_len < self.options.batch_size {
            match producer.next_checked()? {
                Some(new) => {
                    parts_len += new.parts.len();
                    requests.push(new);
                }
                None => break
            }
        }

        Ok(())
    }

    fn take_parts(requests: &RequestStorage, count: usize) -> Vec<&RequestPart> {
        requests.iter()
            .flat_map(|o| &o.parts)
            .filter(|o| matches!(o, RequestPart::Pending { .. }))
            .take(count)
            .collect()
    }

    fn remove_completed(&self, requests: RequestStorage, trees: &[RefCell<DecisionTree>],
                        caches: &mut [Cache]) -> RequestStorage {
        let mut new_requests = RequestStorage::with_capacity(requests.len());

        for req in requests.into_iter() {
            if !req.is_completed() {
                new_requests.push(req);
                continue;  // TODO: break?
            }

            let mut tree = trees[req.game_index].borrow_mut();
            let mut data = Vec::new();

            for part in req.parts {
                if let RequestPart::Completed { m, eval, info, cache, .. } = part {
                    data.push((m, eval, info));
                    // We can just push the cache to the array because the nodes are stored in the same order in the tree
                    caches[req.game_index].insert_next(cache);
                }
            }

            tree.submit_node_children(req.node_index, &data);
        }

        new_requests
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::board_controller::BoardController;
    use crate::chess::decision_tree::building::limiting_factors::LimiterFactors;
    use crate::nn::layers::{dense_layer, sequential_layer};
    use crate::nn::layers::debug_layer::{DebugAction, DebugLayerConfig};
    use crate::nn::layers::filtering::convolution;
    use crate::nn::layers::nn_layers::Layer;
    use crate::nn::loss::loss_func::LossFunc;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::nn::lr_calculators::lr_calculator::LrCalc;
    use super::*;

    #[test]
    fn test_debug() {
        let builder = DecisionTreesBuilder::new(
            vec![DecisionTree::new(true)],
            vec![TreeCursor::new(BoardController::new_start())],
            BuilderOptions {
                next_node_strategy: NextNodeStrategy::Deepest,
                limits: LimiterFactors {
                    max_iterations: Some(5),
                    ..LimiterFactors::default()
                },
                ..BuilderOptions::default()
            },
        );

        let controller = NNController::new(Layer::Sequential(sequential_layer::SequentialConfig {
            layers: vec![
                Layer::Convolution(convolution::ConvolutionConfig {
                    in_channels: 6,
                    stride: 1,
                    kernel_size: 3,
                    init_mode: convolution::ConvolutionInitMode::HeNormal(),
                    out_channels: 2,
                    padding: 0,
                    lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    cache: false,
                }),
                Layer::Debug(DebugLayerConfig {
                    action: DebugAction::PrintShape,
                    tag: "after_conv".to_owned(),
                }),
                Layer::Flatten,
                Layer::Debug(DebugLayerConfig {
                    action: DebugAction::PrintShape,
                    tag: "after_flat".to_owned(),
                }),
                Layer::Dense(dense_layer::DenseConfig {
                    init_mode: dense_layer::DenseLayerInit::Random(),
                    biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
                    out_values: 1,
                    in_values: 6 * 6 * 2,
                }),
                Layer::Debug(DebugLayerConfig {
                    action: DebugAction::PrintShape,
                    tag: "after_dense".to_owned(),
                }),
            ],
        }), LossFunc::Mse).unwrap();

        let (trees, _) = builder.build(&controller);

        println!("len={}", &trees[0].len());
        println!("{:?}", &trees[0]);
        println!("{}", trees[0].to_svg());
    }
}