use std::collections::{HashMap, HashSet};
use bloomfilter::Bloom;
use codebase::chess::board::Board;
use codebase::chess::board_controller::board_hashable::BoardHashable;
use codebase::chess::decision_tree::DecisionTree;

pub fn calc_nodes_confidence_1(tree: &DecisionTree) -> Vec<Option<f32>> {
    let mut values: Vec<_> = tree.nodes.iter().map(|o| {
        let side = o.get_current_side(tree.start_side);

        o.children.as_ref()
            .and_then(|o| {
                // Get the instant evaluation of the best child, regardless of deeper nodes
                let mut evals: Vec<_> = o.iter()
                    .map(|&o| tree.nodes[o].pre_eval)
                    .collect();

                evals.sort_unstable_by(f32::total_cmp);
                if side {
                    evals.last().copied()
                } else {
                    evals.first().copied()
                }
            })
            .map(|eval| (o.pre_eval - eval).abs())
            // Apply function to smooth results
            .map(|o| f32::exp(o * -0.2))
    }).collect();

    for (i, node) in tree.nodes.iter().enumerate().skip(1) {
        if let Some(val) = values[i] {
            let new = val * values[node.parent].unwrap();
            values[i] = Some(new.max(0.1))
        }
    }

    values
}

pub fn calc_nodes_confidence_2(tree: &DecisionTree) -> Vec<Option<f32>> {
    tree.nodes.iter().map(|o| {
        o.children_eval
            .map(|eval| (o.pre_eval - eval).abs())
            // Apply function to smooth results
            .map(|o| f32::exp(o * -1.0))
            .map(|o| f32::max(o, 0.1))
    }).collect()
}

pub fn dedupe(mut items: Vec<(Board, f32)>) -> Vec<(Board, f32)> {
    let mut result = Vec::new();

    dedupe_bloom(&mut items, &mut result);
    dedupe_hashmap(&mut items, &mut result);

    result
}

fn dedupe_bloom(from: &mut Vec<(Board, f32)>, result: &mut Vec<(Board, f32)>) {
    let mut bloom = Bloom::new(2048, from.len());
    from.retain(|item| {
        let hashable = BoardHashable::new(item.0.pieces);
        let might_be_dup = bloom.check_and_set(&hashable);
        if !might_be_dup {
            result.push(item.clone());
        }
        might_be_dup
    })
}

fn dedupe_hashmap(from: &mut Vec<(Board, f32)>, result: &mut Vec<(Board, f32)>)
{
    let mut map = HashMap::new();
    for item in from.drain(..) {
        map.insert(BoardHashable::new(item.0.pieces), item);
    }

    let mut definitely_dupes = HashSet::new();
    for item in result.iter() {
        let hashable = BoardHashable::new(item.0.pieces);
        if map.contains_key(&hashable) {
            definitely_dupes.insert(hashable);
        }
    }

    for (key, value) in map {
        if !definitely_dupes.contains(&key) {
            result.push(value)
        }
    }
}

pub fn calc_mean_and_std_dev(tree: &DecisionTree, node: usize) -> (f64, f64){
    let children = &tree.nodes[node].children.as_ref().unwrap();
    let children_len = children.len() as f64;

    let evals: Vec<_> = children.iter()
        .map(|&o| tree.nodes[o].pre_eval as f64)
        .collect();

    let mean = evals.iter().sum::<f64>() / children_len;

    let std_dev = f64::sqrt(
        evals.iter().map(|&o| f64::powi(o - mean, 2)).sum::<f64>()
            /
            children_len
    );
    (mean, std_dev)
}