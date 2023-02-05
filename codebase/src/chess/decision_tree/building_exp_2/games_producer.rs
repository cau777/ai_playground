use std::cell::RefCell;
use std::ops::RangeFrom;
use std::sync::{Arc, RwLock};
use ndarray_rand::rand::{Rng, thread_rng};
use ndarray_rand::rand::rngs::ThreadRng;
use crate::chess::board_controller::BoardController;
use crate::chess::decision_tree::building_exp_2::{BuilderOptions, NextNodeStrategy};
use crate::chess::decision_tree::building_exp_2::request::{Request, RequestPart};
use crate::chess::decision_tree::cursor::TreeCursor;
use crate::chess::decision_tree::{DecisionTree, NodeExtraInfo};
use crate::chess::decision_tree::building_exp_2::nodes_in_progress_set::NodesInProgressSet;
use crate::chess::game_result::GameResult;
use crate::chess::movement::Movement;

pub struct GamesProducer<'a> {
    current_game: usize,
    workers: Vec<GamesProducerWorker<'a>>,
}

impl<'a> GamesProducer<'a> {
    pub fn new(initial_trees: &'a [DecisionTree], initial_cursors: &'a [TreeCursor],
               trees: &'a [RefCell<DecisionTree>], cursors: &'a [RefCell<TreeCursor>],
               options: &'a BuilderOptions) -> Self {
        let mut workers = Vec::new();
        if initial_trees.len() != initial_cursors.len() ||
            initial_cursors.len() != trees.len() ||
            trees.len() != cursors.len() {
            panic!("Slice sizes don't match");
        }

        let producer = Arc::new(RwLock::new(0..));
        for i in 0..trees.len() {
            workers.push(GamesProducerWorker {
                game_index: i,
                ids_producer: producer.clone(),
                initial_tree: &initial_trees[i],
                initial_cursor: &initial_cursors[i],
                tree: &trees[i],
                cursor: &cursors[i],
                options,
                in_progress: NodesInProgressSet::new(),
            });
        }

        Self {
            current_game: 0,
            workers,
        }
    }

    fn cycle_game(&mut self) {
        self.current_game = (self.current_game + 1) % self.workers.len();
    }
}

impl<'a> Iterator for GamesProducer<'a> {
    type Item = Request;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.workers[self.current_game].work();
        self.cycle_game();
        result
    }
}

type IdsProducer = RangeFrom<usize>;

struct GamesProducerWorker<'a> {
    game_index: usize,
    tree: &'a RefCell<DecisionTree>,
    cursor: &'a RefCell<TreeCursor>,
    initial_tree: &'a DecisionTree,
    initial_cursor: &'a TreeCursor,
    in_progress: NodesInProgressSet,
    ids_producer: Arc<RwLock<IdsProducer>>,
    options: &'a BuilderOptions,
}

impl<'a> GamesProducerWorker<'a> {
    pub fn work(&mut self) -> Option<Request> {
        let tree = self.tree.borrow();
        let mut cursor = self.cursor.borrow_mut();

        if tree.len() == 1 {
            return Some(self.generate_request(&tree, &mut cursor, 0));
        }

        self.in_progress.validate_all(&tree.nodes);

        let next_node = self.choose_next_node(&tree);
        match next_node {
            Some(next) => {
                let result = Some(self.generate_request(&tree, &mut cursor, next));
                self.in_progress.insert(next);
                result
            }
            None => {
                self.tree.replace(self.initial_tree.clone());
                *cursor = self.initial_cursor.clone();

                Some(self.generate_request(&self.tree.borrow(), &mut cursor, 0))
            }
        }
    }

    fn generate_request(&mut self, tree: &DecisionTree, cursor: &mut TreeCursor, node: usize) -> Request {
        let mut result = Request {
            uuid: self.ids_producer.write().unwrap().next().unwrap(),
            game_index: self.game_index,
            node_index: node,
            parts: vec![],
        };
        let mut rng = thread_rng();

        cursor.go_to(node, &tree.nodes);
        let controller = cursor.get_controller();
        let continuations = controller.get_opening_continuations();
        if continuations.is_empty() {
            let side = tree.get_side_at(node);
            let mut controller = controller.clone();
            let moves = controller.get_possible_moves(side);
            for m in moves {
                controller.apply_move(m);

                let moves = controller.get_possible_moves(!side);
                let game_result = controller.get_game_result(&moves);
                result.parts.push(self.handle_game_result(&controller, m, game_result, result.uuid));

                controller.revert();
            };
        } else {
            const SHIFT: f32 = 0.0001;
            for m in continuations {
                result.parts.push(RequestPart::Completed {
                    owner: result.uuid,
                    m,
                    info: NodeExtraInfo { is_ending: false, is_opening: true },
                    cache: None,
                    // Openings are usually good for both sides
                    // Add a small random value so the AI can choose from different openings
                    eval: rng.gen_range((-SHIFT)..SHIFT),
                })
            }
        }

        result
    }

    fn handle_game_result(&self, controller: &BoardController, m: Movement, game_result: GameResult, uuid: usize) -> RequestPart {
        match game_result {
            GameResult::Undefined => {
                RequestPart::Pending {
                    owner: uuid,
                    m,
                    array: controller.current().to_array().into_dyn(),
                }
            }
            GameResult::Win(side, _) => {
                RequestPart::Completed {
                    owner: uuid,
                    m,
                    info: NodeExtraInfo { is_ending: true, is_opening: false },
                    cache: None,
                    eval: if side { 1.0 } else { -1.0 },
                }
            }
            GameResult::Draw(_) => {
                RequestPart::Completed {
                    owner: uuid,
                    m,
                    info: NodeExtraInfo { is_ending: true, is_opening: false },
                    cache: None,
                    eval: 0.0,
                }
            }
        }
    }

    fn choose_next_node(&self, tree: &DecisionTree) -> Option<usize> {
        let mut rng = thread_rng();

        match self.options.next_node_strategy {
            NextNodeStrategy::BestNode => {
                if rng.gen_range(0.0..1.0) < self.options.random_node_chance {
                    self.choose_random_node(tree, &mut rng)
                } else {
                    self.choose_best_unexplored_recursive(tree, 0)
                        .map(|(i, _)| i)
                }
            }
            NextNodeStrategy::Deepest => {
                if rng.gen_range(0.0..1.0) < self.options.random_node_chance {
                    self.choose_random_node(tree, &mut rng)
                } else {
                    self.choose_deepest_node(tree)
                }
            }
        }
    }

    fn choose_random_node(&self, tree: &DecisionTree, rng: &mut ThreadRng) -> Option<usize> {
        tree.nodes.iter()
            .enumerate()
            .filter(|(_, o)| o.children.is_none() && !o.info.is_ending)
            .map(|(i, _)| i)
            .max_by_key(|_| rng.gen_range(0..10_000_000))
    }

    fn choose_deepest_node(&self, tree: &DecisionTree) -> Option<usize> {
        tree.nodes.iter()
            // Instead of accessing all the nodes directly, we call get_ordered_children on all parents
            .filter_map(|o| o.get_ordered_children(tree.start_side))
            .flatten()
            .map(|&i| (i, &tree.nodes[i]))

            .filter(|(_, o)| !o.info.is_ending)
            .filter(|(_, o)| o.children.is_none())
            .filter(|(i, _)| !self.in_progress.contains(*i))

            // max_by_key() can't be used because it returns the last element if several elements are equally maximum
            .reduce(|acc, item| if acc.1.depth >= item.1.depth { acc } else { item })
            .map(|(i, _)| i)

        // tree.nodes.iter()
        //     .enumerate()
        //     .filter(|(_, o)| !o.info.is_ending)
        //     .filter(|(_, o)| o.children.is_none())
        //     .filter(|(i, _)| !self.in_progress.contains(*i))
        //     .max_by_key(|(_, o)| o.depth)
        //     .map(|(i, _)| i)
    }

    // TODO: iterative version
    fn choose_best_unexplored_recursive(&self, tree: &DecisionTree, node: usize) -> Option<(usize, f32)> {
        if tree.nodes[node].info.is_ending {
            return None;
        }

        if self.in_progress.contains(node) {
            return None;
        }

        let children = &tree.nodes[node].children;
        if let Some(children) = children {
            let mut result: Option<(usize, f32)> = None;

            for c in children {
                let option = self.choose_best_unexplored_recursive(tree, *c);

                if let Some(current) = result {
                    if let Some(new) = option {
                        // if side && new.1 > current.1 {
                        //     result = Some(new);
                        // } else if !side && new.1 < current.1 {
                        //     result = Some(new)
                        // }
                        if new.1 > current.1 {
                            result = Some(new);
                        }
                    }
                } else {
                    result = option;
                }
            }

            result.map(|(i, v)| (i, -v))
        } else {
            let side = tree.get_side_at(node);
            Some((node, tree.nodes[node].pre_eval * if side { -1.0 } else { 1.0 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::board_controller::BoardController;
    use crate::chess::movement::Movement;
    use crate::chess::openings::openings_tree::OpeningsTree;
    use super::*;

    #[test]
    fn test_debug() {
        // let (it, t, ic, c) = build(1, "");
        // let options = BuilderOptions::default();
        //
        // let mut producer = GamesProducer::new(&it, &ic, &t, &c, &options);
        //
        // for i in 0..10 {
        //     let request = producer.next().unwrap();
        //     for part in &request.parts {
        //         match part {
        //             RequestPart::Completed { m, .. } => {
        //                 println!("Completed {:?}", m);
        //             }
        //             RequestPart::Pending { m, .. } => {
        //                 println!("Pending {:?}", m);
        //             }
        //         }
        //     }
        //     println!("--------");
        // }
    }

    #[test]
    fn test_should_select_best() {
        let (it, t, ic, c) = build(1, "");
        let options = BuilderOptions::default();

        let mut producer = GamesProducer::new(&it, &ic, &t, &c, &options);

        let r = producer.next().unwrap();

        // White's turn
        let mut evals = prepare_to_submit(&r, 0.0);
        change_movement(&mut evals, "B2", "B3", 1.0);
        t[0].borrow_mut().submit_node_children(0, &evals);
        let r = producer.next().unwrap();
        let branch1 = find_in_parent(&t[0].borrow(), 0, "B2", "B3");
        assert_eq!(r.node_index, branch1);

        // Black's turn
        let mut evals = prepare_to_submit(&r, 0.5);
        change_movement(&mut evals, "A7", "A6", 0.4);
        t[0].borrow_mut().submit_node_children(branch1, &evals);
        let r = producer.next().unwrap();
        let branch2 = find_in_parent(&t[0].borrow(), branch1, "A7", "A6");
        assert_eq!(r.node_index, branch2);

        // White's turn
        let mut evals = prepare_to_submit(&r, 0.2);
        change_movement(&mut evals, "A2", "A3", 0.3);
        change_movement(&mut evals, "A2", "A4", 0.25);
        t[0].borrow_mut().submit_node_children(branch2, &evals);
        let r = producer.next().unwrap();
        let branch3 = find_in_parent(&t[0].borrow(), branch2, "A2", "A3");
        assert_eq!(r.node_index, branch3);

        // Black's turn
        let evals = prepare_to_submit(&r, -1.0);
        t[0].borrow_mut().submit_node_children(branch3, &evals);
        let r = producer.next().unwrap();
        let branch4 = find_in_parent(&t[0].borrow(), branch2, "A2", "A4");
        assert_eq!(r.node_index, branch4);
    }

    #[test]
    fn test_select_deepest() {
        let (it, t, ic, c) = build(1, "");
        let options = BuilderOptions {
            next_node_strategy: NextNodeStrategy::Deepest,
            ..BuilderOptions::default()
        };

        let mut producer = GamesProducer::new(&it, &ic, &t, &c, &options);

        let r = producer.next().unwrap();

        // White's turn
        let mut evals = prepare_to_submit(&r, 0.0);
        change_movement(&mut evals, "B2", "B3", 1.0);
        t[0].borrow_mut().submit_node_children(0, &evals);
        let r = producer.next().unwrap();
        let branch1 = find_in_parent(&t[0].borrow(), 0, "B2", "B3");
        assert_eq!(r.node_index, branch1);

        // Black's turn
        let mut evals = prepare_to_submit(&r, 0.5);
        change_movement(&mut evals, "A7", "A6", 0.4);
        t[0].borrow_mut().submit_node_children(branch1, &evals);
        let r = producer.next().unwrap();
        let branch2 = find_in_parent(&t[0].borrow(), branch1, "A7", "A6");
        assert_eq!(r.node_index, branch2);

        // White's turn
        let mut evals = prepare_to_submit(&r, 0.2);
        change_movement(&mut evals, "A2", "A3", 0.3);
        change_movement(&mut evals, "A2", "A4", 0.25);
        t[0].borrow_mut().submit_node_children(branch2, &evals);
        let r = producer.next().unwrap();
        let branch3 = find_in_parent(&t[0].borrow(), branch2, "A2", "A3");
        assert_eq!(r.node_index, branch3);

        // Black's turn
        let mut evals = prepare_to_submit(&r, -1.0);
        change_movement(&mut evals, "A6", "A5", -1.1);
        t[0].borrow_mut().submit_node_children(branch3, &evals);
        let r = producer.next().unwrap();
        let branch4 = find_in_parent(&t[0].borrow(), branch3, "A6", "A5");
        assert_eq!(r.node_index, branch4);
    }

    #[test]
    fn test_openings() {
        let (it, t, ic, c) = build(1, "||1,2,3\n|a2a3|\n|b2b3|\n|c2c3|");
        let options = BuilderOptions::default();

        let mut producer = GamesProducer::new(&it, &ic, &t, &c, &options);
        let expected = vec![Movement::from_notations("A2", "A3"),
                            Movement::from_notations("B2", "B3"),
                            Movement::from_notations("C2", "C3")];
        let actual = producer.next().unwrap();

        assert_eq!(actual.parts.len(), 3);
        for x in &actual.parts {
            assert!(expected.contains(&x.movement()));
        }
    }

    #[test]
    fn test_return_different_nodes() {
        let (it, t, ic, c) = build(1, "");
        let options = BuilderOptions::default();

        let mut producer = GamesProducer::new(&it, &ic, &t, &c, &options);
        let r0 = producer.next().unwrap();
        let evals = prepare_to_submit(&r0, 0.5);
        t[0].borrow_mut().submit_node_children(0, &evals);

        let r1 = producer.next().unwrap();
        let r2 = producer.next().unwrap();

        assert_ne!(r1.node_index, r2.node_index);
    }

    #[allow(clippy::type_complexity)]
    fn build(count: usize, openings: &str) -> (Vec<DecisionTree>, Vec<RefCell<DecisionTree>>, Vec<TreeCursor>, Vec<RefCell<TreeCursor>>) {
        let mut initial_trees = Vec::with_capacity(count);
        let mut initial_cursors = Vec::with_capacity(count);

        for _ in 0..count {
            initial_trees.push(DecisionTree::new(true));
            let mut controller = BoardController::new_start();
            if !openings.is_empty() {
                controller.add_openings_tree(Arc::new(OpeningsTree::load_from_string(openings).unwrap()));
            }
            initial_cursors.push(TreeCursor::new(controller));
        }

        (
            initial_trees.clone(),
            initial_trees.iter().cloned().map(RefCell::new).collect(),
            initial_cursors.clone(),
            initial_cursors.iter().cloned().map(RefCell::new).collect()
        )
    }

    fn prepare_to_submit(r: &Request, val: f32) -> Vec<(Movement, f32, NodeExtraInfo)> {
        r.parts.iter().map(|o| {
            match o {
                RequestPart::Completed { .. } => panic!(),
                RequestPart::Pending { m, .. } => (*m, val, NodeExtraInfo::default())
            }
        }).collect()
    }

    fn change_movement(arr: &mut [(Movement, f32, NodeExtraInfo)], from: &str, to: &str, val: f32) -> usize {
        let position = arr.iter().position(|o| o.0 == Movement::from_notations(from, to)).unwrap();
        arr[position].1 = val;
        position
    }

    fn find_in_parent(tree: &DecisionTree, parent: usize, from: &str, to: &str) -> usize {
        *tree.nodes[parent].children
            .as_ref().unwrap().iter()
            .find(|&&o| tree.nodes[o].movement == Movement::from_notations(from, to))
            .unwrap()
    }
}

/*
fn get_best_path_variant(&self, tree: &DecisionTree) -> Option<usize> {
        tree.best_path_iter(true)
            .filter_map(|o| o.get_ordered_children(tree.start_side))
            // Get the best and second best unexplored options for a node
            .filter_map(|mut o| {
                let first = o.next();
                let second = o.find(|o| {
                    let node = &tree.nodes[**o];
                    node.children.is_none() && !node.info.is_ending
                });
                match (first, second) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None
                }
            })
            // Compute the difference between the eval of the first continuation option and the second best
            // The logic is that the AI may misevaluate a position by a little and get a wrong result
            .map(|(a, b)| {
                let node_a = &tree.nodes[*a];
                let node_b = &tree.nodes[*b];
                ((node_a.eval() - node_b.eval()).abs(), b)
            })
            .min_by(|(v1, _), (v2, _)| v1.total_cmp(v2))
            .map(|(_, n)| *n)
    }
 */