use crate::ArrayDynF;
use crate::chess::decision_tree::NodeExtraInfo;
use crate::chess::movement::Movement;
use crate::nn::layers::nn_layers::GenericStorage;

#[derive(Debug)]
pub struct Request {
    pub uuid: usize,
    pub game_index: usize,
    pub node_index: usize,
    pub parts: Vec<RequestPart>,
}

#[derive(Debug)]
pub enum RequestPart {
    Completed {
        owner: usize,
        m: Movement,
        eval: f32,
        info: NodeExtraInfo,
        cache: Option<GenericStorage>,
    },
    Pending {
        owner: usize,
        m: Movement,
        array: ArrayDynF,
    }
}

impl RequestPart {
    pub fn movement(&self) -> Movement {
        match self {
            RequestPart::Completed { m, .. } => *m,
            RequestPart::Pending {m,  .. } => *m,
        }
    }

    pub fn owner(&self) -> usize {
        match self {
            RequestPart::Completed {owner,  .. } => *owner,
            RequestPart::Pending { owner, .. } => *owner,
        }
    }
}