use crate::ArrayDynF;
use crate::chess::decision_tree::NodeExtraInfo;
use crate::chess::movement::Movement;
use crate::nn::layers::nn_layers::GenericStorage;

#[derive(Debug, Clone)]
pub struct Request {
    pub uuid: usize,
    pub game_index: usize,
    pub node_index: usize,
    pub parts: Vec<RequestPart>,
}

impl Request {
    pub fn is_completed(&self) -> bool {
        self.parts.iter().all(|o| matches!(o, RequestPart::Completed { .. }))
    }

    pub fn count_pending(&self) -> usize {
        self.parts.iter().filter(|o| matches!(o, RequestPart::Pending {..})).count()
    }
}

// impl PartialEq for Request {
//     fn eq(&self, other: &Self) -> bool {
//         self.uuid == other.uuid
//     }
// }
//
// impl Hash for Request {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         state.write_isize(self.uuid as isize)
//     }
// }

#[derive(Debug, Clone)]
pub enum RequestPart {
    Completed {
        owner: usize,
        m: Movement,
        index_in_owner: usize,
        eval: f32,
        info: NodeExtraInfo,
        cache: Option<GenericStorage>,
    },
    Pending {
        owner: usize,
        m: Movement,
        index_in_owner: usize,
        array: ArrayDynF,
    },
}

impl RequestPart {
    #[inline]
    pub fn movement(&self) -> Movement {
        match self {
            RequestPart::Completed { m, .. } => *m,
            RequestPart::Pending { m, .. } => *m,
        }
    }

    #[inline]
    pub fn owner(&self) -> usize {
        match self {
            RequestPart::Completed { owner, .. } => *owner,
            RequestPart::Pending { owner, .. } => *owner,
        }
    }

    #[inline]
    pub fn index_in_owner(&self) -> usize {
        match self {
            RequestPart::Completed { index_in_owner, .. } => *index_in_owner,
            RequestPart::Pending { index_in_owner, .. } => *index_in_owner,
        }
    }
}