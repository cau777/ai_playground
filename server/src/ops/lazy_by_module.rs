use super::{lazy::LazyWithoutArgs, model_source::ModelSource, task_manager::TaskManager};
use rocket::State;
use std::sync::RwLock;

pub type ModelSourcesState = State<RwLock<ByModule<LazyWithoutArgs<ModelSource>>>>;
pub type TaskManagerState = State<RwLock<ByModule<TaskManager>>>;

pub struct ByModule<T> {
    pub digits: T,
}

impl<T> ByModule<T> {
    pub fn new(digits: T) -> Self {
        Self {
            digits,
        }
    }

    pub fn fields_mut<'a>(&'a mut self) -> Vec<&'a mut T> {
        vec![&mut self.digits]
    }
}
