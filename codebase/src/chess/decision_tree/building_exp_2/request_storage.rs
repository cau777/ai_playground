use std::vec;
use crate::chess::decision_tree::building_exp_2::request::Request;

/// A simple wrapper around vec that takes advantage of the fact that
/// the same id will be requested multiple times in a row
pub struct RequestStorage {
    prev_req: usize,
    prev_res: usize,
    items: Vec<Request>,
}

impl RequestStorage {
    pub fn new() -> Self {
        Self {
            prev_res: 0,
            prev_req: 0,
            items: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            prev_res: 0,
            prev_req: 0,
            items: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn push(&mut self, item: Request) {
        self.items.push(item);
    }

    #[inline]
    pub fn get(&mut self, uuid: usize) -> &Request {
        if self.prev_req == uuid {
            &self.items[self.prev_res]
        } else {
            let found = self.items.iter().position(|o| o.uuid == uuid).unwrap();
            self.prev_req = uuid;
            self.prev_res = found;
            &self.items[found]
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=&Request> {
        self.items.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut Request> {
        self.items.iter_mut()
    }

    fn clear_cached(&mut self) {
        self.prev_req = 0;
        self.prev_res = 0;
    }
}

impl IntoIterator for RequestStorage {
    type Item = Request;
    type IntoIter = vec::IntoIter<Request>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
