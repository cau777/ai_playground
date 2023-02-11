use std::cell::Cell;
use std::vec;
use crate::chess::decision_tree::building_exp_2::request::Request;

/// A simple wrapper around vec that takes advantage of the fact that
/// the same id will be requested multiple times in a row
pub struct RequestStorage {
    prev_req: Cell<(usize, usize)>,
    items: Vec<Request>,
}

impl RequestStorage {
    pub fn new() -> Self {
        Self {
            prev_req: Cell::new((0, 0)),
            items: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            prev_req: Cell::new((0, 0)),
            items: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn push(&mut self, item: Request) {
        self.items.push(item);
    }

    #[inline]
    pub fn get(&self, uuid: usize) -> &Request {
        let (prev_req, prev_res) = self.prev_req.get();
        if prev_req == uuid {
            &self.items[prev_res]
        } else {
            let found = self.items.iter().position(|o| o.uuid == uuid).unwrap();
            self.prev_req.replace((uuid, found));
            &self.items[found]
        }
    }

    #[inline]
    pub fn get_mut(&mut self, uuid: usize) -> &mut Request {
        let (prev_req, prev_res) = self.prev_req.get();
        if prev_req == uuid {
            &mut self.items[prev_res]
        } else {
            let found = self.items.iter().position(|o| o.uuid == uuid).unwrap();
            self.prev_req.replace((uuid, found));
            &mut self.items[found]
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item=&Request> {
        self.items.iter()
    }

    fn clear_cached(&mut self) {
        self.prev_req = Cell::new((0, 0));
    }
}

impl IntoIterator for RequestStorage {
    type Item = Request;
    type IntoIter = vec::IntoIter<Request>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
