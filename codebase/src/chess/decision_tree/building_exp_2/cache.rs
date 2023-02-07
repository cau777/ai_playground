use crate::nn::layers::nn_layers::GenericStorage;

#[derive(Clone)]
pub struct Cache {
    count: usize,
    buffer: Vec<Option<GenericStorage>>,
    max: usize,
    last_searched: usize,
}

impl Cache {
    pub fn new(max: usize) -> Self {
        Self {
            count: 0,
            buffer: vec![None],
            max,
            last_searched: 0,
        }
    }

    pub fn get(&self, index: usize) -> &Option<GenericStorage> {
        &self.buffer[index]
    }

    pub fn push(&mut self, value: Option<GenericStorage>) {
        if value.is_some() {
            self.count += 1;
            self.remove_excess();
        }
        self.buffer.push(value);
    }

    pub fn remove(&mut self, index: usize) {
        if self.buffer[index].is_some() {
            self.count -= 1;
        }
        self.buffer[index] = None;
    }

    fn remove_excess(&mut self) {
        while self.count >= self.max && self.last_searched < self.buffer.len() {
            self.remove(self.last_searched);
            self.last_searched += 1;
        }
    }
}