use std::collections::HashMap;

pub struct KeyAssigner {
    keys: HashMap<String, u16>,
    reverse: bool
}

impl KeyAssigner {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            reverse: false
        }
    }

    pub fn get_key(&mut self, name: String) -> String {
        let current = self.keys.get(&name).copied().unwrap_or(0);
        if self.reverse {
            let key = format!("{}_{}", name, current - 1);
            self.keys.insert(name, current - 1);
            key
        } else {
            let key = format!("{}_{}", name, current);
            self.keys.insert(name, current + 1);
            key
        }
    }

    pub fn reset_keys(&mut self) {
        for pair in self.keys.iter_mut() {
            *pair.1 = 0;
        }
    }
    
    pub fn revert(&mut self) {
        self.reverse = !self.reverse;
    }
}
