use std::collections::HashMap;

pub struct KeyAssigner {
    keys: HashMap<String, u16>
}

impl KeyAssigner {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new()
        }
    }

    pub fn get_key(&mut self, name: String) -> String {
        let current = self.keys.get(&name).copied().unwrap_or(0);
        let key = format!("{}_{}", name, current);
        self.keys.insert(name, current + 1);
        key
    }

    pub fn reset_keys(&mut self) {
        for pair in self.keys.iter_mut() {
            *pair.1 = 0;
        }
    }
}
