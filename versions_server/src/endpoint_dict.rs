/// Utility struct that stores a value for each model name
#[derive(Clone)]
pub struct EndpointDict<T: Clone> {
    pub digits: T,
    pub chess: T,
}

impl<T: Clone> EndpointDict<T> {
    pub fn new(digits: T, chess: T) -> Self {
        Self {digits, chess}
    }

    pub fn get_from_name(&self, name: &str) -> Option<&T> {
        match name.to_ascii_lowercase().as_str() {
            "digits" => Some(&self.digits),
            "chess" => Some(&self.chess),
            _ => None,
        }
    }
}