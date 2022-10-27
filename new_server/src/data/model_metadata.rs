use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub accuracy: f64
}