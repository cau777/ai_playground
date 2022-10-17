use std::collections::HashMap;
use std::time::Instant;

use super::model_metadata::ModelMetadata;
use super::{model_source::ModelSource, path_utils};
use rand::rngs::StdRng;
use rand::{self, Rng, SeedableRng};
use rocket::serde::{Deserialize, Serialize};

pub const TRAIN_TASK: &'static str = "train";
pub const TEST_TASK: &'static str = "test";

#[derive(Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
pub struct Task {
    task_name: &'static str,
    model_config: String,
    model_data: String,
    data_path: String,
    version: Option<u32>,
    batch: Option<u32>,
}

struct VersionToTest {
    awaiting: Vec<u32>,
    assigned: Vec<(u32, Instant)>,
    accumulated: f64,
    count: u32,
}

impl VersionToTest {
    fn new(count: u32) -> Self {
        Self {
            awaiting: (0..count).collect(),
            assigned: Vec::new(),
            accumulated: 0.0,
            count,
        }
    }

    fn get(&mut self) -> Option<u32> {
        let popped = self.awaiting.pop();
        if popped.is_some() {
            let popped = popped.unwrap();
            self.assigned.push((popped, Instant::now()));
            return Some(popped);
        }
        let expired = self
            .assigned
            .iter()
            .position(|(_, instant)| instant.elapsed().as_secs() > 60 * 3)?;
        Some(self.assigned.swap_remove(expired).0)
    }

    fn complete_batch(&mut self, batch: u32, result: f64) -> Result<(), ()> {
        let completed = self
            .assigned
            .iter()
            .position(|(b, _)| *b == batch)
            .ok_or(())?;
        self.assigned.swap_remove(completed);
        self.accumulated += result / self.count as f64;
        Ok(())
    }

    fn get_result(&self) -> Option<f64> {
        if self.awaiting.len() == 0 && self.assigned.len() == 0 {
            Some(self.accumulated)
        } else {
            None
        }
    }
}

pub struct TaskManager {
    testing: HashMap<u32, VersionToTest>,
    rng: StdRng,
    initialized: bool
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            testing: HashMap::new(),
            rng: StdRng::seed_from_u64(777),
            initialized: false
        }
    }

    pub fn get_task(&mut self, source: &mut ModelSource) -> Task {
        self.init(source);
        let name = source.name();

        for (version, batches) in self.testing.iter_mut() {
            let task = batches.get();
            if task.is_some() {
                let batch = task.unwrap();
                return Task {
                    task_name: TEST_TASK,
                    model_config: path_utils::get_model_config_url(name),
                    model_data: path_utils::get_model_url(name, *version),
                    data_path: path_utils::get_test_batch_url(name, batch),
                    version: Some(*version),
                    batch: Some(batch)
                };
            }
        }

        Task {
            task_name: TRAIN_TASK,
            model_config: path_utils::get_model_config_url(name),
            model_data: path_utils::get_model_url(name, source.latest()),
            data_path: path_utils::get_train_batch_url(
                name,
                self.rng.gen_range(0..source.train_count()),
            ),
            version: None,
            batch: None
        }
    }
    
    pub fn add_to_test(&mut self, version: u32, count: u32) {
        if !self.testing.contains_key(&version) {
            self.testing.insert(version, VersionToTest::new(count));
        }
    }

    pub fn complete_test_task(&mut self, source: &mut ModelSource, version: u32,
                              batch: u32, accuracy: f64) {
        self.init(source);
        let testing = self.testing.get_mut(&version).unwrap();
        testing.complete_batch(batch, accuracy).unwrap();
        let result = testing.get_result();
        match result {
            Some(accuracy) => {
                println!("{}", accuracy);
                source.save_model_meta(version, &ModelMetadata { accuracy }).unwrap();
            },
            None => {}
        }
    }
    
    fn init(&mut self, source: &mut ModelSource) {
        if !self.initialized {
            for to_test in source.versions_to_test() {
                self.add_to_test(to_test, source.test_count());
            }
        }
        self.initialized = true;
    }
}
