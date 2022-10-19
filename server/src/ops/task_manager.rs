use std::collections::HashMap;
use std::time::Instant;

use super::model_metadata::ModelMetadata;
use super::url_creator::UrlCreator;
use super::model_source::ModelSource;
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
        // If there's no batch in queue, search for expired tasks
        let expired = self
            .assigned
            .iter_mut()
            .find(|(_, instant)| instant.elapsed().as_secs() > 60 * 3)?;
        expired.1 = Instant::now();
        Some(expired.0)
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
    initialized: bool,
    url_creator: Option<UrlCreator>
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            testing: HashMap::new(),
            rng: StdRng::seed_from_u64(777),
            initialized: false,
            url_creator: None
        }
    }

    pub fn get_task(&mut self, source: &mut ModelSource) -> Task {
        self.init(source);
        let name = source.name();
        let url_creator = self.url_creator.as_ref().unwrap();

        for (version, batches) in self.testing.iter_mut() {
            let task = batches.get();
            if task.is_some() {
                let batch = task.unwrap();
                return Task {
                    task_name: TEST_TASK,
                    model_config: url_creator.gen_model_config(),
                    model_data: url_creator.gen_model_data(*version),
                    data_path: url_creator.get_test_batch(batch),
                    version: Some(*version),
                    batch: Some(batch)
                };
            }
        }

        Task {
            task_name: TRAIN_TASK,
            model_config: url_creator.gen_model_config(),
            model_data: url_creator.gen_model_data(source.latest()),
            data_path: url_creator.get_train_batch(self.rng.gen_range(0..source.train_count())),
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
        let url_creator = self.url_creator.as_ref().unwrap();
        
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
            self.url_creator = Some(UrlCreator::new(source.name()))
        }
        self.initialized = true;
    }
}
