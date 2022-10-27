use std::collections::HashMap;
use std::time::Instant;
use super::model_metadata::ModelMetadata;
use super::url_creator::UrlCreator;
use super::model_source::ModelSource;
use rand::rngs::StdRng;
use rand::{self, Rng, SeedableRng};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Task {
    Train {url: String},
    Test {version: u32, batch: u32, url: String, model_url: String}
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

    fn complete_batch(&mut self, batch: u32, result: f64) -> Option<()> {
        let completed = self
            .assigned
            .iter()
            .position(|(b, _)| *b == batch)?;
        self.assigned.swap_remove(completed);
        self.accumulated += result / self.count as f64;
        Some(())
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
    url_creator: UrlCreator
}

impl TaskManager {
    pub fn new(source: &ModelSource) -> Self {
        let mut result = Self {
            testing: HashMap::new(),
            rng: StdRng::seed_from_u64(777),
            url_creator: UrlCreator::new(source.name()),
        };
        
        // Load versions that are not tested yet
        for to_test in source.versions_to_test() {
            result.add_to_test(to_test, source.test_count());
        }
        
        result
    }

    pub fn get_task(&mut self, source: &ModelSource) -> Task {
        let url_creator = &self.url_creator;

        for (version, batches) in self.testing.iter_mut() {
            let task = batches.get();
            if task.is_some() {
                let batch = task.unwrap();
                return Task::Test {
                    version: *version,
                    batch,
                    url: url_creator.get_test_batch(batch),
                    model_url: url_creator.get_model_data(*version),
                };
            }
        }

        Task::Train {
            url: url_creator.get_train_batch(self.rng.gen_range(0..source.train_count())),
        }
    }
    
    pub fn add_to_test(&mut self, version: u32, count: u32) {
        if !self.testing.contains_key(&version) {
            self.testing.insert(version, VersionToTest::new(count));
        }
    }

    pub fn complete_test_task(&mut self, source: &mut ModelSource, version: u32,
                              batch: u32, accuracy: f64) -> Option<()> {
        let testing = self.testing.get_mut(&version)?;
        testing.complete_batch(batch, accuracy)?;
        let result = testing.get_result();
        match result {
            Some(accuracy) => {
                println!("{}", accuracy);
                source.save_model_meta(version, &ModelMetadata { accuracy }).unwrap();
            },
            None => {}
        }
        Some(())
    }
}
