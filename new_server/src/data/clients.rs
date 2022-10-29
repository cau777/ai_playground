use std::{collections::HashMap, error::Error};

use futures::{channel::mpsc::{UnboundedSender}, SinkExt};
use warp::ws::Message;

type Sender = UnboundedSender<Result<Message, warp::Error>>;

pub struct Clients {
    map: HashMap<String, Sender>
}

impl Clients {
    pub fn new() -> Self {
        Self {
            map: HashMap::new()
        }
    }
    
    pub fn register(&mut self, sender: Sender) -> String {
        let key = uuid::Uuid::new_v4().simple().to_string();
        self.map.insert(key.clone(), sender);
        key
    }
    
    pub async fn send_to(&mut self, id: &str, message: Message) -> Result<(), Box<dyn Error>> {
        let sender = self.map.get_mut(id).ok_or(format!("Id {} not registered", id))?;
        sender.send(Ok(message)).await?;
        Ok(())
    }
}