use hercules_agent::manager::ModelManager;
use std::sync::{Arc, Mutex};
#[tokio::main]
async fn main() {
    let manager = ModelManager::new();
    let progress = Arc::new(Mutex::new(None));
    let logs = Arc::new(Mutex::new(Vec::new()));
    match manager.download_ollama_model("llama3.2:1b", progress.clone(), logs.clone()).await {
        Ok(_) => println!("Success!"),
        Err(e) => println!("Error: {}", e),
    }
}
