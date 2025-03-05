use bebe_ai::document::{self, DocumentFetcher};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let fetcher = document::mv::MieuxVivreFetcher::new();
    let chunks = fetcher.fetch().await.unwrap();
    tracing::info!("Fetched {} chunks", chunks.len());
    // write chunks to JSON disk
    let json = serde_json::to_string_pretty(&chunks).unwrap();
    std::fs::write("chunks.json", json).unwrap();
}