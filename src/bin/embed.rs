use bebe_ai::document::{mv::MieuxVivreMetadata, Chunk};

#[tokio::main]
async fn main() {
    // load chunks
    tracing_subscriber::fmt::init();
    let chunks_json = std::fs::read("chunks.json").unwrap();
    let chunks: Vec<Chunk<MieuxVivreMetadata>> = serde_json::from_slice(&chunks_json).unwrap();
    tracing::info!("Loaded {} chunks", chunks.len());
    let embedded = bebe_ai::embedding::generate_embeddings(chunks).await;
    // write embeddings to JSON
    let json = serde_json::to_string_pretty(&embedded).unwrap();
    std::fs::write("embedded.json", json).unwrap();
}