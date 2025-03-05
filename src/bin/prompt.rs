use bebe_ai::{document::{mv::MieuxVivreMetadata, Chunk}, embedding::{self, EmbeddedChunk}, llm};
use itertools::Itertools;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    tracing::info!("Loading embeddings from disk");
    // fetch embeddings
    let embeddings_json = std::fs::read("embedded.json").unwrap();
    let embeddings: Vec<bebe_ai::embedding::EmbeddedChunk<bebe_ai::document::mv::MieuxVivreMetadata>> = serde_json::from_slice(&embeddings_json).unwrap();

    tracing::info!("Loaded {} embeddings", embeddings.len());

    // prompt user for query using stdin
    let mut query = String::new();
    println!("Demandez une question:");
    print!(">");
    std::io::stdin().read_line(&mut query).unwrap();

    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap();

    tracing::info!("Generating search query from user query");

    // convert user query to search query
    let query = query.trim().to_string();
    let query = llm::chat(&gemini_key, &format!(
        "Convert the following user query to a search query: {}. Only respond with the search query, nothing else.", query
    )).await.unwrap();

    tracing::info!("Using search query: {}", query);


    tracing::info!("Generating embedding vector for serach query");
    // generate embedding for query
    let client = reqwest::Client::new();
    let embedding = embedding::generate_embedding(&client, &query, &gemini_key).await.unwrap();

    let mut similarities = embeddings.iter().map(|embedded| {
        let similarity = cosine_similarity(&embedding, &embedded.embedding);
        (similarity, &embedded.chunk)
    }).collect::<Vec<_>>();

    tracing::info!("Starting K Nearest Neighbors search using cosine similarity");

    let sorted = similarities.as_mut_slice();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // print top 5 results
    let top5 = sorted.iter().take(5).map(|(similarity, chunk)| {
        *chunk
    }).collect::<Vec<_>>();

    tracing::info!("Found top 5, generating context.");

    let context_for_prompt = top5.iter().map(|chunk| {
        format!("Context from mieux vivre: {}\n\n", chunk.text)
    }).collect::<String>();

    let prompt = format!(
        "Using the following context from mieux vivre:\n\n{}\n\nWhat is the answer to this user query: {}",
        context_for_prompt,
        query
    );

    let answer = llm::chat(&gemini_key, &prompt).await.unwrap();

    let context_metadata = top5.iter().map(|chunk| {
        format!("Title: {}\nSection: {}\nSubsection: {}\nURL: {}\n\n", chunk.metadata.title, chunk.metadata.section, chunk.metadata.subsection, chunk.metadata.url)
    }).unique().collect::<String>();

    let answer_with_sources = format!("{}\n\nSources:\n\n{}", answer, context_metadata);

    println!("\n\n");
    println!("{}", answer_with_sources);

}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
    let norm_a = a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|b| b * b).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}