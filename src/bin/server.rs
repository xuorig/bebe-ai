use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Query, State},
    routing::get,
    Router,
};
use bebe_ai::{
    embedding::{self, similarity::SimilarityFinder},
    llm,
};
use itertools::Itertools;
use tower_http::services::ServeDir;

#[derive(Debug, Clone)]
struct AppState {
    embeddings:
        Arc<Vec<bebe_ai::embedding::EmbeddedChunk<bebe_ai::document::mv::MieuxVivreMetadata>>>,
    gemini_key: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let embeddings_json = std::fs::read("embedded.json").unwrap();
    let embeddings: Vec<
        bebe_ai::embedding::EmbeddedChunk<bebe_ai::document::mv::MieuxVivreMetadata>,
    > = serde_json::from_slice(&embeddings_json).unwrap();
    tracing::info!("Loaded {} embeddings", embeddings.len());

    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap();

    let serve_dir = ServeDir::new("public");
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/chat", get(handle_chat))
        .fallback_service(serve_dir)
        .with_state(AppState {
            embeddings: Arc::new(embeddings),
            gemini_key,
        });

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat(
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> String {
    tracing::info!("Generating search query from user query");

    let query = params.get("query").unwrap();

    tracing::info!("User query: {}", query);

    // convert user query to search query
    let query = query.trim().to_string();
    let query = llm::chat(&state.gemini_key, &format!(
        "Convert the following user query to a search query: {}. Only respond with the search query, nothing else.", query
    )).await.unwrap();

    tracing::info!("Using search query: {}", query);

    tracing::info!("Generating embedding vector for serach query");
    // generate embedding for query
    let client = reqwest::Client::new();
    let embedding = embedding::generate_embedding(&client, &query, &state.gemini_key)
        .await
        .unwrap();

    let similarity = embedding::similarity::naive::NaiveSimilarity {};
    let top5 = similarity.find_k_similar(&embedding, state.embeddings.as_ref(), 5);

    tracing::info!("Found top 5, generating context.");

    let context_for_prompt = top5
        .iter()
        .map(|chunk| format!("Context from mieux vivre: {}\n\n", chunk.chunk.text))
        .collect::<String>();

    let prompt = format!(
        "Using the following context from mieux vivre:\n\n{}\n\nWhat is the answer to this user query: {}. Please quote the mieux vivre context in your answer when possible.",
        context_for_prompt,
        query
    );

    let answer = llm::chat(&state.gemini_key, &prompt).await.unwrap();

    let context_metadata = top5
        .iter()
        .map(|chunk| {
            format!(
                "Titre: {}\nSection: {}\nSous-section: {}\nURL: {}\n\n",
                chunk.chunk.metadata.title,
                chunk.chunk.metadata.section,
                chunk.chunk.metadata.subsection,
                chunk.chunk.metadata.url
            )
        })
        .unique()
        .collect::<String>();

    let answer_with_sources = format!("{}\n\nSources:\n\n{}", answer, context_metadata);

    answer_with_sources
}
