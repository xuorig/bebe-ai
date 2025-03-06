use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::document::Chunk;

pub mod similarity;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk<M> {
    pub embedding: Vec<f32>,
    pub chunk: Chunk<M>,
}

pub async fn get_embedded_chunks<M>(chunks: Vec<Chunk<M>>) -> Vec<EmbeddedChunk<M>> {
    let client = reqwest::Client::new();
    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap();

    let embeddings = generate_batch_embeddings(&client, &chunks, &gemini_key)
        .await
        .unwrap();

    chunks
        .into_iter()
        .zip(embeddings)
        .map(|(chunk, embedding)| EmbeddedChunk {
            embedding: embedding.values,
            chunk: chunk,
        })
        .collect()
}

pub async fn generate_embedding(
    client: &reqwest::Client,
    text: &str,
    gemini_key: &str,
) -> Result<Vec<f32>, reqwest::Error> {
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={}", gemini_key);
    let payload = json!(
        {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        }
    );
    let response = client.post(url).json(&payload).send().await?;
    let response: GeminiEmbeddingResponse = response.json().await?;
    Ok(response.embedding.values)
}

async fn generate_batch_embeddings<M>(
    client: &reqwest::Client,
    chunks: &Vec<Chunk<M>>,
    gemini_key: &str,
) -> Result<Vec<GeminiEmbedding>, reqwest::Error> {
    let batches: Vec<&[Chunk<M>]> = chunks.chunks(100).collect::<Vec<_>>();

    let mut embeddings = vec![];

    for (id, batch) in batches.iter().enumerate() {
        tracing::info!(
            "Generating embeddings for batch {} of {}",
            id + 1,
            batches.len()
        );

        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={}", gemini_key);

        let payload = GeminiBatchEmbeddingRequest {
            requests: batch
                .iter()
                .map(|chunk| BatchEmbeddingRequest {
                    model: "models/text-embedding-004".to_string(),
                    content: EmbeddingContent {
                        parts: vec![EmbeddingPart {
                            text: chunk.text.clone(),
                        }],
                    },
                })
                .collect(),
        };

        let response = client.post(url).json(&payload).send().await?;
        let response: GeminiBatchEmbeddingResponse = response.json().await?;
        embeddings.extend(response.embeddings);
    }

    Ok(embeddings)
}

#[derive(Debug, Deserialize)]
struct GeminiBatchEmbeddingResponse {
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiBatchEmbeddingRequest {
    requests: Vec<BatchEmbeddingRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchEmbeddingRequest {
    model: String,
    content: EmbeddingContent,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingContent {
    parts: Vec<EmbeddingPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbeddingResponse {
    embedding: GeminiEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}
