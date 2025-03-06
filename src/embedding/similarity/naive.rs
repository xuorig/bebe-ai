use crate::embedding::EmbeddedChunk;

use super::SimilarityFinder;

pub struct NaiveSimilarity {}

impl<M> SimilarityFinder<M> for NaiveSimilarity {
    fn find_k_similar<'a>(
        &self,
        embedding: &Vec<f32>,
        set: &'a Vec<EmbeddedChunk<M>>,
        k: usize,
    ) -> Vec<&'a EmbeddedChunk<M>> {
        let similarities = set.into_iter().map(|chunk| {
            let similarity = cosine_similarity(&chunk.embedding, &embedding);
            (chunk, similarity)
        });

        let mut sorted = similarities.collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        sorted
            .into_iter()
            .take(k)
            .map(|(chunk, _)| chunk)
            .collect::<Vec<_>>()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
    let norm_a = a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|b| b * b).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
