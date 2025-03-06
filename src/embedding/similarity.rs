use super::EmbeddedChunk;

pub mod naive;

pub trait SimilarityFinder<M> {
    fn find_k_similar<'a>(
        &self,
        embedding: &Vec<f32>,
        set: &'a Vec<EmbeddedChunk<M>>,
        k: usize,
    ) -> Vec<&'a EmbeddedChunk<M>>;
}
