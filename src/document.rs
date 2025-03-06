use serde::{Deserialize, Serialize};

pub mod mv;

pub trait DocumentFetcher<M> {
    #[allow(async_fn_in_trait)]
    async fn fetch(&self) -> Result<Vec<Chunk<M>>, Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk<M> {
    pub text: String,
    pub metadata: M,
}

// tests
#[cfg(test)]
mod tests {
    use super::{Chunk, DocumentFetcher};

    struct SimpleFetcher;

    impl DocumentFetcher<()> for SimpleFetcher {
        async fn fetch(&self) -> Result<Vec<Chunk<()>>, Box<dyn std::error::Error>> {
            Ok(vec![Chunk {
                text: "Hello, world!".to_string(),
                metadata: (),
            }])
        }
    }

    #[tokio::test]
    async fn test_basic_fetch() {
        let fetcher = SimpleFetcher;
        let documents = fetcher.fetch().await.unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "Hello, world!");
    }
}
