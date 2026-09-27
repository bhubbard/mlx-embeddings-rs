pub mod config;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod pooling;
pub mod similarity;

pub use config::{EmbeddingModelConfig, ModelArch, PoolingMode};
pub use error::{EmbeddingsError, Result};
pub use pipeline::EmbeddingPipeline;
pub use pooling::PoolingEngine;
pub use similarity::{SearchResult, Similarity};
