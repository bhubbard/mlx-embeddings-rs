use mlx_rs::Array;
use crate::error::Result;
use crate::pooling::PoolingEngine;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
}

pub struct Similarity;

impl Similarity {
    /// Compute cosine similarity between two normalized or unnormalized 1D vectors
    pub fn cosine_similarity(a: &Array, b: &Array) -> Result<f32> {
        let norm_a = PoolingEngine::normalize_l2(a, 1e-9)?;
        let norm_b = PoolingEngine::normalize_l2(b, 1e-9)?;

        let size = norm_a.size();
        let flat_a = norm_a.reshape(&[size as i32])?;
        let flat_b = norm_b.reshape(&[size as i32])?;

        let slice_a = flat_a.as_slice::<f32>();
        let slice_b = flat_b.as_slice::<f32>();

        let mut dot = 0.0f32;
        for i in 0..size {
            dot += slice_a[i] * slice_b[i];
        }

        Ok(dot)
    }

    /// Compute cosine similarity matrix between queries [Q, D] and docs [N, D]
    pub fn cosine_similarity_matrix(queries: &Array, docs: &Array) -> Result<Array> {
        let norm_q = PoolingEngine::normalize_l2(queries, 1e-9)?;
        let norm_d = PoolingEngine::normalize_l2(docs, 1e-9)?;

        let d_t = norm_d.transpose()?;
        Ok(norm_q.matmul(&d_t)?)
    }

    /// Find Top-K most similar documents for a query embedding
    pub fn top_k(
        query: &Array,
        corpus: &Array,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let scores_mat = Self::cosine_similarity_matrix(query, corpus)?;
        let total = scores_mat.size();
        let flat = scores_mat.reshape(&[total as i32])?;
        let slice = flat.as_slice::<f32>();

        let mut indexed: Vec<SearchResult> = slice
            .iter()
            .enumerate()
            .map(|(i, &s)| SearchResult { index: i, score: s })
            .collect();

        // Sort descending by score
        indexed.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        Ok(indexed)
    }
}
