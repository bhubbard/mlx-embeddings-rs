use mlx_rs::Array;
use mlx_embeddings_rs::pooling::PoolingEngine;
use mlx_embeddings_rs::similarity::Similarity;

#[test]
fn test_cosine_similarity_identical() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
    let score = Similarity::cosine_similarity(&a, &a).unwrap();
    assert!((score - 1.0).abs() < 1e-4);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
    let b = Array::from_slice(&[0.0f32, 1.0], &[1, 2]);
    let score = Similarity::cosine_similarity(&a, &b).unwrap();
    assert!(score.abs() < 1e-5);
}

#[test]
fn test_colbert_maxsim() {
    // 2 query tokens, 3 doc tokens, dim = 2
    let q_data = vec![1.0f32, 0.0, 0.0, 1.0];
    let query = Array::from_slice(&q_data, &[2, 2]);

    let d_data = vec![1.0f32, 0.0, 0.5, 0.5, 0.0, 1.0];
    let doc = Array::from_slice(&d_data, &[3, 2]);

    let score = PoolingEngine::colbert_maxsim(&query, &doc).unwrap();
    // q0 [1, 0] max dot with d is d0 [1, 0] = 1.0
    // q1 [0, 1] max dot with d is d2 [0, 1] = 1.0
    // Total MaxSim = 1.0 + 1.0 = 2.0
    assert!((score - 2.0).abs() < 1e-4);
}
