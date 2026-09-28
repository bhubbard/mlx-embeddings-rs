//! Rigorous Semantic Metric Accuracy & Embedding Invariant Tests
//! Evaluates Cauchy-Schwarz cosine similarity axioms, L2 unit sphere normalization, and ColBERT MaxSim bounds.

use mlx_embeddings_rs::pooling::PoolingEngine;
use mlx_embeddings_rs::similarity::Similarity;
use mlx_rs::Array;

#[test]
fn test_cosine_similarity_metric_axioms() {
    let u = Array::from_slice(&[3.0f32, -4.0, 12.0], &[1, 3]);
    let v = Array::from_slice(&[1.0f32, 2.0, 2.0], &[1, 3]);
    let neg_u = Array::from_slice(&[-3.0f32, 4.0, -12.0], &[1, 3]);

    // 1. Identity / Self-Similarity: cos(u, u) = 1.0
    let sim_uu = Similarity::cosine_similarity(&u, &u).unwrap();
    assert!(
        (sim_uu - 1.0).abs() < 1e-5,
        "Self-similarity must be exactly 1.0, got {sim_uu}"
    );

    // 2. Symmetry: cos(u, v) == cos(v, u)
    let sim_uv = Similarity::cosine_similarity(&u, &v).unwrap();
    let sim_vu = Similarity::cosine_similarity(&v, &u).unwrap();
    assert!(
        (sim_uv - sim_vu).abs() < 1e-5,
        "Cosine similarity must be symmetric: {sim_uv} vs {sim_vu}"
    );

    // 3. Exact Inversion: cos(u, -u) = -1.0
    let sim_neg = Similarity::cosine_similarity(&u, &neg_u).unwrap();
    assert!(
        (sim_neg - (-1.0)).abs() < 1e-5,
        "Opposite vector similarity must be -1.0, got {sim_neg}"
    );

    // 4. Analytical Cauchy-Schwarz Value:
    // u = (3, -4, 12) -> |u| = sqrt(9 + 16 + 144) = sqrt(169) = 13
    // v = (1, 2, 2)   -> |v| = sqrt(1 + 4 + 4) = sqrt(9) = 3
    // u . v = 3*1 + (-4)*2 + 12*2 = 3 - 8 + 24 = 19
    // cos(u, v) = 19 / (13 * 3) = 19 / 39 = 0.487179...
    let expected_cos = 19.0f32 / 39.0f32;
    assert!(
        (sim_uv - expected_cos).abs() < 1e-4,
        "Cosine similarity error: got {sim_uv}, expected {expected_cos}"
    );
}

#[test]
fn test_l2_normalization_unit_sphere_invariant() {
    let vecs = [
        vec![10.0f32, 20.0, -30.0, 40.0],
        vec![0.001f32, -0.002, 0.005, 0.001],
        vec![1e4f32, 2e4, 3e4, 4e4],
    ];

    for data in vecs {
        let arr = Array::from_slice(&data, &[1, 4]);
        let normalized = PoolingEngine::normalize_l2(&arr, 1e-12).unwrap();

        let norm_slice = normalized.as_slice::<f32>();
        let l2_norm: f32 = norm_slice.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (l2_norm - 1.0).abs() < 1e-5,
            "Normalized vector must have unit L2 norm = 1.0, got {l2_norm}"
        );
    }
}

#[test]
fn test_colbert_maxsim_analytical_bound() {
    // 3 query tokens, 4 document tokens, dimension 3
    let q_tokens = vec![
        1.0f32, 0.0, 0.0, // q0: matches d0 exactly
        0.0, 1.0, 0.0,    // q1: matches d1 exactly
        0.0, 0.0, 1.0,    // q2: matches d2 exactly
    ];
    let query = Array::from_slice(&q_tokens, &[3, 3]);

    let d_tokens = vec![
        1.0f32, 0.0, 0.0,  // d0: (1, 0, 0)
        0.0, 1.0, 0.0,     // d1: (0, 1, 0)
        0.0, 0.0, 1.0,     // d2: (0, 0, 1)
        0.5, 0.5, 0.5,     // d3: (0.5, 0.5, 0.5)
    ];
    let doc = Array::from_slice(&d_tokens, &[4, 3]);

    // MaxSim score = max_j(q0.dj) + max_j(q1.dj) + max_j(q2.dj)
    // = 1.0 + 1.0 + 1.0 = 3.0
    let maxsim = PoolingEngine::colbert_maxsim(&query, &doc).unwrap();
    assert!(
        (maxsim - 3.0).abs() < 1e-4,
        "ColBERT MaxSim analytical score error: got {maxsim}, expected 3.0"
    );
}

#[test]
fn test_mean_pooling_zero_pad_isolation() {
    // 2 tokens: token 0 is real (10.0, 20.0), token 1 is garbage pad (999.0, 999.0)
    let token_data = vec![10.0f32, 20.0, 999.0, 999.0];
    let tokens = Array::from_slice(&token_data, &[1, 2, 2]);

    let mask_data = vec![1.0f32, 0.0];
    let mask = Array::from_slice(&mask_data, &[1, 2]);

    let pooled = PoolingEngine::mean_pooling(&tokens, &mask).unwrap();
    let slice = pooled.as_slice::<f32>();

    assert!(
        (slice[0] - 10.0).abs() < 1e-4,
        "Pad token leaked into pooled output 0: got {}",
        slice[0]
    );
    assert!(
        (slice[1] - 20.0).abs() < 1e-4,
        "Pad token leaked into pooled output 1: got {}",
        slice[1]
    );
}
