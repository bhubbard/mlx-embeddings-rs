use mlx_embeddings_rs::config::EmbeddingModelConfig;
use mlx_embeddings_rs::pipeline::EmbeddingPipeline;

#[test]
fn test_pipeline_embed_text() {
    let config = EmbeddingModelConfig::tiny();
    let mut pipeline = EmbeddingPipeline::new(config).unwrap();

    let emb = pipeline.embed_text("Deep learning on Apple Silicon unified memory").unwrap();
    assert_eq!(emb.shape(), &[1, 64]);

    // Check that output is L2 normalized
    let flat = emb.reshape(&[64]).unwrap();
    let slice = flat.as_slice::<f32>();
    let mut sum_sq = 0.0f32;
    for &x in slice {
        sum_sq += x * x;
    }
    assert!((sum_sq.sqrt() - 1.0).abs() < 1e-4);
}

#[test]
fn test_pipeline_search() {
    let config = EmbeddingModelConfig::tiny();
    let mut pipeline = EmbeddingPipeline::new(config).unwrap();

    let query = "machine learning and artificial intelligence";
    let corpus = vec![
        "artificial intelligence algorithms",
        "making sourdough bread at home",
        "deep neural networks on GPU",
    ];

    let results = pipeline.search(query, &corpus, 2).unwrap();
    assert_eq!(results.len(), 2);
    // Highest score should be >= second highest score
    assert!(results[0].score >= results[1].score);
}
