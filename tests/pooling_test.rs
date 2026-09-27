use mlx_rs::Array;
use mlx_embeddings_rs::pooling::PoolingEngine;

fn get_fixtures() -> (Array, Array) {
    // seq 0: 3 real tokens + 1 pad, seq 1: 4 real tokens, no pad
    // shape: [2, 4, 2]
    let token_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 99.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
    ];
    let token_embeddings = Array::from_slice(&token_data, &[2, 4, 2]);

    // attention mask: [2, 4]
    let mask_data = vec![
        1.0f32, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    let attention_mask = Array::from_slice(&mask_data, &[2, 4]);

    (token_embeddings, attention_mask)
}

#[test]
fn test_cls_pooling_exact_values() {
    let (tokens, mask) = get_fixtures();
    let result = PoolingEngine::cls_pooling(&tokens, &mask).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    let flat = result.reshape(&[4]).unwrap();
    let slice = flat.as_slice::<f32>();
    let expected = [1.0f32, 2.0, 10.0, 20.0];

    for (a, b) in slice.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "CLS mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_max_pooling_exact_values() {
    let (tokens, mask) = get_fixtures();
    let result = PoolingEngine::max_pooling(&tokens, &mask).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    let flat = result.reshape(&[4]).unwrap();
    let slice = flat.as_slice::<f32>();
    let expected = [5.0f32, 6.0, 70.0, 80.0];

    for (a, b) in slice.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "Max mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_mean_pooling_exact_values() {
    let (tokens, mask) = get_fixtures();
    let result = PoolingEngine::mean_pooling(&tokens, &mask).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    let flat = result.reshape(&[4]).unwrap();
    let slice = flat.as_slice::<f32>();
    let expected = [3.0f32, 4.0, 40.0, 50.0];

    for (a, b) in slice.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "Mean mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_lasttoken_pooling_exact_values() {
    let (tokens, mask) = get_fixtures();
    let result = PoolingEngine::lasttoken_pooling(&tokens, &mask).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    let flat = result.reshape(&[4]).unwrap();
    let slice = flat.as_slice::<f32>();
    let expected = [5.0f32, 6.0, 70.0, 80.0];

    for (a, b) in slice.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "LastToken mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_cls_left_padded_decoder_models() {
    let token_data = vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    let tokens = Array::from_slice(&token_data, &[2, 4, 1]);

    // left-padded mask: first 1 is at index 2 for row 0, index 1 for row 1
    let mask_data = vec![
        0.0f32, 0.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0,
    ];
    let mask = Array::from_slice(&mask_data, &[2, 4]);

    let result = PoolingEngine::cls_pooling(&tokens, &mask).unwrap();
    assert_eq!(result.shape(), &[2, 1]);

    let flat = result.reshape(&[2]).unwrap();
    let slice = flat.as_slice::<f32>();
    assert!((slice[0] - 3.0).abs() < 1e-4);
    assert!((slice[1] - 6.0).abs() < 1e-4);
}

#[test]
fn test_l2_normalization() {
    let sample = Array::from_slice(&[3.0f32, 4.0f32], &[1, 2]);
    let normalized = PoolingEngine::normalize_l2(&sample, 1e-9).unwrap();

    let flat = normalized.reshape(&[2]).unwrap();
    let slice = flat.as_slice::<f32>();

    // 3/5 = 0.6, 4/5 = 0.8
    assert!((slice[0] - 0.6).abs() < 1e-4);
    assert!((slice[1] - 0.8).abs() < 1e-4);

    let norm = (slice[0] * slice[0] + slice[1] * slice[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}
