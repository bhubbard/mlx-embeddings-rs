use mlx_rs::Array;
use crate::error::Result;

pub struct PoolingEngine;

impl PoolingEngine {
    /// Mean Pooling across valid tokens weighted by attention_mask:
    /// sum(embeddings * mask) / max(sum(mask), 1e-9)
    pub fn mean_pooling(
        token_embeddings: &Array,
        attention_mask: &Array,
    ) -> Result<Array> {
        let batch_size = token_embeddings.shape()[0] as usize;
        let seq_len = token_embeddings.shape()[1] as usize;
        let hidden_dim = token_embeddings.shape()[2] as usize;

        let tokens_flat = token_embeddings.reshape(&[(batch_size * seq_len * hidden_dim) as i32])?;
        let tokens_slice = tokens_flat.as_slice::<f32>();

        let mask_flat = attention_mask.reshape(&[(batch_size * seq_len) as i32])?;
        let mask_slice = mask_flat.as_slice::<f32>();

        let mut out = vec![0.0f32; batch_size * hidden_dim];

        for b in 0..batch_size {
            let mut sum_mask = 0.0f32;
            for s in 0..seq_len {
                sum_mask += mask_slice[b * seq_len + s];
            }
            let denom = sum_mask.max(1e-9);

            for d in 0..hidden_dim {
                let mut sum_emb = 0.0f32;
                for s in 0..seq_len {
                    let m = mask_slice[b * seq_len + s];
                    let val = tokens_slice[(b * seq_len + s) * hidden_dim + d];
                    sum_emb += val * m;
                }
                out[b * hidden_dim + d] = sum_emb / denom;
            }
        }

        Ok(Array::from_slice(&out, &[batch_size as i32, hidden_dim as i32]))
    }

    /// CLS Pooling: selects first attended token in attention_mask for each sequence.
    /// Supports both standard right-padded and decoder-style left-padded inputs.
    pub fn cls_pooling(
        token_embeddings: &Array,
        attention_mask: &Array,
    ) -> Result<Array> {
        let batch_size = token_embeddings.shape()[0] as usize;
        let seq_len = token_embeddings.shape()[1] as usize;
        let hidden_dim = token_embeddings.shape()[2] as usize;

        let tokens_flat = token_embeddings.reshape(&[(batch_size * seq_len * hidden_dim) as i32])?;
        let tokens_slice = tokens_flat.as_slice::<f32>();

        let mask_flat = attention_mask.reshape(&[(batch_size * seq_len) as i32])?;
        let mask_slice = mask_flat.as_slice::<f32>();

        let mut out = vec![0.0f32; batch_size * hidden_dim];

        for b in 0..batch_size {
            // Find first position where mask == 1
            let mut target_idx = 0usize;
            for s in 0..seq_len {
                if mask_slice[b * seq_len + s] > 0.5 {
                    target_idx = s;
                    break;
                }
            }

            for d in 0..hidden_dim {
                out[b * hidden_dim + d] = tokens_slice[(b * seq_len + target_idx) * hidden_dim + d];
            }
        }

        Ok(Array::from_slice(&out, &[batch_size as i32, hidden_dim as i32]))
    }

    /// Max Pooling: takes the maximum value along sequence dimension for each feature,
    /// suppressing masked tokens to -infinity.
    pub fn max_pooling(
        token_embeddings: &Array,
        attention_mask: &Array,
    ) -> Result<Array> {
        let batch_size = token_embeddings.shape()[0] as usize;
        let seq_len = token_embeddings.shape()[1] as usize;
        let hidden_dim = token_embeddings.shape()[2] as usize;

        let tokens_flat = token_embeddings.reshape(&[(batch_size * seq_len * hidden_dim) as i32])?;
        let tokens_slice = tokens_flat.as_slice::<f32>();

        let mask_flat = attention_mask.reshape(&[(batch_size * seq_len) as i32])?;
        let mask_slice = mask_flat.as_slice::<f32>();

        let mut out = vec![f32::NEG_INFINITY; batch_size * hidden_dim];

        for b in 0..batch_size {
            for d in 0..hidden_dim {
                let mut max_val = f32::NEG_INFINITY;
                for s in 0..seq_len {
                    let m = mask_slice[b * seq_len + s];
                    if m > 0.5 {
                        let val = tokens_slice[(b * seq_len + s) * hidden_dim + d];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                out[b * hidden_dim + d] = if max_val.is_finite() { max_val } else { 0.0 };
            }
        }

        Ok(Array::from_slice(&out, &[batch_size as i32, hidden_dim as i32]))
    }

    /// Last Token Pooling: selects the last non-padded token for each sequence.
    /// If all tokens are padded, returns zeros.
    pub fn lasttoken_pooling(
        token_embeddings: &Array,
        attention_mask: &Array,
    ) -> Result<Array> {
        let batch_size = token_embeddings.shape()[0] as usize;
        let seq_len = token_embeddings.shape()[1] as usize;
        let hidden_dim = token_embeddings.shape()[2] as usize;

        let tokens_flat = token_embeddings.reshape(&[(batch_size * seq_len * hidden_dim) as i32])?;
        let tokens_slice = tokens_flat.as_slice::<f32>();

        let mask_flat = attention_mask.reshape(&[(batch_size * seq_len) as i32])?;
        let mask_slice = mask_flat.as_slice::<f32>();

        let mut out = vec![0.0f32; batch_size * hidden_dim];

        for b in 0..batch_size {
            let mut last_idx = None;
            for s in (0..seq_len).rev() {
                if mask_slice[b * seq_len + s] > 0.5 {
                    last_idx = Some(s);
                    break;
                }
            }

            if let Some(idx) = last_idx {
                for d in 0..hidden_dim {
                    out[b * hidden_dim + d] = tokens_slice[(b * seq_len + idx) * hidden_dim + d];
                }
            }
        }

        Ok(Array::from_slice(&out, &[batch_size as i32, hidden_dim as i32]))
    }

    /// L2 normalize embeddings: x / max(||x||_2, eps)
    pub fn normalize_l2(embeddings: &Array, eps: f32) -> Result<Array> {
        let shape = embeddings.shape();
        let total = embeddings.size();
        let flat = embeddings.reshape(&[total as i32])?;
        let slice = flat.as_slice::<f32>();

        let last_dim = *shape.last().unwrap_or(&1) as usize;
        let num_vectors = total / last_dim;

        let mut out = vec![0.0f32; total];

        for v in 0..num_vectors {
            let base = v * last_dim;
            let mut norm_sq = 0.0f32;
            for d in 0..last_dim {
                let val = slice[base + d];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt().max(eps);

            for d in 0..last_dim {
                out[base + d] = slice[base + d] / norm;
            }
        }

        Ok(Array::from_slice(&out, shape))
    }

    /// ColBERT MaxSim operator for multi-vector late interaction:
    /// Score = sum_{q in Q} max_{d in D} (q . d^T)
    pub fn colbert_maxsim(query_tokens: &Array, doc_tokens: &Array) -> Result<f32> {
        // query_tokens: [Q_len, dim], doc_tokens: [D_len, dim]
        let q_shape = query_tokens.shape();
        let d_shape = doc_tokens.shape();

        let q_len = q_shape[0] as usize;
        let dim = q_shape[1] as usize;
        let d_len = d_shape[0] as usize;

        let q_flat = query_tokens.reshape(&[(q_len * dim) as i32])?;
        let q_slice = q_flat.as_slice::<f32>();

        let d_flat = doc_tokens.reshape(&[(d_len * dim) as i32])?;
        let d_slice = d_flat.as_slice::<f32>();

        let mut total_score = 0.0f32;

        for q in 0..q_len {
            let q_base = q * dim;
            let mut max_sim = f32::NEG_INFINITY;

            for d in 0..d_len {
                let d_base = d * dim;
                let mut dot = 0.0f32;
                for i in 0..dim {
                    dot += q_slice[q_base + i] * d_slice[d_base + i];
                }
                if dot > max_sim {
                    max_sim = dot;
                }
            }

            if max_sim.is_finite() {
                total_score += max_sim;
            }
        }

        Ok(total_score)
    }
}
