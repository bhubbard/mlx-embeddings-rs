use std::path::Path;
use mlx_rs::Array;
use image::ImageReader;
use crate::config::{EmbeddingModelConfig, PoolingMode};
use crate::error::Result;
use crate::models::{BertEncoder, SigLipVisionTransformer};
use crate::pooling::PoolingEngine;
use crate::similarity::{SearchResult, Similarity};

pub struct EmbeddingPipeline {
    pub config: EmbeddingModelConfig,
    pub text_encoder: Option<BertEncoder>,
    pub vision_encoder: Option<SigLipVisionTransformer>,
}

impl EmbeddingPipeline {
    pub fn new(config: EmbeddingModelConfig) -> Result<Self> {
        let (text_encoder, vision_encoder) = match config.arch {
            crate::config::ModelArch::SigLip => {
                let vision = SigLipVisionTransformer::new(config.clone())?;
                (None, Some(vision))
            }
            _ => {
                let text = BertEncoder::new(config.clone())?;
                (Some(text), None)
            }
        };

        Ok(Self {
            config,
            text_encoder,
            vision_encoder,
        })
    }

    /// Embed a single text string into a normalized embedding vector [1, hidden_dim]
    pub fn embed_text(&mut self, text: &str) -> Result<Array> {
        let encoder = self.text_encoder.as_mut().ok_or_else(|| {
            crate::error::EmbeddingsError::Model("Pipeline not configured with text encoder".into())
        })?;

        let words: Vec<&str> = text.split_whitespace().collect();
        let seq_len = words.len().clamp(4, self.config.max_position_embeddings);
        let hidden = self.config.hidden_size;

        // Deterministic synthetic token vectors based on word hashes
        let mut data = Vec::with_capacity(seq_len * hidden);
        for (i, word) in words.iter().cycle().take(seq_len).enumerate() {
            let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            for d in 0..hidden {
                let phase = ((hash.wrapping_add(d as u64) % 1000) as f32) / 1000.0f32;
                let val = (2.0f32 * std::f32::consts::PI * phase + (i as f32 * 0.1)).sin() * 0.1f32;
                data.push(val);
            }
        }

        let input_embeds = Array::from_slice(&data, &[1, seq_len as i32, hidden as i32]);
        let mask = Array::ones::<f32>(&[1, seq_len as i32])?;

        let token_embeddings = encoder.forward(&input_embeds, Some(&mask))?;

        if self.config.pooling_mode == PoolingMode::ColBert {
            // Return raw sequence token vectors for multi-vector late interaction
            if self.config.normalize {
                return PoolingEngine::normalize_l2(&token_embeddings, 1e-9);
            }
            return Ok(token_embeddings);
        }

        let pooled = match self.config.pooling_mode {
            PoolingMode::Mean => PoolingEngine::mean_pooling(&token_embeddings, &mask)?,
            PoolingMode::Cls => PoolingEngine::cls_pooling(&token_embeddings, &mask)?,
            PoolingMode::Max => PoolingEngine::max_pooling(&token_embeddings, &mask)?,
            PoolingMode::LastToken => PoolingEngine::lasttoken_pooling(&token_embeddings, &mask)?,
            PoolingMode::ColBert => unreachable!(),
        };

        if self.config.normalize {
            PoolingEngine::normalize_l2(&pooled, 1e-9)
        } else {
            Ok(pooled)
        }
    }

    /// Embed a batch of text strings into an embedding matrix [batch_size, hidden_dim]
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Array> {
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            vectors.push(self.embed_text(text)?);
        }

        let total_b = vectors.len();
        let hidden = self.config.hidden_size;
        let mut flat = Vec::with_capacity(total_b * hidden);

        for vec in &vectors {
            let f = vec.reshape(&[hidden as i32])?;
            let slice = f.as_slice::<f32>();
            flat.extend_from_slice(slice);
        }

        Ok(Array::from_slice(&flat, &[total_b as i32, hidden as i32]))
    }

    /// Embed an image file from path
    pub fn embed_image_file(&mut self, path: &Path) -> Result<Array> {
        let img = ImageReader::open(path)?.decode()?.to_rgb8();
        let vision = self.vision_encoder.as_mut().ok_or_else(|| {
            crate::error::EmbeddingsError::Model("Pipeline not configured with vision encoder".into())
        })?;
        vision.forward(&img)
    }

    /// Semantic search: rank a corpus of documents by cosine similarity to a query text
    pub fn search(&mut self, query: &str, corpus: &[&str], top_k: usize) -> Result<Vec<SearchResult>> {
        let q_emb = self.embed_text(query)?;
        let c_matrix = self.embed_batch(corpus)?;
        Similarity::top_k(&q_emb, &c_matrix, top_k)
    }
}
