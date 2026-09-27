use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingMode {
    Mean,
    Cls,
    Max,
    LastToken,
    ColBert,
}

impl Default for PoolingMode {
    fn default() -> Self {
        Self::Mean
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArch {
    ModernBert,
    Bert,
    SigLip,
    ColQwen,
}

impl Default for ModelArch {
    fn default() -> Self {
        Self::ModernBert
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelConfig {
    pub name: String,
    pub arch: ModelArch,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub pooling_mode: PoolingMode,
    pub normalize: bool,
    pub image_size: Option<usize>,
    pub patch_size: Option<usize>,
}

impl EmbeddingModelConfig {
    pub fn bge_small_en() -> Self {
        Self {
            name: "bge-small-en-v1.5".to_string(),
            arch: ModelArch::Bert,
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 1536,
            max_position_embeddings: 512,
            pooling_mode: PoolingMode::Cls,
            normalize: true,
            image_size: None,
            patch_size: None,
        }
    }

    pub fn modern_bert_base() -> Self {
        Self {
            name: "modernbert-base".to_string(),
            arch: ModelArch::ModernBert,
            vocab_size: 50368,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 8192,
            pooling_mode: PoolingMode::Mean,
            normalize: true,
            image_size: None,
            patch_size: None,
        }
    }

    pub fn siglip_base_patch16_224() -> Self {
        Self {
            name: "siglip-base-patch16-224".to_string(),
            arch: ModelArch::SigLip,
            vocab_size: 32000,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 64,
            pooling_mode: PoolingMode::Mean,
            normalize: true,
            image_size: Some(224),
            patch_size: Some(16),
        }
    }

    pub fn colbert_v2() -> Self {
        Self {
            name: "colbert-v2.0".to_string(),
            arch: ModelArch::Bert,
            vocab_size: 30522,
            hidden_size: 128,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 768,
            max_position_embeddings: 512,
            pooling_mode: PoolingMode::ColBert,
            normalize: true,
            image_size: None,
            patch_size: None,
        }
    }

    pub fn tiny() -> Self {
        Self {
            name: "tiny-embedding".to_string(),
            arch: ModelArch::Bert,
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 128,
            max_position_embeddings: 128,
            pooling_mode: PoolingMode::Mean,
            normalize: true,
            image_size: Some(32),
            patch_size: Some(16),
        }
    }
}
