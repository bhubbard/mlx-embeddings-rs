use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::{softmax_axis, transpose_axes};
use mlx_rs::Array;
use crate::config::EmbeddingModelConfig;
use crate::error::Result;

pub struct BertSelfAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl BertSelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            q_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            k_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            v_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            out_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = hidden_states.shape();
        let bsz = shape[0] as usize;
        let seq_len = shape[1] as usize;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q_s = q.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let q_h = transpose_axes(&q_s, &[0, 2, 1, 3])?;

        let k_s = k.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let k_h = transpose_axes(&k_s, &[0, 2, 1, 3])?;

        let v_s = v.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let v_h = transpose_axes(&v_s, &[0, 2, 1, 3])?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&k_h, &[0, 1, 3, 2])?;
        let mut scores = q_h.matmul(&k_t)?.multiply(&scale_arr)?;

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // mask: [bsz, seq_len] -> [bsz, 1, 1, seq_len]
            let mask_expanded = mask.reshape(&[bsz as i32, 1, 1, seq_len as i32])?;
            let one = Array::from_f32(1.0f32);
            let neg_large = Array::from_f32(-10000.0f32);
            let additive = one.subtract(&mask_expanded)?.multiply(&neg_large)?;
            scores = scores.add(&additive)?;
        }

        let weights = softmax_axis(&scores, -1, None)?;
        let context = weights.matmul(&v_h)?;

        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        let full = context_t.reshape(&[bsz as i32, seq_len as i32, (self.num_heads * self.head_dim) as i32])?;

        Ok(self.out_proj.forward(&full)?)
    }
}

pub struct BertLayer {
    pub attention: BertSelfAttention,
    pub intermediate: Linear,
    pub output: Linear,
    pub hidden_size: usize,
}

impl BertLayer {
    pub fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            attention: BertSelfAttention::new(hidden_size, num_heads)?,
            intermediate: Linear::new(hidden_size as i32, intermediate_size as i32)?,
            output: Linear::new(intermediate_size as i32, hidden_size as i32)?,
            hidden_size,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        // Self attention + residual
        let attn_out = self.attention.forward(hidden_states, attention_mask)?;
        let res1 = hidden_states.add(&attn_out)?;
        let norm1 = layer_norm(&res1, 1e-6)?;

        // Feed forward + residual
        let inter = self.intermediate.forward(&norm1)?;
        let act = mlx_rs::nn::gelu(&inter)?;
        let ff_out = self.output.forward(&act)?;
        let res2 = norm1.add(&ff_out)?;
        layer_norm(&res2, 1e-6)
    }
}

pub struct BertEncoder {
    pub config: EmbeddingModelConfig,
    pub layers: Vec<BertLayer>,
    pub word_embeddings: Linear,
}

impl BertEncoder {
    pub fn new(config: EmbeddingModelConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
            )?);
        }

        let word_embeddings = Linear::new(config.hidden_size as i32, config.hidden_size as i32)?;

        Ok(Self {
            config,
            layers,
            word_embeddings,
        })
    }

    pub fn forward(
        &mut self,
        input_embeds: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        let mut h = self.word_embeddings.forward(input_embeds)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, attention_mask)?;
        }

        Ok(h)
    }
}

pub fn layer_norm(x: &Array, eps: f32) -> Result<Array> {
    let mean = x.mean_axis(-1, true)?;
    let diff = x.subtract(&mean)?;
    let diff_sq = diff.square()?;
    let var = diff_sq.mean_axis(-1, true)?;
    let eps_arr = Array::from_f32(eps);
    let std = var.add(&eps_arr)?.sqrt()?;
    Ok(diff.divide(&std)?)
}
