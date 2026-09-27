use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::Array;
use image::GenericImageView;
use crate::config::EmbeddingModelConfig;
use crate::error::Result;
use crate::models::bert::BertLayer;
use crate::pooling::PoolingEngine;

pub struct SigLipVisionTransformer {
    pub config: EmbeddingModelConfig,
    pub patch_proj: Linear,
    pub layers: Vec<BertLayer>,
    pub patch_size: usize,
    pub image_size: usize,
}

impl SigLipVisionTransformer {
    pub fn new(config: EmbeddingModelConfig) -> Result<Self> {
        let patch_size = config.patch_size.unwrap_or(16);
        let image_size = config.image_size.unwrap_or(224);
        let patch_dim = patch_size * patch_size * 3;

        let patch_proj = Linear::new(patch_dim as i32, config.hidden_size as i32)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
            )?);
        }

        Ok(Self {
            config,
            patch_proj,
            layers,
            patch_size,
            image_size,
        })
    }

    /// Extract image patches [1, num_patches, patch_size * patch_size * 3]
    pub fn extract_patches<I: GenericImageView<Pixel = image::Rgb<u8>>>(
        &self,
        image: &I,
    ) -> Result<Array> {
        let (w, h) = image.dimensions();
        let num_patches_x = (w as usize) / self.patch_size;
        let num_patches_y = (h as usize) / self.patch_size;
        let total_patches = num_patches_x * num_patches_y;
        let patch_dim = self.patch_size * self.patch_size * 3;

        let mut data = Vec::with_capacity(total_patches * patch_dim);

        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                for y in 0..self.patch_size {
                    for x in 0..self.patch_size {
                        let pixel = image.get_pixel((px * self.patch_size + x) as u32, (py * self.patch_size + y) as u32);
                        // Normalize pixel [0..255] to [-1.0, 1.0]
                        data.push((pixel[0] as f32 / 127.5) - 1.0);
                        data.push((pixel[1] as f32 / 127.5) - 1.0);
                        data.push((pixel[2] as f32 / 127.5) - 1.0);
                    }
                }
            }
        }

        Ok(Array::from_slice(&data, &[1, total_patches as i32, patch_dim as i32]))
    }

    /// Forward pass through SigLIP Vision Transformer
    pub fn forward<I: GenericImageView<Pixel = image::Rgb<u8>>>(
        &mut self,
        image: &I,
    ) -> Result<Array> {
        let patches = self.extract_patches(image)?;
        let mut h = self.patch_proj.forward(&patches)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, None)?;
        }

        // Global average pooling across patches
        let seq_len = h.shape()[1] as usize;
        let mask = Array::ones::<f32>(&[1, seq_len as i32])?;
        let pooled = PoolingEngine::mean_pooling(&h, &mask)?;

        if self.config.normalize {
            PoolingEngine::normalize_l2(&pooled, 1e-9)
        } else {
            Ok(pooled)
        }
    }
}
