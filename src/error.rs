use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingsError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Unsupported pooling mode: {0}")]
    UnsupportedPooling(String),

    #[error("Model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingsError>;
