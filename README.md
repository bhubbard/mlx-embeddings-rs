# mlx-embeddings-rs ⚡

[![CI](https://github.com/bhubbard/mlx-embeddings-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/mlx-embeddings-rs/actions/workflows/ci.yml)
[![Pages](https://github.com/bhubbard/mlx-embeddings-rs/actions/workflows/pages.yml/badge.svg)](https://github.com/bhubbard/mlx-embeddings-rs/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Native Apple Silicon Rust engine for high-performance text and image embeddings, powered by MLX.

Interactive documentation & pooling playground: **[code.brandonhubbard.com/mlx-embeddings-rs](http://code.brandonhubbard.com/mlx-embeddings-rs/)**

---

## Highlights

- **Multi-Model Support**: ModernBERT, BERT, RoBERTa, Qwen, and SigLIP Vision Transformer.
- **Comprehensive Pooling Engine**:
  - `Mean Pooling`: Attention-masked weighted averaging with epsilon guard.
  - `CLS Pooling`: First attended token selection supporting both right-padded and decoder-style left-padded inputs.
  - `Max Pooling`: Feature-wise maximum with masked $-\infty$ suppression.
  - `Last-Token Pooling`: Final attended token extraction.
  - `ColBERT v2 MaxSim`: Multi-vector token-level late interaction scoring $\sum_{q \in Q} \max_{d \in D} (q \cdot d^T)$.
- **Zero Python Runtime**: 100% native Rust on Apple Silicon unified memory with Metal acceleration.
- **Vector Search & Similarity**: Fast cosine similarity matrix computation, dot product, and top-$k$ semantic search ranking.

---

## Installation

```bash
# Clone and build
git clone https://github.com/bhubbard/mlx-embeddings-rs.git
cd mlx-embeddings-rs
cargo build --release
```

---

## CLI Usage

### Generate Embeddings
```bash
mlx-embeddings embed \
  --text "High-speed dense vector embeddings on Apple Silicon unified memory" \
  --model modern-bert \
  --pooling mean
```

### Compute Cosine Similarity
```bash
mlx-embeddings similarity \
  -a "Apple Silicon GPU matrix operations" \
  -b "Metal performance acceleration on Mac" \
  --model modern-bert
```

### Top-K Semantic Search
```bash
mlx-embeddings search \
  --query "Machine learning framework" \
  --docs "Rust on Apple Silicon,Baking sourdough bread,Deep neural network library" \
  --top-k 2
```

### Inspect Architecture Specs
```bash
mlx-embeddings info --model modern-bert
```

---

## Rust Library API

```rust
use mlx_embeddings_rs::config::{EmbeddingModelConfig, PoolingMode};
use mlx_embeddings_rs::pipeline::EmbeddingPipeline;
use mlx_embeddings_rs::similarity::Similarity;

fn main() -> mlx_embeddings_rs::Result<()> {
    // Initialize pipeline with ModernBERT
    let config = EmbeddingModelConfig::modern_bert_base();
    let mut pipeline = EmbeddingPipeline::new(config)?;

    // Embed single sentence
    let emb = pipeline.embed_text("High-performance embeddings on Apple Silicon")?;
    println!("Embedding shape: {:?}", emb.shape());

    // Batch embedding
    let corpus = vec![
        "Unified memory on Mac",
        "Deep learning inference in Rust",
        "Artisan bread baking recipes",
    ];
    let matrix = pipeline.embed_batch(&corpus)?;
    println!("Batch matrix shape: {:?}", matrix.shape());

    // Semantic search
    let results = pipeline.search("Apple M4 GPU acceleration", &corpus, 2)?;
    for r in results {
        println!("Match: {} (Score: {:.4})", corpus[r.index], r.score);
    }

    Ok(())
}
```

---

## Test Suite

```bash
cargo test
```

Passes all 11 unit and pipeline integration tests covering exact value pooling verification, left/right padding, L2 normalization, cosine similarity, ColBERT MaxSim, and semantic search.

---

## License

Dual licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

Ported from [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings).
