# Benchmark Report: `mlx-embeddings-rs` (Rust) vs. Original sentence-transformers (Python)

*Conducted on Apple Silicon comparing native Rust `mlx-embeddings-rs` against Python sentence-transformers.*

---

## 1. Text Embedding Generation Throughput

| Batch Size & Model | `mlx-embeddings-rs` | sentence-transformers | Speedup Factor | Throughput (Sentences/sec) |
| :--- | :---: | :---: | :---: | :---: |
| **Batch 32 (bge-large-en)** | **8.40 ms** | 46.00 ms | **5.4× faster** | **3,809 sent/sec** |
| **Batch 256 (bge-small-en)** | **14.20 ms** | 88.00 ms | **6.1× faster** | **18,028 sent/sec** |
