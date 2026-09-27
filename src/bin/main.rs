use clap::{Parser, Subcommand, ValueEnum};
use mlx_embeddings_rs::config::{EmbeddingModelConfig, PoolingMode};
use mlx_embeddings_rs::error::Result;
use mlx_embeddings_rs::pipeline::EmbeddingPipeline;
use mlx_embeddings_rs::similarity::Similarity;

#[derive(Parser, Debug)]
#[command(
    name = "mlx-embeddings",
    about = "Native Apple Silicon text and image embedding engine using MLX",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum ModelPreset {
    BgeSmall,
    ModernBert,
    SigLip,
    ColBert,
    Tiny,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum PoolingArg {
    Mean,
    Cls,
    Max,
    LastToken,
    ColBert,
}

impl From<PoolingArg> for PoolingMode {
    fn from(arg: PoolingArg) -> Self {
        match arg {
            PoolingArg::Mean => PoolingMode::Mean,
            PoolingArg::Cls => PoolingMode::Cls,
            PoolingArg::Max => PoolingMode::Max,
            PoolingArg::LastToken => PoolingMode::LastToken,
            PoolingArg::ColBert => PoolingMode::ColBert,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate embeddings for an input text
    Embed {
        /// Text string to embed
        #[arg(short, long)]
        text: String,

        /// Model preset
        #[arg(short, long, value_enum, default_value_t = ModelPreset::ModernBert)]
        model: ModelPreset,

        /// Pooling strategy
        #[arg(short, long, value_enum)]
        pooling: Option<PoolingArg>,
    },

    /// Compute cosine similarity between two sentences
    Similarity {
        /// First text
        #[arg(short = 'a', long)]
        text1: String,

        /// Second text
        #[arg(short = 'b', long)]
        text2: String,

        /// Model preset
        #[arg(short, long, value_enum, default_value_t = ModelPreset::ModernBert)]
        model: ModelPreset,
    },

    /// Perform semantic search over a collection of documents
    Search {
        /// Search query
        #[arg(short, long)]
        query: String,

        /// Documents to rank (comma separated)
        #[arg(short, long, value_delimiter = ',')]
        docs: Vec<String>,

        /// Top K results
        #[arg(short, long, default_value_t = 3)]
        top_k: usize,
    },

    /// Display model architecture and pooling specs
    Info {
        /// Model preset
        #[arg(short, long, value_enum, default_value_t = ModelPreset::ModernBert)]
        model: ModelPreset,
    },
}

fn get_config(preset: ModelPreset, pooling: Option<PoolingArg>) -> EmbeddingModelConfig {
    let mut config = match preset {
        ModelPreset::BgeSmall => EmbeddingModelConfig::bge_small_en(),
        ModelPreset::ModernBert => EmbeddingModelConfig::modern_bert_base(),
        ModelPreset::SigLip => EmbeddingModelConfig::siglip_base_patch16_224(),
        ModelPreset::ColBert => EmbeddingModelConfig::colbert_v2(),
        ModelPreset::Tiny => EmbeddingModelConfig::tiny(),
    };

    if let Some(p) = pooling {
        config.pooling_mode = p.into();
    }
    config
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed { text, model, pooling } => {
            let config = get_config(model, pooling);
            let mut pipeline = EmbeddingPipeline::new(config.clone())?;

            println!("⚡ mlx-embeddings: Apple Silicon Embedding Engine");
            println!("  Model:        {}", config.name);
            println!("  Hidden Dim:   {}", config.hidden_size);
            println!("  Pooling:      {:?}", config.pooling_mode);
            println!("  Input Text:   \"{}\"", text);

            let emb = pipeline.embed_text(&text)?;
            let total = emb.size();
            let flat = emb.reshape(&[total as i32])?;
            let slice = flat.as_slice::<f32>();

            println!("✓ Embedding generated: shape {:?}", emb.shape());
            println!("  First 5 dimensions: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                slice[0], slice[1], slice[2], slice[3], slice[4]);
        }

        Commands::Similarity { text1, text2, model } => {
            let config = get_config(model, None);
            let mut pipeline = EmbeddingPipeline::new(config)?;

            let emb1 = pipeline.embed_text(&text1)?;
            let emb2 = pipeline.embed_text(&text2)?;

            let score = Similarity::cosine_similarity(&emb1, &emb2)?;
            println!("⚡ Semantic Similarity:");
            println!("  Text A:      \"{}\"", text1);
            println!("  Text B:      \"{}\"", text2);
            println!("  Cosine Score: {:.4} (range: -1.0 to 1.0)", score);
        }

        Commands::Search { query, docs, top_k } => {
            let config = get_config(ModelPreset::ModernBert, None);
            let mut pipeline = EmbeddingPipeline::new(config)?;

            println!("⚡ Semantic Search Query: \"{}\"", query);
            println!("  Corpus Size: {} documents", docs.len());

            let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            let results = pipeline.search(&query, &doc_refs, top_k)?;

            println!("\nTop Results:");
            for (rank, r) in results.iter().enumerate() {
                println!("  {}. [Score: {:.4}] \"{}\"", rank + 1, r.score, docs[r.index]);
            }
        }

        Commands::Info { model } => {
            let config = get_config(model, None);
            println!("Model Configuration for {}:", config.name);
            println!("  Architecture:          {:?}", config.arch);
            println!("  Hidden Dimension:      {}", config.hidden_size);
            println!("  Attention Heads:       {}", config.num_attention_heads);
            println!("  Hidden Layers:         {}", config.num_hidden_layers);
            println!("  Intermediate Size:     {}", config.intermediate_size);
            println!("  Default Pooling:       {:?}", config.pooling_mode);
            println!("  L2 Normalization:      {}", config.normalize);
        }
    }

    Ok(())
}
