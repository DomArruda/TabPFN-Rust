use serde::{Deserialize, Serialize};

/// Mirrors `tabpfn/architectures/tabpfn_v3_5.py::TabPFNV3p5Config` defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabPFNV3p5Config {
    pub name: String,
    pub embed_dim: usize,
    pub dist_embed_num_blocks: usize,
    pub dist_embed_num_heads: usize,
    pub dist_embed_num_inducing_points: usize,
    pub feature_group_size: usize,
    pub feat_agg_num_blocks: usize,
    pub feat_agg_num_heads: usize,
    pub feat_agg_num_cls_tokens: usize,
    pub feat_agg_rope_base: f32,
    pub nlayers: usize,
    pub icl_num_heads: usize,
    pub icl_num_kv_heads: Option<usize>,
    pub icl_num_kv_heads_test: Option<usize>,
    pub decoder_head_dim: usize,
    pub decoder_num_heads: usize,
    pub decoder_use_softmax_scaling: bool,
    pub ff_factor: usize,
    pub softmax_scaling_mlp_hidden_dim: usize,
    pub fourier_encoding_num_frequencies: usize,
    pub cell_ecdf_num_frequencies: usize,
    pub cell_ecdf_num_buckets: usize,
    pub cell_embed_row_chunk_size: Option<usize>,
    pub inference_chunk_cells: usize,
    pub max_batched_estimator_rows: usize,
    pub max_batched_estimator_cells: usize,
    pub inference_col_chunk_size: usize,
    pub max_num_classes: usize,
    pub num_buckets: usize,
}

impl Default for TabPFNV3p5Config {
    fn default() -> Self {
        Self {
            name: "TabPFN-v3.5".into(),
            embed_dim: 128,
            dist_embed_num_blocks: 3,
            dist_embed_num_heads: 8,
            dist_embed_num_inducing_points: 128,
            feature_group_size: 3,
            feat_agg_num_blocks: 3,
            feat_agg_num_heads: 8,
            feat_agg_num_cls_tokens: 8,
            feat_agg_rope_base: 100_000.0,
            nlayers: 24,
            icl_num_heads: 16,
            icl_num_kv_heads: None,
            icl_num_kv_heads_test: Some(1),
            decoder_head_dim: 64,
            decoder_num_heads: 6,
            decoder_use_softmax_scaling: true,
            ff_factor: 2,
            softmax_scaling_mlp_hidden_dim: 64,
            fourier_encoding_num_frequencies: 32,
            cell_ecdf_num_frequencies: 4,
            cell_ecdf_num_buckets: 8192,
            cell_embed_row_chunk_size: Some(2048),
            inference_chunk_cells: 2048 * 768,
            max_batched_estimator_rows: 32768,
            max_batched_estimator_cells: 768_000_000,
            inference_col_chunk_size: 4,
            max_num_classes: 10,
            num_buckets: 1000,
        }
    }
}

impl TabPFNV3p5Config {
    pub fn icl_emsize(&self) -> usize {
        self.embed_dim * self.feat_agg_num_cls_tokens
    }
    pub fn icl_head_dim(&self) -> usize {
        self.icl_emsize() / self.icl_num_heads
    }
    pub fn dist_head_dim(&self) -> usize {
        self.embed_dim / self.dist_embed_num_heads
    }
    pub fn agg_head_dim(&self) -> usize {
        self.embed_dim / self.feat_agg_num_heads
    }
    pub fn validate(&self) -> crate::Result<()> {
        if self.fourier_encoding_num_frequencies < 1 || self.cell_ecdf_num_frequencies < 1 {
            return Err(crate::Error::Config("fourier frequencies must be >= 1".into()));
        }
        if self.cell_ecdf_num_buckets < 2 {
            return Err(crate::Error::Config("cell_ecdf_num_buckets must be >= 2".into()));
        }
        if let Some(kv) = self.icl_num_kv_heads {
            if self.icl_num_heads % kv != 0 {
                return Err(crate::Error::Config("icl_num_heads must divide by icl_num_kv_heads".into()));
            }
        }
        if let Some(kv) = self.icl_num_kv_heads_test {
            if self.icl_num_heads % kv != 0 {
                return Err(crate::Error::Config("icl_num_heads must divide by icl_num_kv_heads_test".into()));
            }
            let eff = self.icl_num_kv_heads.unwrap_or(self.icl_num_heads);
            if kv > eff {
                return Err(crate::Error::Config("icl_num_kv_heads_test must be <= effective kv heads".into()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Multiclass,
    Regression,
}

impl TaskType {
    pub fn from_str(s: &str) -> crate::Result<Self> {
        match s {
            "multiclass" | "Multiclass" | "classification" => Ok(Self::Multiclass),
            "regression" | "Regression" => Ok(Self::Regression),
            other => Err(crate::Error::Config(format!("unsupported task_type: {other}"))),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Multiclass => "multiclass",
            Self::Regression => "regression",
        }
    }
}

/// Tiny deterministic config used by the parity fixtures so `cargo test` is fast.
pub fn tiny_config() -> TabPFNV3p5Config {
    TabPFNV3p5Config {
        embed_dim: 32,
        dist_embed_num_heads: 4,
        dist_embed_num_inducing_points: 16,
        feat_agg_num_heads: 4,
        feat_agg_num_cls_tokens: 4,
        nlayers: 2,
        icl_num_heads: 4,
        icl_num_kv_heads: Some(4),
        icl_num_kv_heads_test: Some(1),
        decoder_num_heads: 2,
        decoder_head_dim: 16,
        cell_ecdf_num_buckets: 8,
        max_num_classes: 10,
        num_buckets: 50,
        ..Default::default()
    }
}
