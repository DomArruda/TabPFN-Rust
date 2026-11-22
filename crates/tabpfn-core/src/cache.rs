//! Explicit KV cache mirroring `TabPFNV3p5Cache` in `tabpfn_v3_5.py`.
//!
//! Stores everything needed to skip stages 0-2 for train rows and reuse
//! cached K/V in the ICL transformer on the test-only path.

use candle_core::Tensor;

/// Per-layer KV entry: key/value already k_norm'd, sliced to test heads.
#[derive(Clone)]
pub struct KvEntry {
    pub key: Tensor,   // (B, N, Hkv_test, D)
    pub value: Tensor, // (B, N, Hkv_test, D)
}

/// Top-level cache container (mirrors the Python dataclass fields).
#[derive(Clone)]
pub struct TabPFNV3p5Cache {
    /// Per-layer ICL KV entries, keyed by layer index.
    pub kv: Vec<KvEntry>,
    /// Many-class decoder keys (B, N, H_dec, D_dec). None for regression.
    pub decoder_keys: Option<Tensor>,
    pub train_shape: (usize, usize), // (B, N)
    pub scaler_mean: Vec<f32>,
    pub scaler_std: Vec<f32>,
    pub ecdf_context: crate::preprocess::EcdfContext,
    /// Per-dist-block inducing hidden states, each (B*C, n_ind, E).
    pub inducing_hidden: Vec<Tensor>,
    pub num_train: usize,
    pub n_cols: usize,
}

impl TabPFNV3p5Cache {
    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }
}
