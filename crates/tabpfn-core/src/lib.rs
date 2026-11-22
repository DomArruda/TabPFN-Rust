//! TabPFN v3.5 inference in pure Rust + Candle.
//!
//! Parity reference: `tabpfn/architectures/tabpfn_v3_5.py` (PriorLabs TabPFN).
//! All numerics are f32. `preprocess.rs` is plain `Vec<f32>` logic so sorting /
//! searchsorted match PyTorch bit-for-bit enough for the ECDF midrank trick.

pub mod attention;
pub mod cache;
pub mod config;
pub mod layers;
pub mod log;
pub mod model;
pub mod preprocess;
pub mod stages;

pub use config::{TabPFNV3p5Config, TaskType};
pub use model::TabPFNV3p5;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("config: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(String),
    #[error("shape: {0}")]
    Shape(String),
}
