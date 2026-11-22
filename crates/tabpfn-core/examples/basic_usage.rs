//! End-to-end Rust usage: load weights, fit once, predict, and inspect the
//! internal row representation.
//!
//!   cargo run --release -p tabpfn-core --example basic_usage
//!
//! Reads the committed fixtures (`scripts/fixtures/`) and one dataset from
//! `scripts/datasets.json`. Run from the workspace root.

use candle_core::{Device, Tensor};
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};

fn hex_f32(s: &str) -> Vec<f32> {
    let bytes: Vec<u8> = (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let dataset_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "clf_iris_s".to_string());

    // ---- 1. Load the model ------------------------------------------------
    // `model.safetensors` uses the exact PyTorch `state_dict` key paths.
    let config_json = {
        let bytes = std::fs::read("scripts/fixtures/golden.safetensors")?;
        let view = safetensors::SafeTensors::deserialize(&bytes)?;
        let tensors = view.tensors();
        let (_, t) = tensors
            .iter()
            .find(|(n, _)| n == "__config_json__")
            .ok_or("fixture has no __config_json__")?;
        String::from_utf8(t.data().to_vec())?
    };
    let config: TabPFNV3p5Config = serde_json::from_str(&config_json)?;
    let weights = std::fs::read("scripts/fixtures/model.safetensors")?;
    let model = TabPFNV3p5::from_safetensors(&weights, config, &device)?;

    // ---- 2. Load a dataset ------------------------------------------------
    let spec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("scripts/datasets.json")?)?;
    let d = spec["datasets"]
        .as_array()
        .ok_or("bad datasets.json")?
        .iter()
        .find(|x| x["name"].as_str() == Some(dataset_name.as_str()))
        .ok_or_else(|| format!("dataset {dataset_name} not found"))?;
    let task = TaskType::from_str(d["task"].as_str().ok_or("missing task")?)?;
    let n_train = d["n_train"].as_u64().ok_or("n_train")? as usize;
    let n_test = d["n_test"].as_u64().ok_or("n_test")? as usize;
    let n_cols = d["n_cols"].as_u64().ok_or("n_cols")? as usize;
    let x_train = hex_f32(d["x_train"].as_str().ok_or("x_train")?);
    let x_test = hex_f32(d["x_test"].as_str().ok_or("x_test")?);
    let y_train = hex_f32(d["y_train"].as_str().ok_or("y_train")?);

    // Tensors are (batch, rows, columns); B = 1.
    let xt_train = Tensor::from_vec(x_train.clone(), (1, n_train, n_cols), &device)?;
    let xt_test = Tensor::from_vec(x_test.clone(), (1, n_test, n_cols), &device)?;

    // ---- 3. fit() once -> KV cache, then predict() many ----------------
    let cache = model.fit(&xt_train, &y_train, task)?;
    let logits = model.predict_with_y(&cache, &xt_test, &y_train, task)?;
    println!(
        "{dataset_name}: {n_train} train / {n_test} test rows, {n_cols} cols, task={task:?}"
    );
    println!(
        "cached predict -> {} values; first 6: {:?}",
        logits.len(),
        &logits[..6.min(logits.len())]
    );

    // ---- 4. Inspect the internal representation ("encoder") --------------
    // Contextual: rows are embedded *given* the training set.
    let mut x_all = x_train;
    x_all.extend_from_slice(&x_test);
    let xt_all = Tensor::from_vec(x_all, (1, n_train + n_test, n_cols), &device)?;
    let emb = model.encode(&xt_all, &y_train, task)?;
    println!(
        "encode -> embeddings {:?}  (rows x icl_emsize)",
        emb.dims()
    );

    // The same test rows, embedded against the fitted cache:
    let emb_cached = model.encode_cached(&cache, &xt_test, &y_train, task)?;
    println!("encode_cached -> embeddings {:?}", emb_cached.dims());

    Ok(())
}
