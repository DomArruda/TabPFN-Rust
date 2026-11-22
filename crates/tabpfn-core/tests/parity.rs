//! Parity tests: Rust outputs vs PyTorch golden fixtures.
//!
//! Fixtures are produced by `scripts/export_weights.py` (tiny deterministic
//! config + synthetic data with NaNs, +-Inf, ties, out-of-range values).
//! If the fixture file is missing the test is skipped so CI without the
//! Python env still passes; run the export script to enable full parity.

use candle_core::{Device, Tensor};
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};

fn fixture_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fixtures/golden.safetensors")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn read_f32_map(path: &std::path::Path) -> std::collections::HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).expect("read fixture");
    let view = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    view.tensors()
        .iter()
        .map(|(name, v)| {
            let bytes = v.data().to_vec();
            let f: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (name.clone(), f)
        })
        .collect()
}

// Re-exported so the fixture loader + parity assertions live in one place.
pub fn _reexport() { let _ = read_f32_map; }

#[test]
fn parity_multiclass_and_regression() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!("SKIP: golden fixture missing at {:?}; run scripts/export_weights.py", path);
        return;
    }
    let bytes = std::fs::read(&path).expect("read fixture");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let map = read_f32_map(&path);
    let _ = tensors;

    let dev = Device::Cpu;
    // config JSON is exported as a uint8 tensor by export_weights.py;
    // recover it by re-reading the raw bytes via the safe-tensors view.
    let config_json = {
        let all = tensors.tensors();
        let (_, v) = all.iter().find(|(n, _)| n == "__config_json__").expect("config tensor");
        String::from_utf8(v.data().to_vec()).expect("config utf8")
    };
    let config: TabPFNV3p5Config = serde_json::from_str(&config_json).expect("config json");

    // Build a tiny in-memory weight file from all __w__ tensors? Instead the
    // export script also writes model.safetensors in the same format.
    let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fixtures/model.safetensors");
    assert!(model_path.exists(), "model.safetensors fixture missing");
    let model_bytes = std::fs::read(&model_path).expect("read model");
    let model = TabPFNV3p5::from_safetensors(&model_bytes, config, &dev).expect("load model");

    for task in ["multiclass", "regression"] {
        let tt = TaskType::from_str(task).unwrap();
        let x_train: Vec<f32> = map[&format!("{task}.x_train")].clone();
        let y_train: Vec<f32> = map[&format!("{task}.y_train")].clone();
        let x_test: Vec<f32> = map[&format!("{task}.x_test")].clone();
        let n_train = map[&format!("{task}.n_train")][0] as usize;
        let n_test = map[&format!("{task}.n_test")][0] as usize;
        let n_cols = map[&format!("{task}.n_cols")][0] as usize;

        // single-pass forward
        let mut x_all = x_train.clone();
        x_all.extend(x_test.iter());
        let x_all_t = Tensor::from_vec(x_all, (1, n_train + n_test, n_cols), &dev).expect("tensor");
        let out_full = model.forward(&x_all_t, &y_train, tt, None, false, None).expect("forward");
        let got_full = out_full.to_vec3::<f32>().expect("vec").into_iter().flat_map(|m| m.into_iter().flat_map(|v| v.into_iter())).collect::<Vec<_>>();
        let exp_full = map[&format!("{task}.logits_full")].clone();
        let d = max_abs_diff(&got_full, &exp_full);
        assert!(d < 1e-4, "{task} full forward max_abs_diff={d}");

        // cached path
        let x_train_t = Tensor::from_vec(x_train.clone(), (1, n_train, n_cols), &dev).expect("tensor");
        let cache = model.fit(&x_train_t, &y_train, tt).expect("fit");
        let x_test_t = Tensor::from_vec(x_test.clone(), (1, n_test, n_cols), &dev).expect("tensor");
        let got_cached = model.predict_with_y(&cache, &x_test_t, &y_train, tt).expect("predict");
        let exp_cached = map[&format!("{task}.logits_cached")].clone();
        let d = max_abs_diff(&got_cached, &exp_cached);
        assert!(d < 1e-4, "{task} cached predict max_abs_diff={d}");
        let d2 = max_abs_diff(&got_cached, &got_full);
        assert!(d2 < 1e-4, "cached vs full max_abs_diff={d2}");
    }
}
