//! Times `TabPFNV3p5::forward` (full single-pass) on the fixture model and
//! writes raw timings + logits to JSON for the Python/WASM/Rust report.
//! Usage: `cargo run --release -p tabpfn-core --example bench_rust -- --reps 10 --out out.json`

use candle_core::{Device, Tensor};
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut reps = 10usize;
    let mut out_path = "scripts/bench_rust.json".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--out" => out_path = args.next().unwrap(),
            _ => {}
        }
    }

    let dev = Device::Cpu;
    let golden_path = "scripts/fixtures/golden.safetensors";
    let model_path = "scripts/fixtures/model.safetensors";
    let golden_bytes = std::fs::read(golden_path)?;
    let view = safetensors::SafeTensors::deserialize(&golden_bytes)?;
    let mut map: std::collections::HashMap<String, Vec<f32>> = Default::default();
    for (name, v) in view.tensors() {
        let b = v.data().to_vec();
        map.insert(name.clone(), b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    let config_json = {
        let all = view.tensors();
        let (_, v) = all.iter().find(|(n, _)| n == "__config_json__").unwrap();
        String::from_utf8(v.data().to_vec())?
    };
    let config: TabPFNV3p5Config = serde_json::from_str(&config_json)?;
    let model_bytes = std::fs::read(model_path)?;
    let model = TabPFNV3p5::from_safetensors(&model_bytes, config, &dev)?;

    let mut report = serde_json::json!({ "runtime": "rust_native", "device": "cpu", "reps": reps, "tasks": {} });
    for task in ["multiclass", "regression"] {
        let tt = TaskType::from_str(task)?;
        let n_train = map[&format!("{task}.n_train")][0] as usize;
        let n_test = map[&format!("{task}.n_test")][0] as usize;
        let n_cols = map[&format!("{task}.n_cols")][0] as usize;
        let mut x_all = map[&format!("{task}.x_train")].clone();
        x_all.extend(map[&format!("{task}.x_test")].iter());
        let y_train = map[&format!("{task}.y_train")].clone();
        let x_all_t = Tensor::from_vec(x_all, (1, n_train + n_test, n_cols), &dev)?;
        // warmup
        let _ = model.forward(&x_all_t, &y_train, tt, None, false, None)?;
        let mut timings_ms = Vec::new();
        let mut last_out: Vec<f32> = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = model.forward(&x_all_t, &y_train, tt, None, false, None)?;
            timings_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = out.to_vec3::<f32>()?.into_iter().flat_map(|m| m.into_iter().flat_map(|v| v.into_iter())).collect();
        }
        let exp = map[&format!("{task}.logits_full")].clone();
        let max_abs = last_out.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        report["tasks"][task] = serde_json::json!({
            "timings_ms": timings_ms,
            "max_abs_diff_vs_golden": max_abs,
            "logits": last_out,
        });
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {out_path}");
    Ok(())
}
