//! Native Rust harness. Same CLI contract as the Python/JS harnesses.
//!
//!   harness_datasets --dataset clf_iris_s --reps 10 --mode full|cached|baseline --out out.json

use candle_core::{Device, Tensor};
use serde_json::json;
use std::time::Instant;
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
    let mut args = std::env::args().skip(1);
    let mut datasets = "scripts/datasets.json".to_string();
    let mut dataset = String::new();
    let mut reps = 10usize;
    let mut mode = "full".to_string();
    let mut out_path = String::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--datasets" => datasets = args.next().unwrap(),
            "--dataset" => dataset = args.next().unwrap(),
            "--reps" => reps = args.next().unwrap().parse()?,
            "--mode" => mode = args.next().unwrap(),
            "--out" => out_path = args.next().unwrap(),
            _ => {}
        }
    }

    let dev = Device::Cpu;

    // model
    let golden = std::fs::read("scripts/fixtures/golden.safetensors")?;
    let gview = safetensors::SafeTensors::deserialize(&golden)?;
    let config_json = {
        let all = gview.tensors();
        let (_, v) = all.iter().find(|(n, _)| n == "__config_json__").unwrap();
        String::from_utf8(v.data().to_vec())?
    };
    let config: TabPFNV3p5Config = serde_json::from_str(&config_json)?;
    let model_bytes = std::fs::read("scripts/fixtures/model.safetensors")?;
    let model = TabPFNV3p5::from_safetensors(&model_bytes, config, &dev)?;

    // dataset
    let spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&datasets)?)?;
    let d = spec["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["name"] == dataset)
        .expect("dataset not found");
    let task = TaskType::from_str(d["task"].as_str().unwrap())?;
    let n_train = d["n_train"].as_u64().unwrap() as usize;
    let n_test = d["n_test"].as_u64().unwrap() as usize;
    let n_cols = d["n_cols"].as_u64().unwrap() as usize;
    let x_train = hex_f32(d["x_train"].as_str().unwrap());
    let y_train = hex_f32(d["y_train"].as_str().unwrap());
    let x_test = hex_f32(d["x_test"].as_str().unwrap());

    let mut report = json!({
        "runtime": "rust_native",
        "dataset": dataset,
        "task": d["task"],
        "mode": mode,
        "reps": if mode == "baseline" { 0 } else { reps },
        "n_train": n_train,
        "n_test": n_test,
        "n_cols": n_cols,
    });

    if mode == "baseline" {
        report["timings_ms"] = json!([]);
        report["logits"] = json!([]);
        std::fs::write(&out_path, serde_json::to_string(&report)?)?;
        return Ok(());
    }

    let mut timings = Vec::with_capacity(reps);
    let mut logits: Vec<f32> = Vec::new();

    if mode == "full" {
        let mut x_all = x_train.clone();
        x_all.extend_from_slice(&x_test);
        let xt = Tensor::from_vec(x_all, (1, n_train + n_test, n_cols), &dev)?;
        let _ = model.forward(&xt, &y_train, task, None, false, None)?; // warmup
        for _ in 0..reps {
            let t0 = Instant::now();
            let out = model.forward(&xt, &y_train, task, None, false, None)?;
            timings.push(t0.elapsed().as_secs_f64() * 1000.0);
            logits = out
                .to_vec3::<f32>()?
                .into_iter()
                .flat_map(|m| m.into_iter().flat_map(|r| r.into_iter()))
                .collect();
        }
    } else {
        let xt_train = Tensor::from_vec(x_train, (1, n_train, n_cols), &dev)?;
        let cache = model.fit(&xt_train, &y_train, task)?;
        let xt_test = Tensor::from_vec(x_test, (1, n_test, n_cols), &dev)?;
        let _ = model.predict_with_y(&cache, &xt_test, &y_train, task)?; // warmup
        for _ in 0..reps {
            let t0 = Instant::now();
            logits = model.predict_with_y(&cache, &xt_test, &y_train, task)?;
            timings.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    report["timings_ms"] = json!(timings);
    report["logits"] = json!(logits);
    std::fs::write(&out_path, serde_json::to_string(&report)?)?;
    Ok(())
}
