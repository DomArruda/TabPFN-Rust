//! WASM bindings: stateful `TabPfnWasm` with flat `Float32Array`-friendly API.
//!
//! JS usage:
//! ```js
//! import init, { TabPfnWasm } from './pkg/tabpfn_wasm.js';
//! await init();
//! const m = new TabPfnWasm(weightsBytes, configJson);
//! m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
//! const out = m.predict(xTest, nTest, nCols, "multiclass");
//! ```

use candle_core::{Device, Tensor};
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TabPfnWasm {
    model: TabPFNV3p5,
    cache: Option<tabpfn_core::cache::TabPFNV3p5Cache>,
    y_train: Vec<f32>,
    task: Option<TaskType>,
}

fn js_err(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}

#[wasm_bindgen]
impl TabPfnWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(weights_bytes: &[u8], config_json: &str) -> Result<TabPfnWasm, JsValue> {
        console_error_panic_hook::set_once();
        let config: TabPFNV3p5Config =
            serde_json::from_str(config_json).map_err(|e| js_err(&e.to_string()))?;
        let device = Device::Cpu;
        let model = TabPFNV3p5::from_safetensors(weights_bytes, config, &device)
            .map_err(|e| js_err(&e.to_string()))?;
        Ok(Self { model, cache: None, y_train: Vec::new(), task: None })
    }

    pub fn fit(&mut self, x_train: &[f32], n_rows: usize, n_cols: usize, y_train: &[f32], task_type: &str) -> Result<(), JsValue> {
        if x_train.len() != n_rows * n_cols {
            return Err(js_err("x_train size mismatch"));
        }
        if y_train.len() != n_rows {
            return Err(js_err("y_train size mismatch"));
        }
        let task = TaskType::from_str(task_type).map_err(|e| js_err(&e.to_string()))?;
        let x = Tensor::from_vec(x_train.to_vec(), (1, n_rows, n_cols), &self.model.device)
            .map_err(|e| js_err(&e.to_string()))?;
        let cache = self.model.fit(&x, y_train, task).map_err(|e| js_err(&e.to_string()))?;
        self.cache = Some(cache);
        self.y_train = y_train.to_vec();
        self.task = Some(task);
        Ok(())
    }

    pub fn predict(&self, x_test: &[f32], n_test: usize, n_cols: usize, task_type: &str) -> Result<Vec<f32>, JsValue> {
        if x_test.len() != n_test * n_cols {
            return Err(js_err("x_test size mismatch"));
        }
        let task = TaskType::from_str(task_type).map_err(|e| js_err(&e.to_string()))?;
        let x = Tensor::from_vec(x_test.to_vec(), (1, n_test, n_cols), &self.model.device)
            .map_err(|e| js_err(&e.to_string()))?;
        let cache = self.cache.as_ref().ok_or_else(|| js_err("call fit() first"))?;
        let out = self.model.predict_with_y(cache, &x, &self.y_train, task).map_err(|e| js_err(&e.to_string()))?;
        Ok(out)
    }

    /// Single-pass forward over train+test rows — same computation Python and
    /// native Rust time as `forward`. Returns flat `(n_test, 1, T)` row-major.
    pub fn forward(
        &self,
        x_all: &[f32],
        n_train: usize,
        n_test: usize,
        n_cols: usize,
        y_train: &[f32],
        task_type: &str,
    ) -> Result<Vec<f32>, JsValue> {
        if x_all.len() != (n_train + n_test) * n_cols {
            return Err(js_err("x_all size mismatch"));
        }
        if y_train.len() != n_train {
            return Err(js_err("y_train size mismatch"));
        }
        let task = TaskType::from_str(task_type).map_err(|e| js_err(&e.to_string()))?;
        let x = Tensor::from_vec(x_all.to_vec(), (1, n_train + n_test, n_cols), &self.model.device)
            .map_err(|e| js_err(&e.to_string()))?;
        let out = self
            .model
            .forward(&x, y_train, task, None, false, None)
            .map_err(|e| js_err(&e.to_string()))?;
        let flat = out
            .to_vec3::<f32>()
            .map_err(|e| js_err(&e.to_string()))?
            .into_iter()
            .flat_map(|m| m.into_iter().flat_map(|r| r.into_iter()))
            .collect();
        Ok(flat)
    }

    /// Post-output-norm row embeddings, flat row-major
    /// `(n_train + n_test, icl_emsize)`. Contextual representation (depends on
    /// the training set), not a fixed feature extractor.
    pub fn encode(
        &self,
        x_all: &[f32],
        n_train: usize,
        n_test: usize,
        n_cols: usize,
        y_train: &[f32],
        task_type: &str,
    ) -> Result<Vec<f32>, JsValue> {
        if x_all.len() != (n_train + n_test) * n_cols {
            return Err(js_err("x_all size mismatch"));
        }
        let task = TaskType::from_str(task_type).map_err(|e| js_err(&e.to_string()))?;
        let x = Tensor::from_vec(x_all.to_vec(), (1, n_train + n_test, n_cols), &self.model.device)
            .map_err(|e| js_err(&e.to_string()))?;
        let emb = self
            .model
            .encode(&x, y_train, task)
            .map_err(|e| js_err(&e.to_string()))?;
        let r = emb.dim(1).map_err(|e| js_err(&e.to_string()))?;
        let e = emb.dim(2).map_err(|e| js_err(&e.to_string()))?;
        let flat = emb
            .reshape((r, e))
            .map_err(|e| js_err(&e.to_string()))?
            .flatten_all()
            .map_err(|e| js_err(&e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| js_err(&e.to_string()))?;
        Ok(flat)
    }
}
