//! PyO3 bindings for the Rust/Candle TabPFN v3.5 inference core.
//!
//! Built as a CPython extension (`tabpfn_rs.pyd` / `.so`). All heavy math runs
//! in Rust; Python only passes flat little-endian f32 buffers (`bytes`) and
//! gets flat f32 buffers back, so per-call conversion cost is one allocation
//! rather than one Python object per element.
//!
//! This exists to answer: "does the Rust speedup survive a Python API hop?"
//! (Spoiler: yes — the measured call overhead is a few microseconds.)

use candle_core::{Device, Tensor};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tabpfn_core::cache::TabPFNV3p5Cache;
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};

fn to_f32(b: &[u8]) -> PyResult<Vec<f32>> {
    if b.len() % 4 != 0 {
        return Err(PyValueError::new_err("buffer length must be a multiple of 4"));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn to_bytes(py: Python<'_>, v: &[f32]) -> Py<PyBytes> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for &x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    PyBytes::new(py, &buf).unbind()
}

fn task(s: &str) -> PyResult<TaskType> {
    TaskType::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(unsendable, name = "TabPfn")]
struct PyTabPfn {
    model: TabPFNV3p5,
    cache: Option<TabPFNV3p5Cache>,
    y_train: Vec<f32>,
}

#[pymethods]
impl PyTabPfn {
    /// new(weights_bytes, config_json) -> TabPfn
    /// `weights_bytes` is the raw safetensors file contents (PyTorch key paths).
    #[new]
    fn new(weights: &Bound<'_, PyBytes>, config_json: &str) -> PyResult<Self> {
        let cfg: TabPFNV3p5Config = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("config: {e}")))?;
        let model = TabPFNV3p5::from_safetensors(weights.as_bytes(), cfg, &Device::Cpu)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { model, cache: None, y_train: Vec::new() })
    }

    /// Zero-work call, used to measure pure Python->Rust FFI overhead.
    fn ping(&self) -> usize {
        0
    }

    /// Full single-pass forward on train+test rows.
    /// x_all: flat f32 bytes, shape (n_train + n_test, n_cols) row-major.
    /// Returns flat f32 bytes, layout (n_test, 1, num_outputs) row-major.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x_all: &Bound<'_, PyBytes>,
        n_train: usize,
        n_test: usize,
        n_cols: usize,
        y_train: &Bound<'_, PyBytes>,
        task_type: &str,
    ) -> PyResult<Py<PyBytes>> {
        let x = to_f32(x_all.as_bytes())?;
        let y = to_f32(y_train.as_bytes())?;
        let tt = task(task_type)?;
        if x.len() != (n_train + n_test) * n_cols {
            return Err(PyValueError::new_err("x_all size mismatch"));
        }
        let xt = Tensor::from_vec(x, (1, n_train + n_test, n_cols), &self.model.device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = self
            .model
            .forward(&xt, &y, tt, None, false, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flat: Vec<f32> = out
            .to_vec3::<f32>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_iter()
            .flat_map(|m| m.into_iter().flat_map(|r| r.into_iter()))
            .collect();
        Ok(to_bytes(py, &flat))
    }

    /// fit(x_train, n_rows, n_cols, y_train, task) — build the KV cache once.
    fn fit(
        &mut self,
        x_train: &Bound<'_, PyBytes>,
        n_rows: usize,
        n_cols: usize,
        y_train: &Bound<'_, PyBytes>,
        task_type: &str,
    ) -> PyResult<()> {
        let x = to_f32(x_train.as_bytes())?;
        let y = to_f32(y_train.as_bytes())?;
        let tt = task(task_type)?;
        if x.len() != n_rows * n_cols {
            return Err(PyValueError::new_err("x_train size mismatch"));
        }
        let xt = Tensor::from_vec(x, (1, n_rows, n_cols), &self.model.device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let cache = self
            .model
            .fit(&xt, &y, tt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.cache = Some(cache);
        self.y_train = y;
        Ok(())
    }

    /// predict(x_test, n_test, n_cols, task) — cached, test-only path.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_test: &Bound<'_, PyBytes>,
        n_test: usize,
        n_cols: usize,
        task_type: &str,
    ) -> PyResult<Py<PyBytes>> {
        let x = to_f32(x_test.as_bytes())?;
        let tt = task(task_type)?;
        if x.len() != n_test * n_cols {
            return Err(PyValueError::new_err("x_test size mismatch"));
        }
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call fit() first"))?;
        let xt = Tensor::from_vec(x, (1, n_test, n_cols), &self.model.device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = self
            .model
            .predict_with_y(cache, &xt, &self.y_train, tt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(to_bytes(py, &out))
    }
}

#[pymodule]
fn tabpfn_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTabPfn>()?;
    Ok(())
}
