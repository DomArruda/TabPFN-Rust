# TabPFN-Rust

A port of TabPFN v3.5's inference architecture to Rust (Candle), with Python
and WebAssembly bindings.

The neural architecture is a faithful inference-only port of
`tabpfn/architectures/tabpfn_v3_5.py`: same stages, same tensor shapes, same
config defaults, and the exact PyTorch `state_dict` key paths (no remapping).
Cross-runtime outputs agree to ~1e-7 against PyTorch on the same weights.

## Layout

```
crates/tabpfn-core/   pure Rust + Candle model, preprocessing, KV cache
crates/tabpfn-wasm/   wasm-bindgen bindings (JS/TS API)
crates/tabpfn-py/     PyO3 bindings (CPython extension)
scripts/              dataset generation, harnesses, stats, fixtures
main.ts               feature demo (Bun)
smoke.test.ts         Bun tests
```

## Quick start

```bash
# Rust core: parity tests against the PyTorch golden fixtures
cargo test -p tabpfn-core

# WASM (SIMD128) + JS glue
npm run build:wasm          # cargo build --target wasm32-unknown-unknown --release + wasm-bindgen

# Bun tests + demo
bun test
bun run main.ts

# Python extension (builds scripts/tabpfn_rs.pyd)
cargo build -p tabpfn-py --release
copy target/release/tabpfn_rs.dll scripts/tabpfn_rs.pyd   # Windows
# cp target/release/libtabpfn_rs.so scripts/tabpfn_rs.so  # Linux/macOS
```

The `pkg/` directory and the `.pyd`/`.so` are build artefacts and are
git-ignored. The `scripts/fixtures/model.safetensors` test fixture (a tiny
2-layer model with deterministic weights) is committed so the Rust and WASM
tests run without a Python environment.

## KV caching: what `fit()` stores, and how to use it

TabPFN's ICL stage lets test rows attend **only to training rows** — never to
each other. Test predictions are therefore conditionally independent given the
training set, so everything that depends solely on the training data can be
computed once and reused across `predict()` calls. That is exactly what
`fit()` does.

| cached field | shape | why it is reusable |
|---|---|---|
| `inducing_hidden` (per dist. block) | `(B·C, n_ind, E)` | inducing points attend to train rows only |
| `kv[layer]` (K and V) | `(B, N, n_kv_heads_test, head_dim)` per layer | ICL keys/values are train-only, already `k_norm`'d and sliced to the test-only heads |
| `decoder_keys` (multiclass) | `(B, N, H_dec, D_dec)` | many-class decoder keys come from train rows only |
| `scaler_mean` / `scaler_std` | `(C,)` | fitted on train rows |
| `ecdf_context` | `(3, B, C, K)` | ECDF bucket edges built from train rows |

`predict(X_test)` then runs only the test-row path: cell embedding, the second
distribution-attention pass (test → cached inducing hidden), column
aggregation, ICL queries against the cached K/V, and the decoder queries.
Cost is **O(M)** in test rows and independent of `N`.

### Same operation on each surface

**Rust**

```rust
let cache = model.fit(&x_train, &y_train, TaskType::Multiclass)?;         // once
let out   = model.predict_with_y(&cache, &x_test, &y_train, TaskType::Multiclass)?; // many
```

**Python (PyO3 binding)**

```python
from tabpfn_rs import TabPfn
m = TabPfn(weights, config_json)
m.fit(x_train.tobytes(), n_train, n_cols, y_train.tobytes(), "multiclass")
out = m.predict(x_test.tobytes(), n_test, n_cols, "multiclass")
```

**WASM / TypeScript**

```ts
const m = new TabPfnWasm(weights, configJson);
m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
const out = m.predict(xTest, nTest, nCols, "multiclass");
```

**PyTorch reference (upstream TabPFN)** — the same two-step pattern already
exists there, either high-level or via the explicit cache:

```python
clf = TabPFNClassifier(fit_mode="fit_with_cache")   # high level
clf.fit(X, y)
probs = clf.predict(X_test)                         # reuses the cache

# low level (this is what the benchmark harness does)
_, cache = model.forward(x_all, y, "multiclass", return_kv_cache=True)
logits = model.forward(x_test, y, "multiclass", kv_cache=cache, x_is_test_only=True)
```

### What it buys

Caching removes the training-row work from the hot path, so the win scales with
`N` and shrinks as the per-row cost grows with feature count. On iris-scale
data a cached `predict` is single-digit milliseconds; on the widest table it is
tens of milliseconds. The exact per-dataset numbers, and a
*Cached vs full (mean over datasets)* summary, are generated in the benchmark
section below by `python scripts/analyze.py --write-readme`.

### Cache size

For the released config (`nlayers=24`, `head_dim=64`, `n_kv_heads_test=1`,
`6×64` decoder, `128` inducing points, `E=128`, `K=8192`), per estimator:

```
ICL K/V       12,288 · N  bytes     (12 MB per 1,000 train rows)
decoder keys   1,536 · N  bytes     (multiclass only)
inducing     196,608 · C  bytes     (20 MB at C = 100)
ECDF context  98,304 · C  bytes     (10 MB at C = 100)
```

The benchmark's tiny config (`E=32`, 2 layers, 16 inducing points, `K=8`) is
correspondingly small (~0.5 MB for the widest dataset).

### Guard rails and caveats

* A cache is valid only for the exact training set it was built from. The Rust
  `predict_with_y` now rejects a mismatched column count, batch size, train-row
  count, layer count, or a regression cache used for multiclass — with an
  explicit message instead of silently returning garbage.
* Multiclass caches carry `decoder_keys`; regression caches do not (and are
  smaller).
* Cached `predict` still costs O(M) per test row; chunk very large test sets.

## Benchmark: 4 runtimes, identical data

We compare the same inference workload across:

| runtime | what it is |
|---|---|
| `python_pytorch` | original Python / PyTorch TabPFN v3.5, CPU only (`torch 2.9.1+cpu`) |
| `python_bindings` | the Rust core called through the PyO3 extension |
| `rust_native` | Rust / Candle, native release binary |
| `wasm_bun` | Rust / Candle compiled to WASM SIMD128, run under Bun |

Two code paths are timed on every dataset:

* **full** — one single-pass forward over `train + test` rows (no cache):
  stages 0–2 preprocessing, distribution embedder, column aggregator, ICL
  transformer, decoder.
* **cached** — `fit()` once (builds inducing-hidden states, per-layer ICL K/V,
  and decoder keys), then time `predict()` on test rows only.

A third `baseline` mode loads the model and data but runs no inference; its
peak RSS is subtracted from the other two to isolate inference memory.

Datasets are 8 real/synthetic sets at four scale levels, generated by
`scripts/gen_datasets.py` and shared byte-for-byte via hex-encoded f32
(`scripts/datasets.json`) so NaN/±Inf survive in all three languages:

| dataset | task | train | test | cols |
|---|---|---:|---:|---:|
| clf_iris_s | multiclass | 30 | 10 | 4 |
| clf_wine_m | multiclass | 100 | 25 | 13 |
| clf_breast_l | multiclass | 300 | 50 | 30 |
| clf_digits_w | multiclass | 200 | 30 | 64 |
| clf_iris_nan | multiclass | 40 | 12 | 4 (NaN/±Inf/out-of-range) |
| reg_diabetes_s | regression | 40 | 10 | 10 |
| reg_synth_m | regression | 120 | 25 | 20 |
| reg_synth_w | regression | 250 | 40 | 50 |

Peak RSS is sampled externally with `psutil` for every subprocess, so all four
runtimes are measured identically rather than self-reporting.

> **Accuracy caveat.** The committed fixture uses randomly initialised weights
> (a 2-layer model), so *predictive* accuracy is chance-level and identical
> across runtimes by construction. "Accuracy" below therefore means
> **cross-runtime numerical equivalence** (parity within `1e-4`) plus argmax
> agreement — the property that actually matters for a port. Real predictive
> accuracy requires the licensed upstream checkpoint.

## Replication

```bash
# 1. fixtures + datasets
pip install tabpfn safetensors       # PyTorch reference
python scripts/export_weights.py     # writes scripts/fixtures/{model,golden}.safetensors
python scripts/gen_datasets.py       # writes scripts/datasets.json

# 2. build every runtime
cargo build --release -p tabpfn-core --example harness_datasets
npm run build:wasm
cargo build -p tabpfn-py --release && copy target\release\tabpfn_rs.dll scripts\tabpfn_rs.pyd

# 3. run the matrix (4 runtimes x 8 datasets x 3 modes, 10 reps each)
python scripts/run_benchmarks.py --reps 10

# 4. statistics -> scripts/results/stats.json, and refresh this README section
python scripts/analyze.py --write-readme
```

Statistics: one-way ANOVA per dataset (speed and memory), a pooled two-way
ANOVA with replication (factor A = runtime, factor B = dataset, response =
`log10(ms)`), Tukey–Kramer post-hoc tests, and a one-sided equivalence test
for parity (mean absolute logit difference `< 1e-4`).

<!-- BENCH:START -->
Measured on 8 datasets x 10 reps; equivalence tolerance `0.0001`. Speed ANOVAs use `log10(ms)`; memory uses `log10(delta + 1)`.

### Headline

* **full**: Rust-native beats PyTorch on 3/8 datasets (clf_iris_nan, clf_iris_s, reg_diabetes_s); PyTorch wins on clf_breast_l, clf_digits_w, clf_wine_m, reg_synth_m, reg_synth_w. Interaction F=190, p=3.59e-155, partial η²=0.93 — the ranking is **dataset-dependent**, so no single runtime is universally fastest.
* **cached**: Rust-native beats PyTorch on 5/8 datasets (clf_iris_nan, clf_iris_s, clf_wine_m, reg_diabetes_s, reg_synth_m); PyTorch wins on clf_breast_l, clf_digits_w, reg_synth_w. Interaction F=478, p=4.07e-210, partial η²=0.97 — the ranking is **dataset-dependent**, so no single runtime is universally fastest.

The Rust advantage appears where per-operation overhead dominates (few features, short rows) and disappears on wide tables, where the ported per-head SDPA loop and Candle's small-matrix kernels lose to PyTorch's BLAS-backed batched ops. The cached path pushes the crossover out because train-row work is paid once.

### Speed — Full forward (train + test, no cache)

| dataset | Python (PyTorch) (ms) | Python (PyO3) (ms) | Rust native (ms) | WASM (Bun) (ms) | fastest |
|---|---:|---:|---:|---:|---:|
| clf_breast_l | 56.97 ± 1.83 | 184.30 ± 2.92 | 187.31 ± 6.87 | 254.29 ± 1.98 | Python (PyTorch) |
| clf_digits_w | 64.66 ± 1.45 | 330.26 ± 43.63 | 316.55 ± 45.67 | 529.66 ± 199.15 | Python (PyTorch) |
| clf_iris_nan | 19.32 ± 1.70 | 8.53 ± 0.41 | 10.03 ± 0.32 | 9.60 ± 0.40 | Python (PyO3) |
| clf_iris_s | 14.55 ± 2.13 | 7.07 ± 0.39 | 8.36 ± 0.61 | 8.82 ± 1.29 | Python (PyO3) |
| clf_wine_m | 26.88 ± 1.16 | 37.63 ± 2.62 | 41.52 ± 2.78 | 45.94 ± 2.94 | Python (PyTorch) |
| reg_diabetes_s | 19.88 ± 0.68 | 14.39 ± 0.47 | 15.79 ± 1.13 | 15.88 ± 0.69 | Python (PyO3) |
| reg_synth_m | 27.59 ± 1.70 | 55.10 ± 4.73 | 54.35 ± 3.92 | 71.69 ± 0.77 | Python (PyTorch) |
| reg_synth_w | 69.02 ± 2.38 | 289.28 ± 44.79 | 292.69 ± 25.18 | 501.32 ± 199.19 | Python (PyTorch) |

Two-way ANOVA (runtime x dataset, `log10(ms)`):

| factor | df | F | p | partial η² |
|---|---:|---:|---:|---:|
| runtime | 3 | 519.4 | 8.05e-116 | 0.844 |
| dataset | 7 | 5434.2 | <1e-300 | 0.992 |
| interaction | 21 | 190.0 | 3.59e-155 | 0.933 |

Tukey–Kramer post-hoc on runtime (pooled):

| pair | Δ log10(ms) | q | p | significant |
|---|---:|---:|---:|---|
| Python (PyTorch) vs Python (PyO3) | -0.184 | 33.31 | <1e-300 | yes |
| Python (PyTorch) vs Rust native | -0.211 | 38.18 | <1e-300 | yes |
| Python (PyTorch) vs WASM (Bun) | -0.300 | 54.26 | <1e-300 | yes |
| Python (PyO3) vs Rust native | -0.027 | 4.87 | 0.00362 | yes |
| Python (PyO3) vs WASM (Bun) | -0.116 | 20.95 | <1e-300 | yes |
| Rust native vs WASM (Bun) | -0.089 | 16.07 | <1e-300 | yes |

### Speed — Cached (fit once, predict test only)

| dataset | Python (PyTorch) (ms) | Python (PyO3) (ms) | Rust native (ms) | WASM (Bun) (ms) | fastest |
|---|---:|---:|---:|---:|---:|
| clf_breast_l | 18.49 ± 1.54 | 24.10 ± 0.79 | 25.78 ± 1.02 | 31.72 ± 0.37 | Python (PyTorch) |
| clf_digits_w | 20.85 ± 0.81 | 33.97 ± 1.04 | 35.13 ± 1.67 | 43.29 ± 0.76 | Python (PyTorch) |
| clf_iris_nan | 8.85 ± 1.21 | 1.46 ± 0.08 | 1.48 ± 0.04 | 2.64 ± 0.41 | Python (PyO3) |
| clf_iris_s | 7.60 ± 0.39 | 1.37 ± 0.08 | 1.35 ± 0.06 | 2.50 ± 0.76 | Rust native |
| clf_wine_m | 10.68 ± 0.90 | 7.38 ± 0.20 | 8.96 ± 0.37 | 8.36 ± 0.25 | Python (PyO3) |
| reg_diabetes_s | 8.55 ± 0.56 | 1.86 ± 0.05 | 1.96 ± 0.04 | 3.73 ± 0.47 | Python (PyO3) |
| reg_synth_m | 12.47 ± 0.62 | 10.55 ± 0.24 | 10.51 ± 0.36 | 11.05 ± 0.25 | Rust native |
| reg_synth_w | 16.40 ± 0.60 | 34.68 ± 0.80 | 38.00 ± 1.34 | 45.26 ± 3.90 | Python (PyTorch) |

Two-way ANOVA (runtime x dataset, `log10(ms)`):

| factor | df | F | p | partial η² |
|---|---:|---:|---:|---:|
| runtime | 3 | 876.6 | 1.98e-144 | 0.901 |
| dataset | 7 | 9099.1 | <1e-300 | 0.995 |
| interaction | 21 | 478.1 | 4.07e-210 | 0.972 |

Tukey–Kramer post-hoc on runtime (pooled):

| pair | Δ log10(ms) | q | p | significant |
|---|---:|---:|---:|---|
| Python (PyTorch) vs Python (PyO3) | +0.221 | 62.26 | <1e-300 | yes |
| Python (PyTorch) vs Rust native | +0.197 | 55.51 | <1e-300 | yes |
| Python (PyTorch) vs WASM (Bun) | +0.069 | 19.41 | <1e-300 | yes |
| Python (PyO3) vs Rust native | -0.024 | 6.75 | 1.72e-05 | yes |
| Python (PyO3) vs WASM (Bun) | -0.152 | 42.85 | <1e-300 | yes |
| Rust native vs WASM (Bun) | -0.128 | 36.10 | <1e-300 | yes |

### Cached vs full (mean over datasets)

| runtime | full (ms) | cached (ms) | speedup |
|---|---:|---:|---:|
| Python (PyTorch) | 37.4 | 13.0 | 2.9× |
| Python (PyO3) | 115.8 | 14.4 | 8.0× |
| Rust native | 115.8 | 15.4 | 7.5× |
| WASM (Bun) | 179.7 | 18.6 | 9.7× |

### Numerical parity vs PyTorch (max |logit diff|)

| dataset | mode | Python (PyO3) | Rust native | WASM (Bun) |
|---|---|---:|---:|---:|
| clf_breast_l | full | 1.43e-06 | 1.43e-06 | 1.07e-06 |
| clf_breast_l | cached | 1.43e-06 | 1.43e-06 | 1.07e-06 |
| clf_digits_w | full | 4.77e-07 | 4.77e-07 | 2.38e-07 |
| clf_digits_w | cached | 4.77e-07 | 4.77e-07 | 2.38e-07 |
| clf_iris_nan | full | 0.00e+00 | 0.00e+00 | 1.19e-07 |
| clf_iris_nan | cached | 0.00e+00 | 0.00e+00 | 1.19e-07 |
| clf_iris_s | full | 2.38e-07 | 2.38e-07 | 2.38e-07 |
| clf_iris_s | cached | 2.38e-07 | 2.38e-07 | 2.38e-07 |
| clf_wine_m | full | 4.17e-07 | 4.17e-07 | 4.17e-07 |
| clf_wine_m | cached | 4.17e-07 | 4.17e-07 | 4.17e-07 |
| reg_diabetes_s | full | 5.96e-07 | 5.96e-07 | 3.58e-07 |
| reg_diabetes_s | cached | 5.96e-07 | 5.96e-07 | 3.58e-07 |
| reg_synth_m | full | 5.96e-07 | 5.96e-07 | 3.58e-07 |
| reg_synth_m | cached | 5.96e-07 | 5.96e-07 | 3.58e-07 |
| reg_synth_w | full | 4.77e-07 | 4.77e-07 | 1.79e-07 |
| reg_synth_w | cached | 4.77e-07 | 4.77e-07 | 1.79e-07 |

All 3 non-baseline runtimes pass the one-sided equivalence test (mean|diff| < 0.0001, p < 0.05) on every dataset/mode; worst observed max |diff| = 1.43e-06.

### Accuracy / agreement

Randomly initialised weights, so predictive accuracy is chance level — this table verifies that all runtimes make the *same* predictions.

| dataset | task | Python (PyTorch) acc | Python (PyO3) acc | Rust native acc | WASM (Bun) acc | argmax agreement |
|---|---|---:|---:|---:|---:|---:|
| clf_breast_l | multiclass | 0.620 | 0.620 | 0.620 | 0.620 | 100% |
| clf_digits_w | multiclass | 0.100 | 0.100 | 0.100 | 0.100 | 100% |
| clf_iris_nan | multiclass | 0.333 | 0.333 | 0.333 | 0.333 | 100% |
| clf_iris_s | multiclass | 0.300 | 0.300 | 0.300 | 0.300 | 100% |
| clf_wine_m | multiclass | 0.400 | 0.400 | 0.400 | 0.400 | 100% |
| reg_diabetes_s | regression (MSE) | 6.0108 | 6.0108 | 6.0108 | 6.0108 | 100% |
| reg_synth_m | regression (MSE) | 5.0389 | 5.0389 | 5.0389 | 5.0389 | 100% |
| reg_synth_w | regression (MSE) | 5.2348 | 5.2348 | 5.2348 | 5.2348 | 100% |

### Compute / memory footprint

Peak RSS is sampled externally per subprocess. *Baseline* = interpreter + framework + model + data, no inference. *Delta* isolates the inference transients.

| dataset | mode | Python (PyTorch) base / peak / Δ MB | Python (PyO3) base / peak / Δ MB | Rust native base / peak / Δ MB | WASM (Bun) base / peak / Δ MB |
|---|---|---:|---:|---:|---:|
| clf_breast_l | full | 423 / 460 / 36.4 | 35 / 68 / 33.0 | 7 / 40 / 32.6 | 83 / 239 / 156.2 |
| clf_breast_l | cached | 423 / 459 / 35.3 | 35 / 59 / 23.7 | 7 / 31 / 23.1 | 83 / 189 / 106.3 |
| clf_digits_w | full | 423 / 471 / 48.2 | 35 / 87 / 51.7 | 7 / 58 / 50.8 | 74 / 185 / 110.9 |
| clf_digits_w | cached | 423 / 471 / 48.2 | 35 / 77 / 41.8 | 7 / 49 / 42.2 | 74 / 246 / 171.5 |
| clf_iris_nan | full | 422 / 436 / 14.6 | 35 / 41 / 5.4 | 7 / 12 / 5.6 | 72 / 130 / 58.5 |
| clf_iris_nan | cached | 422 / 434 / 12.3 | 35 / 40 / 5.0 | 7 / 12 / 5.4 | 72 / 114 / 41.8 |
| clf_iris_s | full | 423 / 433 / 10.2 | 35 / 40 / 5.2 | 9 / 12 / 3.2 | 77 / 119 / 41.3 |
| clf_iris_s | cached | 423 / 433 / 10.2 | 35 / 40 / 4.6 | 9 / 12 / 3.3 | 77 / 116 / 38.6 |
| clf_wine_m | full | 422 / 439 / 16.7 | 35 / 45 / 9.2 | 7 / 16 / 9.4 | 79 / 140 / 60.9 |
| clf_wine_m | cached | 422 / 439 / 17.0 | 35 / 43 / 7.9 | 7 / 15 / 7.9 | 79 / 124 / 45.1 |
| reg_diabetes_s | full | 423 / 437 / 13.6 | 36 / 42 / 5.2 | 7 / 13 / 6.3 | 77 / 131 / 54.0 |
| reg_diabetes_s | cached | 423 / 435 / 12.1 | 36 / 41 / 4.3 | 7 / 12 / 5.4 | 77 / 117 / 40.1 |
| reg_synth_m | full | 424 / 442 / 17.7 | 35 / 48 / 12.6 | 10 / 20 / 10.1 | 87 / 143 / 56.0 |
| reg_synth_m | cached | 424 / 441 / 16.7 | 35 / 46 / 10.5 | 10 / 17 / 7.3 | 87 / 143 / 56.3 |
| reg_synth_w | full | 422 / 471 / 49.0 | 37 / 83 / 45.6 | 8 / 53 / 45.8 | 87 / 220 / 133.1 |
| reg_synth_w | cached | 422 / 470 / 48.0 | 37 / 75 / 37.7 | 8 / 44 / 37.0 | 87 / 193 / 106.1 |

Randomized-block two-way ANOVA on inference delta (full forward; dataset is the block, runtime the treatment):

| factor | df | F | p | partial η² |
|---|---:|---:|---:|---:|
| runtime | 3 | 62.8 | 1.18e-10 | 0.900 |
| dataset | 7 | 27.9 | 2.92e-09 | 0.903 |

Python→Rust FFI call overhead (no-op `ping`, 20k calls): **0.03 µs/call** — negligible against multi-millisecond inference.

<!-- BENCH:END -->

## Scope and limitations

* Inference only: no autograd, no `torch.compile`, no checkpointing. No
  quantisation (full f32 safetensors), no batched ensembles (`B=1`, but `B` is
  kept in tensor shapes), and no stage-0-2 row/column chunking.
* WASM is CPU SIMD128, not WebGPU.
* Attention is query-chunked (`Q_CHUNK=128`) so score memory stays
  `O(H·chunk·N)` instead of `O(H·N²)`; this is mathematically equivalent and
  is what keeps WASM32 inside its memory budget.
* Validated on the tiny parity fixture, not the full 24-layer checkpoint.
