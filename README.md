# TabPFN-Rust

Inference-only port of TabPFN v3.5 to Rust, using Candle. Ships as a Python
extension and as a WebAssembly module.

The network is the same as `tabpfn/architectures/tabpfn_v3_5.py`: same stages,
shapes and config defaults, and the original PyTorch `state_dict` key names, so
checkpoints load without a rename map. Outputs match PyTorch to about 1e-6 on
the same weights.

No training, no autograd, no quantisation. CPU only.

## Layout

```
crates/tabpfn-core/   Rust + Candle model, preprocessing, KV cache
crates/tabpfn-wasm/   wasm-bindgen bindings (JS/TS)
crates/tabpfn-py/     PyO3 bindings (CPython extension)
docs/RUST.md          using the core directly from Rust
scripts/              fixtures, dataset generation, benchmarks, stats
main.ts               feature demo
smoke.test.ts         Bun tests
```

## Build

```bash
# Rust core tests
cargo test -p tabpfn-core

# WASM + JS glue (outputs to pkg/, git-ignored)
npm run build:wasm

# Python extension (outputs scripts/tabpfn_rs.pyd / .so)
cargo build -p tabpfn-py --release
copy target\release\tabpfn_rs.dll scripts\tabpfn_rs.pyd   # Windows
# cp target/release/libtabpfn_rs.so scripts/tabpfn_rs.so  # Linux/macOS
```

`scripts/fixtures/model.safetensors` is a small 2-layer model with fixed
weights. It is committed so the Rust and WASM tests run without PyTorch.

To publish the WASM module as an npm package, see
[docs/PUBLISH.md](docs/PUBLISH.md).

## Using the bindings

Both bindings take and return flat little-endian `f32` buffers. Row-major,
shape `(rows, cols)`. Call `fit()` once per training set, then `predict()` as
often as you like.

### Python

```python
import numpy as np
from tabpfn_rs import TabPfn

# weights: raw safetensors bytes
# config_json: the architecture config as JSON (from the checkpoint metadata)
m = TabPfn(open("model.safetensors", "rb").read(), config_json)

x_train = np.ascontiguousarray(x_train, dtype=np.float32)
x_test = np.ascontiguousarray(x_test, dtype=np.float32)
y_train = np.ascontiguousarray(y_train, dtype=np.float32)

# classification
m.fit(x_train.tobytes(), len(x_train), x_train.shape[1], y_train.tobytes(), "multiclass")
logits = np.frombuffer(
    m.predict(x_test.tobytes(), len(x_test), x_test.shape[1], "multiclass"),
    dtype="<f4",
).reshape(len(x_test), -1)
labels = logits.argmax(axis=1)

# regression: same calls with "regression"; output is (n_test, num_buckets)
```

One `TabPfn` instance holds one fitted training set. Build a second instance
for a second dataset.

### JavaScript / TypeScript

```ts
import init, { TabPfnWasm } from "./pkg/tabpfn_wasm.js";

await init();

// weights: raw safetensors bytes; configJson: architecture config as JSON
const weights = new Uint8Array(await (await fetch("model.safetensors")).arrayBuffer());
const m = new TabPfnWasm(weights, configJson);

const xTrain = new Float32Array(flatTrainingRows);
const xTest = new Float32Array(flatTestRows);
const yTrain = new Float32Array(trainLabels);

m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
const logits = m.predict(xTest, nTest, nCols, "multiclass");
// logits: Float32Array, row-major (nTest * numOutputs)
```

The module is built with `-C target-feature=+simd128`. Bun and browsers both
work; with the `web` target, load it from a page or worker, not from Node's
`fetch`.

### Output shapes

| task | output | how to read it |
|---|---|---|
| `multiclass` | `n_test x max_num_classes` logits | `argmax` per row |
| `regression` | `n_test x num_buckets` bar logits | softmax, then weighted mean over bucket centres |

Bucket borders are stored in the checkpoint as `heads.regression_borders`
(`num_buckets + 1` values). Read them from the safetensors file to build the
centres. The bindings do not expose them yet.

### Row embeddings

`encode()` returns the representation the decoder consumes, `(rows,
icl_emsize)`. It is contextual, so the same row produces a different vector for
a different training set. Train rows include their target embedding, so they
cluster by label; use test rows for a label-free view.

```python
emb = np.frombuffer(
    m.encode(x_all.tobytes(), n_train, n_test, n_cols, y_train.tobytes(), "multiclass"),
    dtype="<f4",
).reshape(n_train + n_test, -1)
```

```ts
const emb = m.encode(xAll, nTrain, nTest, nCols, yTrain, "multiclass");
```

`python scripts/embed_projection.py --dataset clf_digits_w` projects these with
PCA, t-SNE and UMAP into `scripts/results/embeddings_*.png`. With the committed
random-weight fixture the test-row silhouette is 0.00, which is expected: the
pipeline works, the weights are not trained.

### Core Rust

See [docs/RUST.md](docs/RUST.md) for loading weights, `forward`, `fit` /
`predict_with_y`, `encode`, and the runnable `basic_usage` example.

## Examples

Runnable consumers of the published package live in [examples](examples):

| runtime | directory | run |
|---|---|---|
| Bun | [examples/bun](examples/bun) | `cd examples/bun && bun install && bun run main.ts` |
| Node | [examples/node](examples/node) | `cd examples/node && npm install && node main.mjs` |
| browser, no bundler | [examples/browser](examples/browser) | serve the repo root, open the page |

```bash
npm install tabpfn-wasm
# or
bun add tabpfn-wasm
```

```ts
import init, { TabPfnWasm } from "tabpfn-wasm";

await init();                                   // once
const m = new TabPfnWasm(weights, configJson);  // weights: safetensors bytes
m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
const logits = m.predict(xTest, nTest, nCols, "multiclass");
```

The examples read `scripts/fixtures/` (a random-initialised 2-layer model), so
they run without the licensed checkpoint. Node needs the `.wasm` bytes passed
to `init()` because the package ships the `web` target; [examples/node](examples/node)
shows that call.


### Why caching helps

Test rows attend only to training rows, never to each other. The training-side
work therefore only has to run once. `fit()` stores the inducing hidden states,
the per-layer ICL keys and values, the decoder keys, the scaler statistics and
the ECDF context. `predict()` then processes just the new rows.

| cached item | shape | why it can be reused |
|---|---|---|
| inducing hidden states | `(B*C, n_ind, E)` per block | built from train rows |
| ICL keys and values | `(B, N, n_kv_heads_test, head_dim)` per layer | train rows only, already normalised |
| decoder keys | `(B, N, H_dec, D_dec)` | multiclass only, built from train rows |
| scaler mean/std | `(C,)` | fitted on train rows |
| ECDF context | `(3, B, C, K)` | bucket edges from train rows |

Cache size for the released config, per estimator:

```
ICL K/V       12,288 * N bytes     12 MB per 1,000 train rows
decoder keys   1,536 * N bytes     multiclass only
inducing     196,608 * C bytes     20 MB at 100 columns
ECDF context  98,304 * C bytes     10 MB at 100 columns
```

If you only call `predict()` once, skip `fit()` and use the single-pass
`forward` path instead.

## Using the real checkpoint (optional)

Everything above runs on a small random-weight fixture. Real predictions need
the upstream TabPFN v3.5 weights.

**License:** the code in this repo is Apache-2.0. The TabPFN-3.5 weights are
licensed separately by Prior Labs and are **non-commercial**. They are not
included or redistributed here. Downloading them means accepting
[their license](https://huggingface.co/Prior-Labs/tabpfn_3_5). Keep them out of
git (`models/` is ignored).

```powershell
python scripts/fetch_checkpoint.py
# models/tabpfn-v3.5-20260909.safetensors   ~ the weights
# models/config.json                        ~ architecture config
```

The script downloads from the public `Prior-Labs/tabpfn_3_5` repository, reads
the architecture config out of the safetensors header metadata, and checks the
keys against what this port requires. Use `--variant fast` for the smaller
checkpoint, or `--file path.safetensors` to read a file you already have.

Point the examples at it:

```powershell
$env:TABPFN_MODEL  = "models/tabpfn-v3.5-20260909.safetensors"
$env:TABPFN_CONFIG = "models/config.json"
cd examples/bun; bun run main.ts
```

The browser example takes query parameters:
`?model=/models/tabpfn-v3.5-20260909.safetensors&config=/models/config.json`.

For Rust, pass the same two paths to `TabPFNV3p5::from_safetensors` and
`serde_json::from_str` (see [docs/RUST.md](docs/RUST.md)).

One honest caveat: this port has only been validated numerically against the
tiny fixture. The full checkpoint is a large download and has not been run
here. If upstream changes the config schema, the fetch script reports the
missing fields instead of failing silently.

## Nerd stuff

Benchmarks and hypothesis tests. Everything here is reproducible from the
committed fixtures.

### What is compared

| runtime | what it is |
|---|---|
| `python_pytorch` | upstream Python / PyTorch TabPFN v3.5, CPU only (`torch 2.9.1+cpu`) |
| `python_bindings` | the Rust core through the PyO3 extension |
| `rust_native` | Rust / Candle, native release binary |
| `wasm_bun` | Rust / Candle as WASM SIMD128, run under Bun |

Two code paths per dataset:

* **full**: one forward over train and test rows together.
* **cached**: `fit()` once, then time `predict()` on test rows.

A third **baseline** run loads the model and data without inference, so its
peak RSS can be subtracted to isolate inference memory.

### Datasets

Eight sets at four sizes, generated by `scripts/gen_datasets.py` and passed to
every runtime as the same hex-encoded f32 bytes, so NaN and infinities survive.

| dataset | task | train | test | cols |
|---|---|---:|---:|---:|
| clf_iris_s | multiclass | 30 | 10 | 4 |
| clf_wine_m | multiclass | 100 | 25 | 13 |
| clf_breast_l | multiclass | 300 | 50 | 30 |
| clf_digits_w | multiclass | 200 | 30 | 64 |
| clf_iris_nan | multiclass | 40 | 12 | 4 (NaN, inf, out-of-range) |
| reg_diabetes_s | regression | 40 | 10 | 10 |
| reg_synth_m | regression | 120 | 25 | 20 |
| reg_synth_w | regression | 250 | 40 | 50 |

Peak RSS is sampled from outside each process with `psutil`, so all four
runtimes are measured the same way.

The fixture weights are random (2-layer model). Predictive accuracy is
therefore chance level and identical across runtimes. The accuracy table below
checks that every runtime makes the same predictions, which is what matters for
a port. Real accuracy needs the licensed checkpoint.

### Reproduce

```bash
pip install tabpfn safetensors
python scripts/export_weights.py
python scripts/gen_datasets.py

cargo build --release -p tabpfn-core --example harness_datasets
npm run build:wasm
cargo build -p tabpfn-py --release
copy target\release\tabpfn_rs.dll scripts\tabpfn_rs.pyd

python scripts/run_benchmarks.py --reps 10
python scripts/analyze.py --write-readme    # rewrites the block below
```

Tests: one-way ANOVA per dataset on speed and memory, a pooled two-way ANOVA
(runtime x dataset, `log10(ms)`), Tukey post-hoc comparisons, and a one-sided
equivalence test on parity (mean absolute logit difference below 1e-4).

<!-- BENCH:START -->
8 datasets, 10 reps each. Speed ANOVAs use `log10(ms)`; memory uses `log10(delta + 1)`. Equivalence tolerance `0.0001`.

### What the numbers say

Rust native was the fastest full-forward runtime on 3 of 8 datasets (clf_iris_nan, clf_iris_s and reg_diabetes_s). PyTorch took the rest. With the cache, Rust native was fastest on 5 of 8 (clf_iris_nan, clf_iris_s, clf_wine_m, reg_diabetes_s and reg_synth_m).

The split is consistent rather than noise. Which runtime wins depends on the dataset, and that interaction is significant in both modes:

| mode | interaction F | p | partial eta^2 |
|---|---:|---:|---:|
| full | 190 | 3.59e-155 | 0.93 |
| cached | 478 | 4.07e-210 | 0.97 |

In practice, narrow tables favour the Rust path because it avoids the per-operation overhead PyTorch pays on small tensors. Wide tables favour PyTorch, which uses tuned BLAS kernels and fused attention while this port loops over heads with small matrices. Caching moves the crossover toward Rust, since the training rows leave the hot path.

### Speed: full forward (train and test together)

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

Two-way ANOVA on `log10(ms)`:

| factor | df | F | p | partial eta^2 |
|---|---:|---:|---:|---:|
| runtime | 3 | 519.4 | 8.05e-116 | 0.844 |
| dataset | 7 | 5434.2 | <1e-300 | 0.992 |
| interaction | 21 | 190.0 | 3.59e-155 | 0.933 |

Tukey post-hoc comparisons, runtimes pooled:

| pair | delta log10(ms) | q | p | significant |
|---|---:|---:|---:|---|
| Python (PyTorch) vs Python (PyO3) | -0.184 | 33.31 | <1e-300 | yes |
| Python (PyTorch) vs Rust native | -0.211 | 38.18 | <1e-300 | yes |
| Python (PyTorch) vs WASM (Bun) | -0.300 | 54.26 | <1e-300 | yes |
| Python (PyO3) vs Rust native | -0.027 | 4.87 | 0.00362 | yes |
| Python (PyO3) vs WASM (Bun) | -0.116 | 20.95 | <1e-300 | yes |
| Rust native vs WASM (Bun) | -0.089 | 16.07 | <1e-300 | yes |

### Speed: cached (fit once, predict test rows)

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

Two-way ANOVA on `log10(ms)`:

| factor | df | F | p | partial eta^2 |
|---|---:|---:|---:|---:|
| runtime | 3 | 876.6 | 1.98e-144 | 0.901 |
| dataset | 7 | 9099.1 | <1e-300 | 0.995 |
| interaction | 21 | 478.1 | 4.07e-210 | 0.972 |

Tukey post-hoc comparisons, runtimes pooled:

| pair | delta log10(ms) | q | p | significant |
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
| Python (PyTorch) | 37.4 | 13.0 | 2.9x |
| Python (PyO3) | 115.8 | 14.4 | 8.0x |
| Rust native | 115.8 | 15.4 | 7.5x |
| WASM (Bun) | 179.7 | 18.6 | 9.7x |

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

PyO3, native Rust and WASM all pass the one-sided equivalence test on every dataset and mode (mean|diff| below 0.0001, p < 0.05). Worst observed max |diff| was 1.43e-06.

### Accuracy / agreement

The fixture uses randomly initialised weights, so predictive accuracy is chance level. What this table checks is that all four runtimes make the same predictions.

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

Peak RSS is sampled from outside each process. *Baseline* is interpreter, framework, model and data with no inference. *Delta* is the inference transient.

| dataset | mode | Python (PyTorch) base / peak / delta MB | Python (PyO3) base / peak / delta MB | Rust native base / peak / delta MB | WASM (Bun) base / peak / delta MB |
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

Randomized-block ANOVA on inference memory. Dataset is the block, runtime is the treatment:

| factor | df | F | p | partial eta^2 |
|---|---:|---:|---:|---:|
| runtime | 3 | 62.8 | 1.18e-10 | 0.900 |
| dataset | 7 | 27.9 | 2.92e-09 | 0.903 |

Python to Rust call overhead, measured with a no-op call 20,000 times: **0.03 us/call**. Negligible next to inference.

<!-- BENCH:END -->

## Scope

* Inference only. No autograd, no `torch.compile`, no checkpointing, no
  quantisation, no batched ensembles (`B = 1`), no row or column chunking in
  stages 0 to 2.
* WASM is CPU SIMD128. There is no WebGPU path.
* Attention is query-chunked (`Q_CHUNK = 128`), so score memory is
  `O(H * chunk * N)` instead of `O(H * N^2)`. This is mathematically equivalent
  and keeps WASM32 inside its memory budget.
* Validated on the tiny parity fixture. The full 24-layer checkpoint has not
  been run.
