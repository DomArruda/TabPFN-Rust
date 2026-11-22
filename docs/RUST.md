# Using the Rust core

Most people will use the Python or WASM bindings. This page is for calling
`tabpfn-core` directly from Rust.

## Add the dependency

```toml
[dependencies]
tabpfn-core = { path = "crates/tabpfn-core" }
candle-core = "0.10"
serde_json = "1"
```

## Load a model

Weights are a standard safetensors file that keeps the PyTorch `state_dict`
key names, so a checkpoint loads without a rename map. The architecture config
travels as JSON.

```rust
use candle_core::{Device, Tensor};
use tabpfn_core::{TabPFNV3p5, TabPFNV3p5Config, TaskType};

let device = Device::Cpu;
let config: TabPFNV3p5Config = serde_json::from_str(&config_json)?;
let model = TabPFNV3p5::from_safetensors(&weights_bytes, config, &device)?;
```

Tensors use `(batch, rows, columns)` layout. Batch is 1 for now, but the
dimension is kept so batched ensembles can be added later.

```rust
let x_train = Tensor::from_vec(x_train_flat, (1, n_train, n_cols), &device)?;
let x_test  = Tensor::from_vec(x_test_flat,  (1, n_test,  n_cols), &device)?;
```

## Single pass

Pass training rows first, then test rows, in one tensor.

```rust
let mut x_all = x_train_flat.clone();
x_all.extend_from_slice(&x_test_flat);
let x_all = Tensor::from_vec(x_all, (1, n_train + n_test, n_cols), &device)?;

let out = model.forward(
    &x_all,
    &y_train,
    TaskType::Multiclass,
    None,   // no cache
    false,  // x is not test-only
    None,   // infer the number of classes from y_train
)?;
```

`out` is `(n_test, 1, num_outputs)`. Flatten it to a `Vec<f32>`:

```rust
let logits: Vec<f32> = out
    .to_vec3::<f32>()?
    .into_iter()
    .flat_map(|m| m.into_iter().flat_map(|r| r.into_iter()))
    .collect();
```

Multiclass output is one logit per class. Regression output is one logit per
bucket. To turn regression buckets into a point estimate, read
`heads.regression_borders` from the safetensors file (length
`num_buckets + 1`), softmax the logits, and take the weighted mean of the
bucket centres.

## Fit once, predict many

`fit()` runs the training-only work and returns a cache. `predict_with_y()`
reuses it, so repeated calls only process the new rows.

```rust
let cache  = model.fit(&x_train, &y_train, TaskType::Multiclass)?;
let logits = model.predict_with_y(&cache, &x_test, &y_train, TaskType::Multiclass)?;
```

The cache is tied to the training set it was built from. `predict_with_y`
returns a `Shape` error if the column count, train-row count, batch size or
layer count do not match, or if a regression cache is used for multiclass.

You can also `clone()` the cache and share it across threads, or keep several
caches for different training sets.

## Row embeddings

`encode()` returns the representation the decoder consumes, after the ICL
transformer and output norm: `(batch, rows, icl_emsize)`. It is contextual, so
the same row gets a different vector for a different training set. Train rows
include their target embedding by design, which makes them cluster by label.
Score test rows if you want a label-free view.

```rust
let emb_all  = model.encode(&x_all, &y_train, TaskType::Multiclass)?;         // train + test
let emb_test = model.encode_cached(&cache, &x_test, &y_train, TaskType::Multiclass)?;
```

## Runnable example

`crates/tabpfn-core/examples/basic_usage.rs` is a complete program that reads
the committed fixtures and prints logits and embedding shapes.

```bash
cargo run --release -p tabpfn-core --example basic_usage
# clf_iris_s: 30 train / 10 test rows, 4 cols, task=Multiclass
# cached predict -> 100 values; first 6: [-1.1866, -1.0800, -1.0350, -10.1266, ...]
# encode -> embeddings [1, 40, 128]  (rows x icl_emsize)
# encode_cached -> embeddings [1, 10, 128]
```

## Errors

Everything returns `tabpfn_core::Error`, which covers Candle, safetensors,
config, shape and IO failures. It implements `std::error::Error`, so `?` works
in `fn main() -> Result<(), Box<dyn std::error::Error>>`.

## Supported output types

| task | output | notes |
|---|---|---|
| `TaskType::Multiclass` | `(n_test, 1, max_num_classes)` | logits, `argmax` for the label |
| `TaskType::Regression` | `(n_test, 1, num_buckets)` | bar-distribution logits |

Back to the [README](../README.md).
