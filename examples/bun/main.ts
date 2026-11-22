// Bun example for the published WASM package.
//
//   cd examples/bun
//   bun install
//   bun run main.ts
//
// The repo's committed fixtures are used by default. Point TABPFN_ROOT at the
// repository root to run the same file from somewhere else.

import init, { TabPfnWasm } from "tabpfn-wasm";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.env.TABPFN_ROOT ?? join(import.meta.dir, "../..");
const FIXTURES = join(ROOT, "scripts", "fixtures");

// Point these at a real checkpoint if you have one (see scripts/fetch_checkpoint.py).
const MODEL = process.env.TABPFN_MODEL ?? join(FIXTURES, "model.safetensors");
const CONFIG = process.env.TABPFN_CONFIG ?? join(FIXTURES, "config.json");

// wasm-bindgen modules need one async init before use.
await init();

const weights = new Uint8Array(readFileSync(MODEL));
const configJson = readFileSync(CONFIG, "utf8");

const hexToF32 = (hex: string) => {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return new Float32Array(bytes.buffer);
};

const spec = JSON.parse(readFileSync(join(ROOT, "scripts", "datasets.json"), "utf8"));
const load = (name: string) => spec.datasets.find((d: any) => d.name === name);

// ---------------------------------------------------------------- classification
{
  const d = load("clf_iris_s");
  const xTrain = hexToF32(d.x_train);
  const xTest = hexToF32(d.x_test);
  const yTrain = hexToF32(d.y_train);

  const m = new TabPfnWasm(weights, configJson);

  const t0 = performance.now();
  m.fit(xTrain, d.n_train, d.n_cols, yTrain, "multiclass");
  const tFit = performance.now() - t0;

  const t1 = performance.now();
  const logits = m.predict(xTest, d.n_test, d.n_cols, "multiclass");
  const tPredict = performance.now() - t1;

  const nClasses = logits.length / d.n_test;
  const labels: number[] = [];
  for (let row = 0; row < d.n_test; row++) {
    let best = 0;
    for (let c = 1; c < nClasses; c++) {
      if (logits[row * nClasses + c] > logits[row * nClasses + best]) best = c;
    }
    labels.push(best);
  }

  console.log(`multiclass: ${d.n_train} train rows, ${d.n_cols} cols`);
  console.log(`  fit ${tFit.toFixed(1)} ms, predict ${tPredict.toFixed(1)} ms`);
  console.log(`  first labels: ${labels.slice(0, 10).join(", ")}`);
}

// ------------------------------------------------------------------- regression
{
  const d = load("reg_diabetes_s");
  const xTrain = hexToF32(d.x_train);
  const xTest = hexToF32(d.x_test);
  const yTrain = hexToF32(d.y_train);

  const m = new TabPfnWasm(weights, configJson);
  m.fit(xTrain, d.n_train, d.n_cols, yTrain, "regression");
  const out = m.predict(xTest, d.n_test, d.n_cols, "regression");

  const nBuckets = out.length / d.n_test;
  console.log(`regression: ${d.n_train} train rows, ${d.n_cols} cols`);
  console.log(`  ${nBuckets} bucket logits per row, first 4: ${Array.from(out.slice(0, 4)).map((v) => v.toFixed(3)).join(", ")}`);
}

// --------------------------------------------------------------- row embeddings
{
  const d = load("clf_iris_s");
  const xTrain = hexToF32(d.x_train);
  const xTest = hexToF32(d.x_test);
  const yTrain = hexToF32(d.y_train);
  const xAll = new Float32Array(xTrain.length + xTest.length);
  xAll.set(xTrain, 0);
  xAll.set(xTest, xTrain.length);

  const m = new TabPfnWasm(weights, configJson);
  const emb = m.encode(xAll, d.n_train, d.n_test, d.n_cols, yTrain, "multiclass");
  console.log(`embeddings: ${emb.length / (d.n_train + d.n_test)} dims per row, ${d.n_train + d.n_test} rows`);
}

console.log("\nNote: the committed fixture is a randomly initialised 2-layer model.");
console.log("Predictions are structurally valid but not meaningful.");
