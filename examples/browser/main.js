// Browser example. The import map in index.html points at the npm CDN, so
// there is no bundler step. Fetch the model and data from the served repo root.

import init, { TabPfnWasm } from "tabpfn-wasm";

const out = document.getElementById("out");
const log = (line) => {
  out.textContent += line + "\n";
};

// Override with ?model=/models/ckpt.safetensors&config=/models/config.json
const params = new URLSearchParams(location.search);
const MODEL_URL = params.get("model") ?? "/scripts/fixtures/model.safetensors";
const CONFIG_URL = params.get("config") ?? "/scripts/fixtures/config.json";

const hexToF32 = (hex) => {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return new Float32Array(bytes.buffer);
};

try {
  out.textContent = "";
  log("initialising wasm…");
  await init();

  const [weightsBuf, configJson, spec] = await Promise.all([
    fetch(MODEL_URL).then((r) => r.arrayBuffer()),
    fetch(CONFIG_URL).then((r) => r.text()),
    fetch("/scripts/datasets.json").then((r) => r.json()),
  ]);

  const weights = new Uint8Array(weightsBuf);
  const d = spec.datasets.find((x) => x.name === "clf_iris_s");

  const xTrain = hexToF32(d.x_train);
  const xTest = hexToF32(d.x_test);
  const yTrain = hexToF32(d.y_train);

  const m = new TabPfnWasm(weights, configJson);

  const t0 = performance.now();
  m.fit(xTrain, d.n_train, d.n_cols, yTrain, "multiclass");
  log(`fit: ${(performance.now() - t0).toFixed(1)} ms`);

  const t1 = performance.now();
  const logits = m.predict(xTest, d.n_test, d.n_cols, "multiclass");
  log(`predict: ${(performance.now() - t1).toFixed(1)} ms`);

  const nClasses = logits.length / d.n_test;
  const labels = [];
  for (let row = 0; row < d.n_test; row++) {
    let best = 0;
    for (let c = 1; c < nClasses; c++) {
      if (logits[row * nClasses + c] > logits[row * nClasses + best]) best = c;
    }
    labels.push(best);
  }
  log(`labels: ${labels.join(", ")}`);

  const xAll = new Float32Array(xTrain.length + xTest.length);
  xAll.set(xTrain, 0);
  xAll.set(xTest, xTrain.length);
  const emb = m.encode(xAll, d.n_train, d.n_test, d.n_cols, yTrain, "multiclass");
  log(`embeddings: ${emb.length / (d.n_train + d.n_test)} dims x ${d.n_train + d.n_test} rows`);

  log("");
  log("Note: the fixture is a randomly initialised 2-layer model.");
  log("Predictions are structurally valid but not meaningful.");
} catch (err) {
  out.textContent = "error: " + (err && err.message ? err.message : String(err));
  console.error(err);
}
