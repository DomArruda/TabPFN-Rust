// Bun harness: WASM build of tabpfn-wasm. Same CLI contract as the others.
//   bun run scripts/harness_wasm.ts --dataset clf_iris_s --reps 10 --mode full --out out.json

import init, { TabPfnWasm } from "../pkg/tabpfn_wasm.js";
import { readFileSync, writeFileSync } from "fs";

function arg(name: string, fallback?: string): string {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  if (fallback !== undefined) return fallback;
  throw new Error(`missing --${name}`);
}

function hexF32(s: string): Float32Array {
  const bytes = new Uint8Array(s.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16);
  }
  return new Float32Array(bytes.buffer);
}

async function main() {
  await init();

  const datasetsPath = arg("datasets", "scripts/datasets.json");
  const name = arg("dataset");
  const reps = parseInt(arg("reps", "10"), 10);
  const mode = arg("mode");
  const outPath = arg("out");

  const spec = JSON.parse(readFileSync(datasetsPath, "utf8"));
  const d = spec.datasets.find((x: any) => x.name === name);
  if (!d) throw new Error(`dataset ${name} not found`);

  const weights = new Uint8Array(readFileSync("scripts/fixtures/model.safetensors"));
  const goldenRaw = new Uint8Array(readFileSync("scripts/fixtures/golden.safetensors"));
  const headerLen = new DataView(goldenRaw.buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(goldenRaw.slice(8, 8 + headerLen)));
  const cfgMeta = header["__config_json__"];
  const configJson = new TextDecoder().decode(
    goldenRaw.slice(8 + headerLen + cfgMeta.data_offsets[0], 8 + headerLen + cfgMeta.data_offsets[1])
  );

  const xTrain = hexF32(d.x_train);
  const xTest = hexF32(d.x_test);
  const yTrain = hexF32(d.y_train);

  const m = new TabPfnWasm(weights, configJson);

  const report: any = {
    runtime: "wasm_bun",
    dataset: d.name,
    task: d.task,
    mode,
    reps: mode === "baseline" ? 0 : reps,
    n_train: d.n_train,
    n_test: d.n_test,
    n_cols: d.n_cols,
  };

  if (mode === "baseline") {
    report.timings_ms = [];
    report.logits = [];
    writeFileSync(outPath, JSON.stringify(report));
    return;
  }

  const xAll = new Float32Array(xTrain.length + xTest.length);
  xAll.set(xTrain, 0);
  xAll.set(xTest, xTrain.length);

  let timings: number[] = [];
  let logits: number[] = [];

  if (mode === "full") {
    m.forward(xAll, d.n_train, d.n_test, d.n_cols, yTrain, d.task); // warmup
    for (let i = 0; i < reps; i++) {
      const t0 = performance.now();
      const out = m.forward(xAll, d.n_train, d.n_test, d.n_cols, yTrain, d.task);
      timings.push(performance.now() - t0);
      logits = Array.from(out);
    }
  } else {
    m.fit(xTrain, d.n_train, d.n_cols, yTrain, d.task);
    m.predict(xTest, d.n_test, d.n_cols, d.task); // warmup
    for (let i = 0; i < reps; i++) {
      const t0 = performance.now();
      const out = m.predict(xTest, d.n_test, d.n_cols, d.task);
      timings.push(performance.now() - t0);
      logits = Array.from(out);
    }
  }

  report.timings_ms = timings;
  report.logits = logits;
  writeFileSync(outPath, JSON.stringify(report));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
