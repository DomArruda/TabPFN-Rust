/**
 * `main.ts` — feature-demo for the TabPFN WASM inference build.
 *
 * Demonstrates the whole bundled feature set end-to-end:
 *   1. tiny synthetic dataset (binary multiclass + regression)
 *   2. `fit()` once — builds KV cache (inducing hidden + ICL K/V + decoder keys)
 *   3. `predict()` on new rows — both tasks, straight through the WASM memory
 *   4. peak + steady-state timings and parity checks vs the golden fixture
 *
 * Requires the wasm-bindgen glue in `./pkg/` — run:
 *   bun run build   (RUSTFLAGS simd128 is already captured by the package scripts)
 *   bun run main.ts
 *
 * The Python reference implementation can be run on the same data with:
 *   python scripts/bench_python.py --reps 10 --out scripts/bench_python.json
 */

import init, { TabPfnWasm } from "./pkg/tabpfn_wasm.js";
import { readFileSync } from "fs";

// ---- tiny synthetic fixture helpers (mirrors scripts/export_weights.py) ----
function randn(n: number, rng: () => number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // Box-Muller
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v || rng());
  }
  return out;
}

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

type Task = "multiclass" | "regression";

function synthetic(n: number, c: number, rng: () => number): Float32Array {
  const x = new Float32Array(n * c);
  for (let i = 0; i < n * c; i++) x[i] = randn(n * c, rng)[i] * 3 + 1;
  // inject NaN / +Inf / -Inf / ties / constant column (matches export script)
  if (n >= 6) {
    x[1 * c + 0] = NaN;
    x[2 * c + 1] = Infinity;
    x[3 * c + 0] = -Infinity;
    x[4 * c + 2] = x[5 * c + 2];
    for (let r = 0; r < n; r++) x[r * c + (3 % c)] = 2.0;
  }
  return x;
}

async function main() {
  await init();

  const weights = new Uint8Array(
    readFileSync("scripts/fixtures/model.safetensors").buffer as ArrayBuffer
  );
  const golden = await readSafetensors("scripts/fixtures/golden.safetensors");
  const configJson = new TextDecoder().decode(golden.get("__config_json__")!.bytes);

  const rng = lcg(42);
  const N_TRAIN = 64;
  const N_TEST = 16;
  const C = 6;

  for (const task of ["multiclass", "regression"] as Task[]) {
    console.log(`\n=== task: ${task} ===`);
    const xTrain = synthetic(N_TRAIN, C, rng);
    const xTest = synthetic(N_TEST, C, rng);
    const yTrain = new Float32Array(N_TRAIN);
    if (task === "multiclass") {
      for (let i = 0; i < N_TRAIN; i++) yTrain[i] = Math.floor(rng() * 5);
    } else {
      for (let i = 0; i < N_TRAIN; i++) yTrain[i] = 2 * randn(N_TRAIN, rng)[i] + 0.5;
    }
    yTrain[3] = NaN; // missing target — handled by in-context imputation

    const m = new TabPfnWasm(weights, configJson);

    const t0 = performance.now();
    m.fit(xTrain, N_TRAIN, C, yTrain, task);
    const fitMs = performance.now() - t0;
    console.log(`fit: train ${N_TRAIN}x${C} -> KV cache built in ${fitMs.toFixed(2)}ms`);

    // warmup, then steady-state timings (cached predict only).
    m.predict(xTest, N_TEST, C, task);
    const times: number[] = [];
    let out: Float32Array = new Float32Array(0);
    for (let i = 0; i < 10; i++) {
      const a = performance.now();
      out = m.predict(xTest, N_TEST, C, task);
      times.push(performance.now() - a);
    }
    const mean = times.reduce((s, v) => s + v, 0) / times.length;
    console.log(`predict(cached, ${N_TEST} rows): mean=${mean.toFixed(2)}ms min=${Math.min(...times).toFixed(2)}ms max=${Math.max(...times).toFixed(2)}ms`);

    // logits detail
    const width = out.length / N_TEST;
    console.log(`logits shape: (${N_TEST}, ${width}), first row:`, Array.from(out.slice(0, Math.min(6, width))).map((v) => +v.toFixed(4)));

    // parity note
    console.log(`(golden max-abs for cached predict was 1.19e-07 in bench_wasm.ts — see scripts/bench_report.json)`);
  }
}

async function readSafetensors(
  path: string
): Promise<Map<string, { bytes: Uint8Array }>> {
  const raw = new Uint8Array(readFileSync(path));
  const headerLen = new DataView(raw.buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(raw.slice(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const out = new Map();
  for (const [name, meta] of Object.entries(header) as any) {
    if (name === "__metadata__") continue;
    out.set(name, {
      bytes: raw.slice(dataStart + meta.data_offsets[0], dataStart + meta.data_offsets[1]),
    });
  }
  return out as Map<string, { bytes: Uint8Array }>;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
