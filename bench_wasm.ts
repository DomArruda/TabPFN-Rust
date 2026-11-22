// Bun/Node runner: instantiates the WebAssembly build of `tabpfn-wasm`,
// reads the SAME fixture weights/config/data the Rust + Python harnesses
// use, and writes raw timings + logits to `scripts/bench_wasm.json`.
//
// This is the user-facing smoke harness. Everything goes through the
// `TabPfnWasm` JS API: `new TabPfnWasm(weightsBytes, configJson)`,
// `fit(xTrain, nTrain, nCols, yTrain, task)`, `predict(xTest, nTest, nCols, task)`.
//
// Usage: `bun run bench_wasm.ts [--reps 10] [--out scripts/bench_wasm.json]`

import init, { TabPfnWasm } from "./pkg/tabpfn_wasm.js";

const OUT_PATH =
  process.argv.indexOf("--out") >= 0
    ? process.argv[process.argv.indexOf("--out") + 1]
    : "scripts/bench_wasm.json";
const REPS =
  process.argv.indexOf("--reps") >= 0
    ? parseInt(process.argv[process.argv.indexOf("--reps") + 1], 10)
    : 10;

type Fixture = {
  config: string;
  model: Uint8Array;
  tasks: Record<
    string,
    {
      x_train: Float32Array;
      y_train: Float32Array;
      x_test: Float32Array;
      n_train: number;
      n_test: number;
      n_cols: number;
      logits_full: Float32Array;
      logits_cached: Float32Array;
    }
  >;
};

// safetensors file: header_len(u32 LE) + header JSON + raw bytes.
// We only need the `__config_json__` string and per-tensor byte slices.
async function readSafetensors(
  path: string
): Promise<Map<string, { dtype: string; shape: number[]; bytes: Uint8Array }>> {
  const raw = new Uint8Array(await fs_readFile(path));
  const headerLen = new DataView(raw.buffer).getUint32(0, true);
  const headerText = new TextDecoder().decode(raw.slice(8, 8 + headerLen));
  const header = JSON.parse(headerText) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const dataStart = 8 + headerLen;
  const out = new Map();
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    out.set(name, {
      dtype: meta.dtype,
      shape: meta.shape,
      bytes: raw.slice(
        dataStart + meta.data_offsets[0],
        dataStart + meta.data_offsets[1]
      ),
    });
  }
  return out;
}

async function fs_readFile(path: string): Promise<ArrayBuffer> {
  const { readFile } = await import("fs/promises");
  const buf = await readFile(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function u8ToF32(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function main() {
  await init();
  const { readFile } = await import("fs/promises");
  const modelBuf = await readFile("scripts/fixtures/model.safetensors");
  const modelBytes = new Uint8Array(
    modelBuf.buffer.slice(modelBuf.byteOffset, modelBuf.byteOffset + modelBuf.byteLength)
  );
  const golden = await readSafetensors("scripts/fixtures/golden.safetensors");

  const configJson = new TextDecoder().decode(
    golden.get("__config_json__")!.bytes
  );

  const tasks: Fixture["tasks"] = {};
  for (const task of ["multiclass", "regression"]) {
    const get = (k: string) => golden.get(`${task}.${k}`)!;
    tasks[task] = {
      x_train: u8ToF32(get("x_train").bytes),
      y_train: u8ToF32(get("y_train").bytes),
      x_test: u8ToF32(get("x_test").bytes),
      n_train: u8ToF32(get("n_train").bytes)[0],
      n_test: u8ToF32(get("n_test").bytes)[0],
      n_cols: u8ToF32(get("n_cols").bytes)[0],
      logits_full: u8ToF32(get("logits_full").bytes),
      logits_cached: u8ToF32(get("logits_cached").bytes),
    };
  }

  const report = { runtime: "wasm", device: "cpu", reps: REPS, tasks: {} as any };

  for (const task of ["multiclass", "regression"]) {
    const t = tasks[task];
    const inst = new TabPfnWasm(modelBytes, configJson);

    // Reproduce the Rust parity path exactly:
    // 1) fit on train rows -> KV cache (inducing_hidden + ICL KV + decoder keys)
    // 2) predict on test rows (test-only slices; feature grouping + ECDF rank +
    //    col-y embedding are keyed on the cached train context)
    {
      const fitTime: number[] = [];
      for (let i = 0; i < REPS; i++) {
        const c = new TabPfnWasm(modelBytes, configJson);
        const t0 = performance.now();
        c.fit(t.x_train, t.n_train, t.n_cols, t.y_train, task);
        fitTime.push(performance.now() - t0);
      }
      void fitTime;
    }

    // Cached-path timing (fit once, time only predict) — matches Rust parity.
    inst.fit(t.x_train, t.n_train, t.n_cols, t.y_train, task);
    const timings: number[] = [];
    let last: Float32Array = new Float32Array(0);
    // warmup
    last = inst.predict(t.x_test, t.n_test, t.n_cols, task);
    for (let i = 0; i < REPS; i++) {
      const t0 = performance.now();
      last = inst.predict(t.x_test, t.n_test, t.n_cols, task);
      timings.push(performance.now() - t0);
    }
    let maxAbs = 0;
    for (let i = 0; i < last.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(last[i] - t.logits_cached[i]));
    }
    report.tasks[task] = {
      timings_ms: timings,
      max_abs_diff_vs_golden: maxAbs,
      logits: Array.from(last),
    };
  }

  const { writeFile } = await import("fs/promises");
  await writeFile(OUT_PATH, JSON.stringify(report, null, 2));
  console.log(`wrote ${OUT_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
