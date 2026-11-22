// Bun tests for the WASM smoke harness + demo.
// Run: `bun test` (discovers *.test.ts)

import { describe, expect, test } from "bun:test";
import init, { TabPfnWasm } from "./pkg/tabpfn_wasm.js";
import { readFileSync } from "fs";

function readSafetensors(path: string) {
  const raw = new Uint8Array(readFileSync(path));
  const headerLen = new DataView(raw.buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(raw.slice(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const out = new Map<string, { bytes: Uint8Array }>();
  for (const [name, meta] of Object.entries(header) as any) {
    if (name === "__metadata__") continue;
    out.set(name, {
      bytes: raw.slice(dataStart + meta.data_offsets[0], dataStart + meta.data_offsets[1]),
    });
  }
  return out;
}

function u8ToF32(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function fixture() {
  await init();
  const golden = readSafetensors("scripts/fixtures/golden.safetensors");
  const configJson = new TextDecoder().decode(golden.get("__config_json__")!.bytes);
  const modelBytes = new Uint8Array(readFileSync("scripts/fixtures/model.safetensors") as any);
  return { golden, configJson, modelBytes };
}

function f32(name: string, gold: Map<string, any>): Float32Array {
  return u8ToF32(gold.get(name)!.bytes);
}

describe("TabPFN WASM (Bun smoke)", () => {
  test("multiclass cached predict within 1e-4 of golden", async () => {
    const { golden, configJson, modelBytes } = await fixture();
    const m = new TabPfnWasm(modelBytes, configJson);
    const xTrain = f32("multiclass.x_train", golden);
    const yTrain = f32("multiclass.y_train", golden);
    const xTest = f32("multiclass.x_test", golden);
    const nTrain = f32("multiclass.n_train", golden)[0];
    const nTest = f32("multiclass.n_test", golden)[0];
    const nCols = f32("multiclass.n_cols", golden)[0];
    const exp = f32("multiclass.logits_cached", golden);
    m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
    const got = m.predict(xTest, nTest, nCols, "multiclass");
    expect(got.length).toBe(exp.length);
    let maxAbs = 0;
    for (let i = 0; i < got.length; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i] - exp[i]));
    expect(maxAbs).toBeLessThan(1e-4);
  });

  test("regression cached predict within 1e-4 of golden", async () => {
    const { golden, configJson, modelBytes } = await fixture();
    const m = new TabPfnWasm(modelBytes, configJson);
    const xTrain = f32("regression.x_train", golden);
    const yTrain = f32("regression.y_train", golden);
    const xTest = f32("regression.x_test", golden);
    const nTrain = f32("regression.n_train", golden)[0];
    const nTest = f32("regression.n_test", golden)[0];
    const nCols = f32("regression.n_cols", golden)[0];
    const exp = f32("regression.logits_cached", golden);
    m.fit(xTrain, nTrain, nCols, yTrain, "regression");
    const got = m.predict(xTest, nTest, nCols, "regression");
    expect(got.length).toBe(exp.length);
    let maxAbs = 0;
    for (let i = 0; i < got.length; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i] - exp[i]));
    expect(maxAbs).toBeLessThan(1e-4);
  });

  test("fusion: fit once, predict twice is stable", async () => {
    const { golden, configJson, modelBytes } = await fixture();
    const m = new TabPfnWasm(modelBytes, configJson);
    const xTrain = f32("multiclass.x_train", golden);
    const yTrain = f32("multiclass.y_train", golden);
    const xTest = f32("multiclass.x_test", golden);
    const nTrain = f32("multiclass.n_train", golden)[0];
    const nTest = f32("multiclass.n_test", golden)[0];
    const nCols = f32("multiclass.n_cols", golden)[0];
    m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
    const a = m.predict(xTest, nTest, nCols, "multiclass");
    const b = m.predict(xTest, nTest, nCols, "multiclass");
    expect(a.length).toBe(b.length);
    let maxAbs = 0;
    for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    expect(maxAbs).toBeLessThan(1e-7);
  });
});
