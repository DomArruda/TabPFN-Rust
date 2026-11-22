"""
Orchestrator: runs all four runtimes on every dataset, in 3 memory modes,
and records wall-clock timings + peak RSS for each (runtime, dataset, mode).

Runtimes
  python_pytorch  : original Python / PyTorch TabPFN v3.5 (CPU)
  python_bindings : Rust core via the PyO3 extension
  rust_native     : Rust / Candle, native binary
  wasm_bun        : Rust / Candle compiled to WASM SIMD128, run under Bun

Modes
  baseline : load model + decode data, no inference   (RSS floor)
  full     : single-pass forward over train+test rows
  cached   : fit() once, then time test-only predict()

Peak RSS is sampled externally (psutil) so all four runtimes are measured the
same way. Raw results -> scripts/results/raw.json

Usage:
  python scripts/run_benchmarks.py [--reps 10] [--only clf_iris_s] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "scripts" / "results"
PER_RUN = RES / "per_run"


def commands(runtime: str, dataset: str, mode: str, reps: int, out: Path):
    py = sys.executable
    ds = str(ROOT / "scripts" / "datasets.json")
    if runtime == "python_pytorch":
        return [py, str(ROOT / "scripts" / "harness_pytorch.py"),
                "--datasets", ds, "--dataset", dataset,
                "--reps", str(reps), "--mode", mode, "--out", str(out)]
    if runtime == "python_bindings":
        return [py, str(ROOT / "scripts" / "harness_bindings.py"),
                "--datasets", ds, "--dataset", dataset,
                "--reps", str(reps), "--mode", mode, "--out", str(out)]
    if runtime == "rust_native":
        exe = ROOT / "target" / "release" / "examples" / "harness_datasets.exe"
        if not exe.exists():
            exe = ROOT / "target" / "release" / "examples" / "harness_datasets"
        return [str(exe), "--datasets", ds, "--dataset", dataset,
                "--reps", str(reps), "--mode", mode, "--out", str(out)]
    if runtime == "wasm_bun":
        return ["bun", "run", str(ROOT / "scripts" / "harness_wasm.ts"),
                "--datasets", ds, "--dataset", dataset,
                "--reps", str(reps), "--mode", mode, "--out", str(out)]
    raise ValueError(runtime)


def run_one(runtime: str, dataset: str, mode: str, reps: int) -> dict:
    out = PER_RUN / f"{runtime}__{dataset}__{mode}.json"
    cmd = commands(runtime, dataset, mode, reps, out)
    out.unlink(missing_ok=True)
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    peak = 0
    ps = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.002)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t_start
    if proc.returncode != 0:
        return {
            "runtime": runtime, "dataset": dataset, "mode": mode,
            "error": (stderr or stdout or b"").decode("utf-8", "replace")[-4000:],
            "peak_rss_mb": peak / 1e6, "wall_s": wall,
        }
    rec = json.loads(out.read_text())
    rec["peak_rss_mb"] = peak / 1e6
    rec["wall_s"] = wall
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--only", default=None, help="single dataset name")
    ap.add_argument("--quick", action="store_true", help="skip baseline mode")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    PER_RUN.mkdir(parents=True, exist_ok=True)

    spec = json.loads((ROOT / "scripts" / "datasets.json").read_text())
    datasets = [d["name"] for d in spec["datasets"]]
    if args.only:
        datasets = [d for d in datasets if d == args.only]

    runtimes = ["python_pytorch", "python_bindings", "rust_native", "wasm_bun"]
    modes = ["full", "cached"] + ([] if args.quick else ["baseline"])

    records: list[dict] = []
    total = len(runtimes) * len(datasets) * len(modes)
    i = 0
    for runtime in runtimes:
        for dataset in datasets:
            for mode in modes:
                i += 1
                print(f"[{i:3d}/{total}] {runtime:16s} {dataset:16s} {mode:8s}", flush=True)
                rec = run_one(runtime, dataset, mode, args.reps)
                if "error" in rec:
                    print(f"    FAILED: {rec['error'].splitlines()[-1] if rec['error'] else '?'}")
                else:
                    t = rec.get("timings_ms") or []
                    mean = sum(t) / len(t) if t else 0.0
                    print(f"    rss={rec['peak_rss_mb']:7.1f}MB  mean={mean:8.3f}ms  n={len(t)}")
                records.append(rec)

    (RES / "raw.json").write_text(json.dumps(records, indent=1))
    fails = [r for r in records if "error" in r]
    print(f"\nwrote {RES/'raw.json'}  ({len(records)} records, {len(fails)} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
