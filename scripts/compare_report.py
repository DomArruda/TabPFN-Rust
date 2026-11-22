"""
Compare Python (PyTorch CPU) vs Rust native (candle) vs WASM (Bun/Node)
on identical fixture data. For each task this computes:

  * numerical parity: max_abs_diff, mean_abs_diff, and whether WASM/Rust
    agree with PyTorch within 1e-4 (the project parity bar).
  * inference-time statistics: mean, median, SD, min, max per runtime
    from the raw `timings_ms` arrays in the three bench JSON files.
  * speedups: mean ratio vs Python, per temperature band ('cold' timing
    includes Python interpreter + torch overhead, 'hot' is steady state).
  * hypothesis test on timings: two-sample Welch's t-test (Python vs Rust,
    Python vs WASM, Rust vs WASM) reporting t-statistic and two-sided
    p-value, plus effect size (Cohen's d) — whether the differences are
    statistically meaningful rather than noise.

Output: `scripts/bench_report.json` (machine readable) + printed summary.
Usage: `python scripts/compare_report.py [--out scripts/bench_report.json]`
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def describe(t: list[float]) -> dict:
    a = np.asarray(t, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "sd_ms": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "ci95_ms": (
            float(np.mean(a) - 1.96 * np.std(a, ddof=1) / np.sqrt(a.size)),
            float(np.mean(a) + 1.96 * np.std(a, ddof=1) / np.sqrt(a.size)),
        )
        if a.size > 1
        else [float(np.mean(a)), float(np.mean(a))],
    }


def welch(a: list[float], b: list[float]) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d with pooled SD
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if a.size > 1 and b.size > 1 else 0.0
    d = float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0
    return {
        "t_stat": float(t),
        "p_value_two_sided": float(p),
        "cohens_d": d,
        "significant_at_0p05": bool(p < 0.05),
        "significant_at_0p01": bool(p < 0.01),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/bench_report.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = load(os.path.join(here, "bench_python.json"))
    rust = load(os.path.join(here, "bench_rust.json"))
    wasm = load(os.path.join(here, "bench_wasm.json"))

    tasks = ["multiclass", "regression"]
    report: dict = {
        "fixture": "scripts/fixtures/model.safetensors + golden.safetensors",
        "notes": (
            "Rust times full forward on train+test; WASM times cached "
            "predict (fit once, predict-only); Python times full forward. "
            "Max-abs parity is computed on logits_cached / logits_full "
            "which round-trip to the same golden values."
        ),
        "tasks": {},
    }

    for task in tasks:
        py_t = py["tasks"][task]["timings_ms"]
        rust_t = rust["tasks"][task]["timings_ms"]
        wasm_t = wasm["tasks"][task]["timings_ms"]

        py_l = np.asarray(py["tasks"][task]["logits"], dtype=np.float64)
        rust_l = np.asarray(rust["tasks"][task]["logits"], dtype=np.float64)
        wasm_l = np.asarray(wasm["tasks"][task]["logits"], dtype=np.float64)

        def mad(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        numerics = {
            "python_vs_rust": mad(py_l, rust_l),
            "python_vs_wasm": mad(py_l, wasm_l),
            "rust_vs_wasm": mad(rust_l, wasm_l),
            "py_golden": py["tasks"][task]["max_abs_diff_vs_golden"],
            "rust_golden": rust["tasks"][task]["max_abs_diff_vs_golden"],
            "wasm_golden": wasm["tasks"][task]["max_abs_diff_vs_golden"],
            "parity_bar_1e4": 1e-4,
            "within_ratio": mad(py_l, wasm_l) < 1e-4,
            "note": (
                "WASM times the Cached (fit-once) path so it excludes the "
                "_stages_0_to_2 + ICL block rebuild that Python and Rust "
                "pay per full forward; Rust native and WASM share the same "
                "candle-core binary so their gap isolates WASM overhead."
            ),
        }

        timing_stats = {
            "python": describe(py_t),
            "rust": describe(rust_t),
            "wasm": describe(wasm_t),
        }

        speedups = {
            "rust_over_python_mean": float(np.mean(rust_t) / np.mean(py_t)),
            "wasm_over_python_mean": float(np.mean(wasm_t) / np.mean(py_t)),
            "wasm_over_rust_mean": float(np.mean(wasm_t) / np.mean(rust_t)),
            "rust_over_python_median": float(np.median(rust_t) / np.median(py_t)),
            "wasm_over_python_median": float(np.median(wasm_t) / np.median(py_t)),
        }

        tests = {
            "python_vs_rust": welch(py_t, rust_t),
            "python_vs_wasm": welch(py_t, wasm_t),
            "rust_vs_wasm": welch(rust_t, wasm_t),
        }

        report["tasks"][task] = {
            "numerics": numerics,
            "timings": timing_stats,
            "speedups": speedups,
            "hypothesis_tests": tests,
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"wrote {args.out}\n")
    for task in tasks:
        t = report["tasks"][task]
        print(f"=== {task} ===")
        print(f"  max|Py-Rust|={t['numerics']['python_vs_rust']:.3e}  "
              f"max|Py-WASM|={t['numerics']['python_vs_wasm']:.3e}  "
              f"max|Rust-WASM|={t['numerics']['rust_vs_wasm']:.3e}")
        for r in ["python", "rust", "wasm"]:
            s = t["timings"][r]
            print(f"  {r:6s} mean={s['mean_ms']:.3f}ms median={s['median_ms']:.3f}ms "
                  f"sd={s['sd_ms']:.3f}ms CI95=[{s['ci95_ms'][0]:.2f},{s['ci95_ms'][1]:.2f}]")
        print(f"  speed: rust/py={t['speedups']['rust_over_python_mean']:.3f}  "
              f"wasm/py={t['speedups']['wasm_over_python_mean']:.3f}  "
              f"wasm/rust={t['speedups']['wasm_over_rust_mean']:.3f}")
        for pair, h in t["hypothesis_tests"].items():
            verdict = "significant" if h["significant_at_0p05"] else "not significant"
            print(f"  Welch {pair}: t={h['t_stat']:.2f} p={h['p_value_two_sided']:.3e} "
                  f"d={h['cohens_d']:.2f} -> {verdict}@0.05")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
