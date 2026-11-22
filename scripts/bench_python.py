/// Times the Python (PyTorch CPU) reference and writes raw timings +
/// logits to `scripts/bench_python.json` for the cross-runtime report.
/// Usage: `python scripts/bench_python.py [--reps 10] [--out path.json]`
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5, TabPFNV3p5Config

SEED = 42
TINY = dict(
    embed_dim=32,
    dist_embed_num_heads=4,
    dist_embed_num_inducing_points=16,
    feat_agg_num_heads=4,
    feat_agg_num_cls_tokens=4,
    nlayers=2,
    icl_num_heads=4,
    icl_num_kv_heads=4,
    icl_num_kv_heads_test=1,
    decoder_num_heads=2,
    decoder_head_dim=16,
    cell_ecdf_num_buckets=8,
    max_num_classes=10,
    num_buckets=50,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="scripts/bench_python.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model = TabPFNV3p5(config=TabPFNV3p5Config(**TINY), device="cpu", dtype=torch.float32)
    model.load_state_dict(load_file(os.path.join(here, "fixtures", "model.safetensors")))
    model.eval()
    golden = load_file(os.path.join(here, "fixtures", "golden.safetensors"))

    report = {"runtime": "python_pytorch", "device": "cpu", "reps": args.reps, "tasks": {}}
    for task in ["multiclass", "regression"]:
        n_train = int(golden[f"{task}.n_train"][0])
        n_test = int(golden[f"{task}.n_test"][0])
        c = int(golden[f"{task}.n_cols"][0])
        x_train = golden[f"{task}.x_train"].numpy()
        x_test = golden[f"{task}.x_test"].numpy()
        y_train = golden[f"{task}.y_train"].numpy()
        # x_train/x_test are flat; reshape to (rows, cols)
        x_train2 = x_train.reshape(n_train, c)
        x_test2 = x_test.reshape(n_test, c)
        x_all = np.concatenate([x_train2, x_test2], axis=0)
        x_t = torch.from_numpy(x_all).unsqueeze(1)
        y_t = torch.from_numpy(y_train.astype(np.float32))

        with torch.no_grad():
            _ = model.forward(x_t, y_t, task, only_return_standard_out=True)  # warmup
            timings = []
            last = None
            for _ in range(args.reps):
                t0 = time.perf_counter()
                last = model.forward(x_t, y_t, task, only_return_standard_out=True)
                timings.append((time.perf_counter() - t0) * 1000.0)

        exp = golden[f"{task}.logits_full"].numpy()
        got = last.flatten().numpy()
        report["tasks"][task] = {
            "timings_ms": timings,
            "max_abs_diff_vs_golden": float(np.max(np.abs(got - exp))),
            "logits": got.tolist(),
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
