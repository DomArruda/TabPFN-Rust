"""
Reference harness: original Python / PyTorch TabPFN v3.5 (CPU).

Contract (shared by all four runtimes):
  python scripts/harness_pytorch.py --dataset clf_iris_s --reps 10 \
      --mode full|cached|baseline --out /tmp/x.json

Modes
  baseline : load weights + decode dataset, then exit  (peak-RSS floor)
  full     : time a single-pass forward over train+test rows
  cached   : fit once (return_kv_cache) then time test-only predict

Writes timings_ms + logits so the analyzer can do parity/equivalence tests.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5, TabPFNV3p5Config

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def decode(hexstr: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexstr), dtype="<f4").astype(np.float32)


def load_model() -> TabPFNV3p5:
    golden = load_file(os.path.join(FIXTURES, "golden.safetensors"))
    raw = golden["__config_json__"].numpy().tobytes().decode()
    cfg = TabPFNV3p5Config(**json.loads(raw))
    model = TabPFNV3p5(config=cfg, device="cpu", dtype=torch.float32)
    model.load_state_dict(load_file(os.path.join(FIXTURES, "model.safetensors")))
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="scripts/datasets.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mode", choices=["baseline", "full", "cached"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.datasets) as f:
        spec = json.load(f)
    d = next(x for x in spec["datasets"] if x["name"] == args.dataset)

    model = load_model()

    x_train = decode(d["x_train"]).reshape(d["n_train"], d["n_cols"])
    x_test = decode(d["x_test"]).reshape(d["n_test"], d["n_cols"])
    y_train = decode(d["y_train"])
    x_all = np.concatenate([x_train, x_test], axis=0)

    out = {
        "runtime": "python_pytorch",
        "dataset": d["name"],
        "task": d["task"],
        "mode": args.mode,
        "reps": args.reps if args.mode != "baseline" else 0,
        "n_train": d["n_train"],
        "n_test": d["n_test"],
        "n_cols": d["n_cols"],
    }
    if args.mode == "baseline":
        out["timings_ms"] = []
        out["logits"] = []
        with open(args.out, "w") as f:
            json.dump(out, f)
        return 0

    xt_all = torch.from_numpy(x_all).unsqueeze(1)          # (R, 1, C)
    xt_test = torch.from_numpy(x_test).unsqueeze(1)
    yt = torch.from_numpy(y_train)

    timings: list[float] = []
    last = None
    with torch.no_grad():
        if args.mode == "full":
            _ = model.forward(xt_all, yt, d["task"], only_return_standard_out=True)  # warmup
            for _ in range(args.reps):
                t0 = time.perf_counter()
                last = model.forward(xt_all, yt, d["task"], only_return_standard_out=True)
                timings.append((time.perf_counter() - t0) * 1000.0)
        else:
            _, cache = model.forward(
                xt_all, yt, d["task"], only_return_standard_out=True,
                return_kv_cache=True, x_is_test_only=False,
            )
            _ = model.forward(
                xt_test, yt, d["task"], only_return_standard_out=True,
                kv_cache=cache, x_is_test_only=True,
            )  # warmup
            for _ in range(args.reps):
                t0 = time.perf_counter()
                last = model.forward(
                    xt_test, yt, d["task"], only_return_standard_out=True,
                    kv_cache=cache, x_is_test_only=True,
                )
                timings.append((time.perf_counter() - t0) * 1000.0)

    out["timings_ms"] = timings
    out["logits"] = last.detach().cpu().flatten().tolist()
    with open(args.out, "w") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
