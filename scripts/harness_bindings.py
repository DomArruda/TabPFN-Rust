"""
Harness: Rust core called through the PyO3 extension (`tabpfn_rs`).

Same contract as harness_pytorch.py, so the analyzer can compare like-for-like.
Also reports pure FFI call overhead (`ping` averaged over 20k calls).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tabpfn_rs  # noqa: E402  (built by scripts/build_pyd.ps1 / cargo build -p tabpfn-py)

FIXTURES = os.path.join(HERE, "fixtures")


def read_st_tensor_bytes(path: str, name: str) -> bytes:
    """Read one tensor's raw bytes from safetensors without importing torch."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        start, end = header[name]["data_offsets"]
        f.seek(8 + header_len + start)
        return f.read(end - start)


def decode(hexstr: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexstr), dtype="<f4").astype(np.float32)


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

    config_json = read_st_tensor_bytes(
        os.path.join(FIXTURES, "golden.safetensors"), "__config_json__"
    ).decode()
    weights = open(os.path.join(FIXTURES, "model.safetensors"), "rb").read()

    x_train = decode(d["x_train"])
    x_test = decode(d["x_test"])
    y_train = decode(d["y_train"])
    x_all = np.concatenate([x_train, x_test])

    m = tabpfn_rs.TabPfn(weights, config_json)

    out = {
        "runtime": "python_bindings",
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

    # Pure FFI overhead: 20k no-op calls.
    n_ping = 20000
    t0 = time.perf_counter()
    for _ in range(n_ping):
        m.ping()
    out["ffi_overhead_us"] = (time.perf_counter() - t0) / n_ping * 1e6

    x_all_b = x_all.tobytes()
    x_test_b = x_test.tobytes()
    y_b = y_train.tobytes()
    task = d["task"]

    timings: list[float] = []
    last_bytes = b""
    if args.mode == "full":
        last_bytes = m.forward(x_all_b, d["n_train"], d["n_test"], d["n_cols"], y_b, task)  # warmup
        for _ in range(args.reps):
            t0 = time.perf_counter()
            last_bytes = m.forward(x_all_b, d["n_train"], d["n_test"], d["n_cols"], y_b, task)
            timings.append((time.perf_counter() - t0) * 1000.0)
    else:
        m.fit(x_train.tobytes(), d["n_train"], d["n_cols"], y_b, task)
        _ = m.predict(x_test_b, d["n_test"], d["n_cols"], task)  # warmup
        for _ in range(args.reps):
            t0 = time.perf_counter()
            last_bytes = m.predict(x_test_b, d["n_test"], d["n_cols"], task)
            timings.append((time.perf_counter() - t0) * 1000.0)

    out["timings_ms"] = timings
    out["logits"] = np.frombuffer(last_bytes, dtype="<f4").tolist()
    with open(args.out, "w") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
