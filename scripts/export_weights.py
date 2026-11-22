"""Export PyTorch TabPFN-v3.5 weights + golden parity fixtures to safetensors.

Tiny deterministic config so the Rust `cargo test -p tabpfn-core` parity suite
runs in seconds:

    python scripts/export_weights.py

Produces:
    scripts/fixtures/model.safetensors   — state_dict with exact PyTorch key paths
    scripts/fixtures/golden.safetensors   — config JSON + synthetic data + outputs
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

# Make sure the installed tabpfn package is importable
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


def synthetic(rng: np.random.RandomState, n: int, c: int) -> np.ndarray:
    x = rng.randn(n, c).astype(np.float32) * 3.0 + 1.0
    # NaNs, +-Inf, ties, constant-ish column
    if n >= 6:
        x[1, 0] = np.nan
        x[2, 1] = np.inf
        x[3, 0] = -np.inf
        x[4, 2] = x[5, 2]  # tie
        x[:, 3 % c] = 2.0  # constant column (std == 0 -> 1)
    return x


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    cfg = TabPFNV3p5Config(**TINY)
    model = TabPFNV3p5(config=cfg, device="cpu", dtype=torch.float32)
    model.eval()

    # weights with exact PyTorch key paths
    from safetensors.torch import save_file

    save_file(model.state_dict(), os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    tensors: dict[str, torch.Tensor] = {
        "__config_json__": torch.tensor(list(json.dumps(cfg.__dict__).encode()), dtype=torch.uint8),
    }

    for task, y_fn in [
        ("multiclass", lambda r: r.randint(0, 5, size=12).astype(np.float32)),
        ("regression", lambda r: (r.randn(12) * 2.0 + 0.5).astype(np.float32)),
    ]:
        n_train, n_test, c = 12, 5, 5
        x_train = synthetic(rng, n_train, c)
        x_test = synthetic(rng, n_test, c)
        # out-of-range test value
        x_test[0, 0] = 1e6
        x_test[1, 1] = -1e6
        y_train = y_fn(rng)
        y_train[3] = np.nan  # missing target

        x_all = np.concatenate([x_train, x_test], axis=0)
        # model expects (R, B, C)
        x_t = torch.from_numpy(x_all).unsqueeze(1)
        y_t = torch.from_numpy(y_train)

        with torch.no_grad():
            full = model.forward(x_t, y_t, task, only_return_standard_out=True)
            # cached path via return_kv_cache
            out_cached, cache = model.forward(
                x_t, y_t, task, only_return_standard_out=True,
                return_kv_cache=True,
                x_is_test_only=False,
            )
            # test-only predict from cache
            x_test_t = torch.from_numpy(x_test).unsqueeze(1)
            cached_pred = model.forward(
                x_test_t, y_t, task, only_return_standard_out=True,
                kv_cache=cache, x_is_test_only=True,
            )

        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().reshape(-1).contiguous()

        tensors[f"{task}.x_train"] = torch.from_numpy(x_train).reshape(-1)
        tensors[f"{task}.y_train"] = torch.from_numpy(y_train).reshape(-1)
        tensors[f"{task}.x_test"] = torch.from_numpy(x_test).reshape(-1)
        tensors[f"{task}.n_train"] = torch.tensor([n_train], dtype=torch.float32)
        tensors[f"{task}.n_test"] = torch.tensor([n_test], dtype=torch.float32)
        tensors[f"{task}.n_cols"] = torch.tensor([c], dtype=torch.float32)
        tensors[f"{task}.logits_full"] = flat(full)
        tensors[f"{task}.logits_cached"] = flat(cached_pred)

    save_file(tensors, os.path.join(out_dir, "golden.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
        f.write("\n")
    print(f"wrote {out_dir}/model.safetensors and golden.safetensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
