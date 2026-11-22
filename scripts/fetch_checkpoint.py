"""
Download the upstream TabPFN v3.5 checkpoint and extract its architecture config.

This repo does not redistribute weights. The TabPFN-3.5 weights are licensed
separately by Prior Labs and are non-commercial. Downloading them means you
accept that license: https://huggingface.co/Prior-Labs/tabpfn_3_5

The checkpoint is a safetensors file whose header metadata carries the
architecture config as a JSON string. This script pulls the file, reads that
metadata (no torch needed), writes `config.json` next to it, and checks the
keys against what the Rust port expects.

Usage:
    python scripts/fetch_checkpoint.py
    python scripts/fetch_checkpoint.py --variant fast --out models
    python scripts/fetch_checkpoint.py --file path/to/checkpoint.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "Prior-Labs/tabpfn_3_5"
VARIANTS = {
    "default": "tabpfn-v3.5-20260909.safetensors",
    "multiclass": "tabpfn-v3.5-20260909_multiclass.safetensors",
    "fast": "tabpfn-v3.5-fast-20260909.safetensors",
}

# Fields the Rust `TabPFNV3p5Config` deserializer requires.
REQUIRED = [
    "name", "embed_dim",
    "dist_embed_num_blocks", "dist_embed_num_heads", "dist_embed_num_inducing_points",
    "feature_group_size",
    "feat_agg_num_blocks", "feat_agg_num_heads", "feat_agg_num_cls_tokens", "feat_agg_rope_base",
    "nlayers", "icl_num_heads", "icl_num_kv_heads", "icl_num_kv_heads_test",
    "decoder_head_dim", "decoder_num_heads", "decoder_use_softmax_scaling",
    "ff_factor", "softmax_scaling_mlp_hidden_dim",
    "fourier_encoding_num_frequencies", "cell_ecdf_num_frequencies",
    "cell_ecdf_num_buckets", "cell_embed_row_chunk_size",
    "inference_chunk_cells", "max_batched_estimator_rows",
    "max_batched_estimator_cells", "inference_col_chunk_size",
    "max_num_classes", "num_buckets",
]


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    """Return the `__metadata__` header of a safetensors file."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header.get("__metadata__", {})


def extract_config(path: Path) -> tuple[dict, str | None]:
    """Pull the architecture config and architecture name out of the header."""
    meta = read_safetensors_metadata(path)
    if "config" not in meta:
        raise SystemExit(
            f"{path.name} has no 'config' entry in its safetensors metadata; "
            "this does not look like a TabPFN checkpoint."
        )
    config = json.loads(meta["config"])
    arch = meta.get("architecture_name")
    if arch is not None:
        arch = json.loads(arch)
    return config, arch


def download(filename: str, out_dir: Path) -> Path:
    dest = out_dir / filename
    if dest.exists():
        print(f"already present: {dest}")
        return dest
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp.replace(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="default")
    ap.add_argument("--file", default=None, help="use a local checkpoint instead of downloading")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        ckpt = Path(args.file)
        if not ckpt.exists():
            raise SystemExit(f"not found: {ckpt}")
    else:
        ckpt = download(VARIANTS[args.variant], out_dir)

    config, arch = extract_config(ckpt)
    missing = [k for k in REQUIRED if k not in config]

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"\ncheckpoint : {ckpt}  ({ckpt.stat().st_size / 1e6:.1f} MB)")
    print(f"config     : {config_path}")
    print(f"architecture: {arch}")
    print(
        "shape      : embed_dim={embed_dim}, nlayers={nlayers}, "
        "max_num_classes={max_num_classes}, num_buckets={num_buckets}".format(**config)
    )
    if arch is not None and "v3_5" not in str(arch):
        print(f"warning: architecture is {arch!r}, this port implements v3.5", file=sys.stderr)
    if missing:
        print(f"warning: config is missing fields the Rust port needs: {missing}", file=sys.stderr)
        return 1

    print("\nNext:")
    print(f'  $env:TABPFN_MODEL  = "{ckpt.as_posix()}"')
    print(f'  $env:TABPFN_CONFIG = "{config_path.as_posix()}"')
    print("  see examples/bun and examples/node")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
