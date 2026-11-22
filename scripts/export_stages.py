"""Bisect Rust vs PyTorch stage by stage using intermediate golden tensors."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

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


def flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().reshape(-1).contiguous()


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "fixtures")
    g = load_file(os.path.join(out_dir, "golden.safetensors"))
    sd = load_file(os.path.join(out_dir, "model.safetensors"))
    cfg = TabPFNV3p5Config(**TINY)
    model = TabPFNV3p5(config=cfg, device="cpu", dtype=torch.float32)
    model.load_state_dict(sd)
    model.eval()

    tensors: dict[str, torch.Tensor] = {
        "__config_json__": g["__config_json__"],
    }
    for task in ["multiclass", "regression"]:
        x_train = g[f"{task}.x_train"].numpy()
        y_train = g[f"{task}.y_train"].numpy()
        # reconstruct n/c from sizes: n_rows = len/n_cols; infer from stored scalars
        n_train = int(g[f"{task}.n_train"][0].item())
        n_test = int(g[f"{task}.n_test"][0].item())
        c = int(g[f"{task}.n_cols"][0].item())
        x_train2 = x_train.reshape(n_train, c)
        x_test2 = g[f"{task}.x_test"].numpy().reshape(n_test, c)
        x_all = np.concatenate([x_train2, x_test2], axis=0)
        x_t = torch.from_numpy(x_all).unsqueeze(1)
        y_t = torch.from_numpy(y_train)

        with torch.no_grad():
            # instrument stages 0-2 manually
            x_RiBC = x_t
            num_train = int(y_t.shape[0])
            # replicate _stages_0_to_2 internals
            from tabpfn.architectures.tabpfn_v3_5 import _ECDF_CONTEXT_KEY

            x_BRiC_nan, nan_ind, ecdf, _y_col_emb, scaler_stats = model._preprocess_no_group(
                x_RiBC, y_t, num_train, None, task
            )
            x_grouped = model._group_features(x_BRiC_nan, nan_ind, ecdf)
            x_emb = model.x_embed(x_grouped)
            # add col y
            from tabpfn.architectures.tabpfn_v3_5 import _prepare_targets, _impute_target_nan_and_inf

            # y_col_emb path
            y_col_BN = model._prepare_y(y_t, num_train, 1, task_type=task)
            y_col_emb = model._embed_col_y(y_col_BN, task_type=task)
            x_emb = x_emb.clone()
            x_emb[:, :num_train] = x_emb[:, :num_train] + y_col_emb.unsqueeze(2)
            x_dist, _ = model.feature_distribution_embedder(
                x_BRiCE=x_emb, num_train_rows=num_train
            )
            x_agg = model.column_aggregator(x_BRiCE=x_dist)
            x_icl_in = x_agg.flatten(-2)
            # ICL embedding + blocks
            y_icl = model._prepare_y(y_t, num_train, 1, task_type=task)
            y_icl_emb = model._embed_icl_y(y_icl, task_type=task)
            x_icl_in = x_icl_in.clone()
            x_icl_in[:, :num_train] = x_icl_in[:, :num_train] + y_icl_emb
            x_icl = x_icl_in
            for block in model.icl_blocks:
                x_icl, _ = block(x_icl, num_train)
            x_icl_n = model.output_norm(x_icl)
            test_emb = x_icl_n[:, num_train:]
            train_emb = x_icl_n[:, :num_train]
            if task == "multiclass":
                keys = model.heads.project_decoder_keys(train_emb)
                test_out = model.heads(keys, test_emb, y_t.unsqueeze(0)[:, :num_train], task_type=task, num_present_classes=int(torch.nan_to_num(y_t, nan=0.0).max().item()) + 1)
            else:
                test_out = model.heads(None, test_emb, y_t.unsqueeze(0)[:, :num_train], task_type=task, num_present_classes=None)

        prefix = f"{task}.stage"
        tensors[f"{prefix}.x_grouped"] = flat(x_grouped)
        tensors[f"{prefix}.x_embed"] = flat(x_emb)
        tensors[f"{prefix}.x_dist"] = flat(x_dist)
        tensors[f"{prefix}.x_agg"] = flat(x_agg)
        tensors[f"{prefix}.x_icl_in"] = flat(x_icl_in)
        tensors[f"{prefix}.x_icl_n_test"] = flat(test_emb)
        tensors[f"{prefix}.logits"] = flat(torch.nan_to_num(test_out, nan=0.0))

    save_file(tensors, os.path.join(out_dir, "stages.safetensors"), metadata={"format": "pt"})
    print("wrote stages.safetensors with keys:", sorted(tensors.keys())[:6], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
