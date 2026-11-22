"""
Inspect TabPFN's internal row representation ("encoder" output).

Runs `TabPFN.encode()` on one dataset, then projects the row embeddings to 2D
with PCA, t-SNE and (if installed) UMAP, colouring points by class. Also
reports a silhouette score for the embeddings vs a raw-feature PCA baseline.

Outputs (scripts/results/):
    embeddings_pca.png, embeddings_tsne.png, embeddings_umap.png, embeddings.json

Usage:
    python scripts/embed_projection.py [--dataset clf_digits_w]

IMPORTANT: the committed fixture has randomly initialised weights, so class
structure in these projections is meaningless. The script demonstrates the
plumbing; use the licensed upstream checkpoint for interpretable embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tabpfn_rs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
OUT = os.path.join(HERE, "results")


def read_st_tensor_bytes(path: str, name: str) -> bytes:
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
    ap.add_argument("--dataset", default="clf_digits_w")
    args = ap.parse_args()

    spec = json.loads(open(os.path.join(HERE, "datasets.json")).read())
    d = next(x for x in spec["datasets"] if x["name"] == args.dataset)
    n_train, n_test, c = d["n_train"], d["n_test"], d["n_cols"]

    x_train = decode(d["x_train"]).reshape(n_train, c)
    x_test = decode(d["x_test"]).reshape(n_test, c)
    y_train = decode(d["y_train"])
    y_test = decode(d["y_test"])
    x_all = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test]).astype(int)

    config_json = read_st_tensor_bytes(
        os.path.join(FIXTURES, "golden.safetensors"), "__config_json__"
    ).decode()
    weights = open(os.path.join(FIXTURES, "model.safetensors"), "rb").read()

    m = tabpfn_rs.TabPfn(weights, config_json)
    emb_bytes = m.encode(
        x_all.tobytes(), n_train, n_test, c, y_train.tobytes(), d["task"]
    )
    emb = np.frombuffer(emb_bytes, dtype="<f4").reshape(n_train + n_test, -1)
    print(f"embedding matrix: {emb.shape[0]} rows x {emb.shape[1]} dims")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    os.makedirs(OUT, exist_ok=True)
    result: dict = {
        "dataset": d["name"],
        "task": d["task"],
        "n_rows": int(emb.shape[0]),
        "embed_dim": int(emb.shape[1]),
        "note": "random-init fixture weights: no semantic class structure expected",
        "methods": {},
    }

    # Silhouette on the *test* rows only. Train-row embeddings include the
    # injected target embedding by design (that is how ICL works), so scoring
    # them measures label leakage, not representation quality.
    n_all = emb.shape[0]
    test_mask = np.arange(n_all) >= n_train
    if len(set(labels[test_mask].tolist())) > 1:
        result["silhouette_test_embeddings"] = float(
            silhouette_score(emb[test_mask], labels[test_mask])
        )
        result["silhouette_test_raw_features"] = float(
            silhouette_score(x_all[test_mask], labels[test_mask])
        )
    if len(set(labels[:n_train].tolist())) > 1:
        result["silhouette_train_embeddings_informational"] = float(
            silhouette_score(emb[:n_train], labels[:n_train])
        )
    result["pca_explained_variance_ratio_2d"] = float(
        PCA(n_components=2, random_state=0).fit(emb).explained_variance_ratio_.sum()
    )

    coords: dict[str, np.ndarray] = {}
    coords["pca"] = PCA(n_components=2, random_state=0).fit_transform(emb)
    pts = min(50, emb.shape[0] - 1)
    perplex = max(5, min(30, pts // 3))
    coords["tsne"] = TSNE(
        n_components=2, perplexity=perplex, init="pca", random_state=0
    ).fit_transform(emb)

    try:
        import umap

        n_neighbors = max(2, min(15, emb.shape[0] - 1))
        coords["umap"] = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, random_state=0
        ).fit_transform(emb)
    except Exception as e:  # noqa: BLE001
        print(f"UMAP skipped: {e}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    for name, xy in coords.items():
        result["methods"][name] = xy.tolist()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        for cls in sorted(set(labels.tolist())):
            mask = labels == cls
            ax.scatter(
                xy[mask, 0], xy[mask, 1], s=18, alpha=0.85,
                color=cmap(cls % 10), label=f"class {cls}",
            )
        ax.set_title(
            f"{d['name']} — TabPFN row embeddings ({emb.shape[1]}-d) via {name.upper()}\n"
            "train rows carry their target embedding; test rows do not",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for cls in sorted(set(labels.tolist())):
            mask = (labels == cls) & test_mask
            if mask.any():
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], s=34, facecolors="none",
                    edgecolors=cmap(cls % 10), linewidths=1.1, label=None,
                )
        ax.legend(fontsize=7, frameon=False, loc="best")
        fig.tight_layout()
        path = os.path.join(OUT, f"embeddings_{name}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")

    with open(os.path.join(OUT, "embeddings.json"), "w") as f:
        json.dump(result, f, indent=1)
    st = result.get("silhouette_test_embeddings")
    sr = result.get("silhouette_test_raw_features")
    if st is not None:
        print(
            f"silhouette on TEST rows: embedding={st:.3f} raw={sr:.3f}\n"
            f"  (train-row silhouette={result.get('silhouette_train_embeddings_informational', float('nan')):.3f} "
            f"— high by construction because train rows carry their target embedding)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
