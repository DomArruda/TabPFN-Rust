"""
Generate variable-scale, real classification + regression datasets for the
4-runtime benchmark. Output: scripts/datasets.json

Datasets are encoded as HEX little-endian f32 bytes (not JSON floats) so that
NaN / +-Inf survive exactly in Rust, JS and Python alike.

Levels: small / medium / large / wide (varying n_train, n_test, n_features),
plus one NaN/Inf/out-of-range stress set. All runtimes consume these same bytes.

Usage: python scripts/gen_datasets.py [--out scripts/datasets.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_regression,
)
from sklearn.model_selection import train_test_split

SEED = 42


def hexf(a: np.ndarray) -> str:
    return np.ascontiguousarray(a, dtype="<f4").tobytes().hex()


def inject_special(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """NaN, +-Inf, ties, and an out-of-range test value."""
    x = x.copy()
    n, c = x.shape
    if n >= 6 and c >= 2:
        x[1, 0] = np.nan
        x[2, 1 % c] = np.inf
        x[3, 0] = -np.inf
        x[4, 2 % c] = x[5, 2 % c]  # tie
    return x


def make(name, task, X, y, n_train, n_test, standardize_y=False, special=False):
    rng = np.random.RandomState(SEED)
    strat = y if task == "multiclass" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=SEED, stratify=strat
    )
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float32)
    yte = np.asarray(yte, dtype=np.float32)
    if standardize_y:
        mu, sd = float(ytr.mean()), float(ytr.std())
        ytr = ((ytr - mu) / (sd if sd > 0 else 1.0)).astype(np.float32)
        yte = ((yte - mu) / (sd if sd > 0 else 1.0)).astype(np.float32)
    if special:
        Xtr = inject_special(Xtr, rng)
        Xte = inject_special(Xte, rng)
        Xte[0, 0] = 1e6
        Xte[1, 1 % Xte.shape[1]] = -1e6
    return {
        "name": name,
        "task": task,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_cols": int(Xtr.shape[1]),
        "x_train": hexf(Xtr),
        "y_train": hexf(ytr),
        "x_test": hexf(Xte),
        "y_test": hexf(yte),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/datasets.json")
    args = ap.parse_args()

    iris = load_iris()
    wine = load_wine()
    breast = load_breast_cancer()
    digits = load_digits()
    diabetes = load_diabetes()

    ds = [
        make("clf_iris_s", "multiclass", iris.data, iris.target, 30, 10),
        make("clf_wine_m", "multiclass", wine.data, wine.target, 100, 25),
        make("clf_breast_l", "multiclass", breast.data, breast.target, 300, 50),
        make("clf_digits_w", "multiclass", digits.data, digits.target, 200, 30),
        make("clf_iris_nan", "multiclass", iris.data, iris.target, 40, 12, special=True),
        make("reg_diabetes_s", "regression", diabetes.data, diabetes.target, 40, 10, standardize_y=True),
        make(
            "reg_synth_m", "regression",
            *make_regression(n_samples=200, n_features=20, noise=0.1, random_state=SEED),
            120, 25, standardize_y=True,
        ),
        make(
            "reg_synth_w", "regression",
            *make_regression(n_samples=400, n_features=50, noise=0.1, random_state=SEED),
            250, 40, standardize_y=True,
        ),
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"seed": SEED, "datasets": ds}, f)
    for d in ds:
        print(f"{d['name']:16s} task={d['task']:11s} train={d['n_train']:3d} test={d['n_test']:3d} cols={d['n_cols']:3d}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
