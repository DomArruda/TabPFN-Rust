"""
Build the WASM binding and turn `pkg/` into a publishable npm package.

    python scripts/build_wasm_pkg.py
    npm publish ./pkg --access public

Environment overrides:
    NPM_NAME      package name (default "tabpfn-wasm")
    NPM_VERSION   package version (default: version from crates/tabpfn-wasm/Cargo.toml)

The output is the `web` target: an ES module plus a `.wasm` file and generated
TypeScript declarations. Consumers call `await init()` once before using
`TabPfnWasm` (see docs/PUBLISH.md).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "pkg"
WASM = ROOT / "target" / "wasm32-unknown-unknown" / "release" / "tabpfn_wasm.wasm"


def cargo_version() -> str:
    text = (ROOT / "crates" / "tabpfn-wasm" / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "0.1.0"


def bindgen_version() -> str:
    out = subprocess.run(
        ["wasm-bindgen", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out.split()[-1]


def main() -> int:
    name = os.environ.get("NPM_NAME", "tabpfn-wasm")
    version = os.environ.get("NPM_VERSION", cargo_version())

    # wasm-bindgen's schema version must match the crate exactly.
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    m = re.search(r'name = "wasm-bindgen"\nversion = "([^"]+)"', lock)
    crate_wb = m.group(1) if m else None
    cli_wb = bindgen_version()
    if crate_wb and cli_wb != crate_wb:
        print(
            f"warning: wasm-bindgen crate {crate_wb} != CLI {cli_wb}\n"
            f"  cargo install -f wasm-bindgen-cli --version {crate_wb}",
            file=sys.stderr,
        )

    env = os.environ.copy()
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -C target-feature=+simd128").strip()

    print("building wasm (simd128, release)")
    subprocess.run(
        ["cargo", "build", "-p", "tabpfn-wasm", "--target", "wasm32-unknown-unknown", "--release"],
        cwd=ROOT, env=env, check=True,
    )

    print("running wasm-bindgen (target=web, with types)")
    subprocess.run(
        [
            "wasm-bindgen", str(WASM),
            "--out-dir", str(PKG),
            "--target", "web",
        ],
        cwd=ROOT, check=True,
    )

    manifest = {
        "name": name,
        "version": version,
        "description": "TabPFN v3.5 inference in WebAssembly. Unofficial Rust port.",
        "license": "Apache-2.0",
        "type": "module",
        "main": "tabpfn_wasm.js",
        "module": "tabpfn_wasm.js",
        "types": "tabpfn_wasm.d.ts",
        "exports": {
            ".": {
                "types": "./tabpfn_wasm.d.ts",
                "import": "./tabpfn_wasm.js",
                "default": "./tabpfn_wasm.js",
            },
            "./tabpfn_wasm_bg.wasm": "./tabpfn_wasm_bg.wasm",
        },
        "files": [
            "tabpfn_wasm.js",
            "tabpfn_wasm.d.ts",
            "tabpfn_wasm_bg.wasm",
            "tabpfn_wasm_bg.wasm.d.ts",
            "README.md",
        ],
        "sideEffects": False,
        "keywords": ["tabpfn", "wasm", "webassembly", "tabular", "machine-learning"],
    }
    (PKG / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = (
        f"# {name}\n\n"
        "TabPFN v3.5 inference compiled to WebAssembly (SIMD128). Unofficial Rust port.\n\n"
        "```ts\n"
        f'import init, {{ TabPfnWasm }} from "{name}";\n\n'
        "await init();\n"
        "const m = new TabPfnWasm(weightsBytes, configJson);\n"
        'm.fit(xTrain, nTrain, nCols, yTrain, "multiclass");\n'
        'const logits = m.predict(xTest, nTest, nCols, "multiclass");\n'
        "```\n\n"
        "`init()` must be awaited once before constructing `TabPfnWasm`. The module\n"
        "loads `tabpfn_wasm_bg.wasm` from the same directory.\n\n"
        "Model weights are not included. Architecture and weight-key layout follow\n"
        "Prior Labs' TabPFN (Apache-2.0); weights are licensed separately.\n"
    )
    (PKG / "README.md").write_text(readme, encoding="utf-8")

    license_src = ROOT / "LICENSE"
    if license_src.exists():
        shutil.copyfile(license_src, PKG / "LICENSE")

    print(f"\npackage: {PKG}")
    for f in sorted(PKG.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:8.1f} KB")
    print(f"\nverify:  npm pack --dry-run ./pkg")
    print(f"publish: npm publish ./pkg --access public")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
