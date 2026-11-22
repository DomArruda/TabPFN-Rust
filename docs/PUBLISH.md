# Publishing the WASM package to npm

The Rust workspace is not published. Only the generated `pkg/` directory is,
as a small npm package containing the ES module, the `.wasm` binary and
TypeScript declarations.

## 1. Prerequisites

```bash
rustup target add wasm32-unknown-unknown
```

`wasm-bindgen-cli` must be the **same version** as the `wasm-bindgen` crate in
`Cargo.lock`, otherwise the generated glue is rejected:

```bash
# check both
cargo tree -p tabpfn-wasm -i wasm-bindgen | Select-Object -First 2
wasm-bindgen --version

# install the matching CLI (0.2.130 at time of writing)
cargo install -f wasm-bindgen-cli --version 0.2.130
```

You also need an npm account (`npm login`) or a registry token. Bun can publish
too (`bun publish`).

## 2. Pick a package name

The repo uses the unscoped name `tabpfn-wasm`, which was free when this was
written. Unscoped names are global and first-come-first-served, so check before
publishing. If it is taken, switch to a scope you own (`@your-user/tabpfn-wasm`
or an org) by setting `NPM_NAME`.

```bash
npm view tabpfn-wasm version   # 404 means the name is free
```

## 3. Build the package

```bash
# defaults: name tabpfn-wasm, version from crates/tabpfn-wasm/Cargo.toml
python scripts/build_wasm_pkg.py

# or set them explicitly
$env:NPM_NAME="tabpfn-wasm"
$env:NPM_VERSION="0.1.0"
python scripts/build_wasm_pkg.py
```

This compiles for `wasm32-unknown-unknown` with `+simd128`, runs
`wasm-bindgen --target web`, and writes `pkg/package.json`, `pkg/README.md` and
(if present) `pkg/LICENSE`.

The output is the **web** target: an ES module that loads
`tabpfn_wasm_bg.wasm` from the same directory.

## 4. Inspect before publishing

```bash
npm pack --dry-run ./pkg
```

You should see exactly six files and a compressed size around 400 kB:

```
README.md, package.json, tabpfn_wasm.js, tabpfn_wasm.d.ts,
tabpfn_wasm_bg.wasm, tabpfn_wasm_bg.wasm.d.ts
```

Optionally install the tarball in a scratch project to prove it resolves:

```bash
npm pack ./pkg --pack-destination $env:TEMP
mkdir scratch; cd scratch
bun init -y
bun add $env:TEMP\tabpfn-wasm-0.1.0.tgz
```

## 5. Publish

```bash
npm publish ./pkg --access public
# or
bun publish ./pkg --access public
```

`--access public` is the default for unscoped packages and required for scoped
ones. Publishing is permanent: bump the version for every release.

## 6. Consume

### Bun

```bash
bun add tabpfn-wasm
```

```ts
import init, { TabPfnWasm } from "tabpfn-wasm";

await init();                       // must run once before constructing the model

const weights = new Uint8Array(await Bun.file("model.safetensors").arrayBuffer());
const m = new TabPfnWasm(weights, configJson);
m.fit(xTrain, nTrain, nCols, yTrain, "multiclass");
const logits = m.predict(xTest, nTest, nCols, "multiclass");
```

### Browser

Same imports. The module fetches `tabpfn_wasm_bg.wasm` relative to itself, so
serve the package directory over HTTP (bundlers do this for you).

### Node

The `web` target uses `fetch`, which Node rejects for `file:` URLs. Pass the
bytes instead:

```js
import init, { TabPfnWasm } from "tabpfn-wasm";
import { readFileSync } from "node:fs";

await init({ module_or_path: readFileSync("node_modules/tabpfn-wasm/tabpfn_wasm_bg.wasm") });
```

Alternatively build a second package with `wasm-bindgen --target nodejs` and
publish it under a different name or export condition.

## 7. Versioning

Keep `crates/tabpfn-wasm/Cargo.toml` and the npm version in step, and record
what changed in the package README. The `.wasm` binary is pinned to the Rust
code at build time, so rebuilding after a model change always produces a new
artifact.

## Licensing note

The Rust code and bindings are Apache-2.0. Model weights are **not** included
and are licensed separately by Prior Labs (non-commercial for the v3.5
checkpoints). Keep the "unofficial" wording in the package description.
