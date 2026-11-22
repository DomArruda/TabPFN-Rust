# Examples

Three ways to call the published package, `tabpfn-wasm`:

| directory | runtime | run |
|---|---|---|
| `bun/` | Bun | `cd bun && bun install && bun run main.ts` |
| `node/` | Node | `cd node && npm install && node main.mjs` |
| `browser/` | browser, no bundler | `python -m http.server 8000` from the repo root, then open `http://localhost:8000/examples/browser/` |

All three do the same thing: load the committed fixture model, `fit` on a
training set, `predict` on test rows, and (Bun) compute row embeddings. The
Node example needs one extra step because the package ships the `web` target:
pass the `.wasm` bytes to `init()` instead of letting it fetch.

## Weights and data

The package contains code only. Weights are not redistributed; the examples
read `scripts/fixtures/model.safetensors`, a small random-initialised 2-layer
model committed so the repo can be tested without the licensed checkpoint.
`scripts/fixtures/config.json` holds its architecture config.

Swap in a real checkpoint and its config to get meaningful predictions. The
call sequence does not change.

Because the fixture is random, the printed labels and timings demonstrate that
the pipeline runs, not model quality.

## Local package instead of npm

If the npm package is not available yet, install the local build:

```bash
cd examples/bun
bun add ../../../pkg      # or: npm install ../../../pkg
bun run main.ts
```
