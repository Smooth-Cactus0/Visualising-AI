# CNN Training Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third page, `cnn-train/`, that trains the toy CNN live on 16×16 MNIST digits, showing one step in full (forward, loss, backward, gradient, nudge) and the arc of many steps (curves, kernels over time, held-out accuracy), with the trained weights loadable into the 2D and 3D pages.

**Architecture:** Pure maths in `core/` (a backward pass and SGD next to the existing forward pass, a reader for the digit file), shared canvas helpers in `ui/grid.js` extracted from the 2D page, and one self-contained page that imports them. The digits come from a 1 MB binary file produced by a standard-library Python script. Spec: `docs/superpowers/specs/2026-10-12-cnn-training-page-design.md`.

**Tech Stack:** Plain HTML + ES modules served statically (no build step), Node 24 for tests (`node file.test.mjs`), Python 3.12 standard library for the data script, headless Edge for browser checks.

## Global Constraints

- No build step; pages are plain HTML with `<script type="module">`; external scripts only from cdnjs or jsDelivr, pinned. This feature needs none.
- `core/` stays DOM-free. `ui/` holds shared DOM helpers.
- Network fixed at 16×16, 4 → 8 kernels, FC 8×10, stride 1, padding 1, dilation 1, ReLU, max pooling. Weights shapes: `bank` 4×9, `l2` 8×4×9, `W` 8×10, `B` 10.
- Colour grammar: green = receptive field and gradient sources; yellow is not used on the training page. Diverging colours for signed values, grey for the digit.
- Every technical term gets a dotted definition through `ui/terms.js` and an entry in a Terms section.
- Wherever a prediction appears, the page says the weights are being trained here. The explorer pages say "random" unless trained weights are loaded.
- Theme-aware (light/dark via the CSS tokens of the 2D page), responsive, reduced motion respected, canvases sized in CSS as well as pixels.
- Files are LF; `core.autocrlf` is on, so do not worry about the CRLF warnings. In this environment, text fed to bash through heredocs loses backslashes: write files with the Write tool.
- Conventional Commits, local commits only; a push happens once at the end, after the browser checks, and is reported.
- NZ English in all copy.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tools/make_mnist16.py` | Download, select, resample, write `data/mnist16.bin` | 1 |
| `data/mnist16.bin`, `data/README.md` | The digits and their provenance | 1 |
| `.gitignore` | `tools/.cache/` | 1 |
| `core/data.js`, `core/data.test.mjs` | Parse the digit file, give one digit as a 0–1 `Float32Array` | 2 |
| `core/train.js`, `core/train.test.mjs` | PRNG, init, loss and gradients, SGD step, batch average, predict, accuracy, shuffled order | 3, 4 |
| `ui/grid.js` | Canvas grid drawing, cell hit-testing, colours, `fmt` (moved out of the 2D page) | 5 |
| `cnn-2d/index.html` | Imports `ui/grid.js`; loads trained weights; banner | 5, 8 |
| `ui/terms.js` | Nine training terms | 6 |
| `cnn-train/index.html` | The page: step band (6), play band (7) | 6, 7 |
| `ui/trained.js` | Local-storage hand-off helpers shared by the three pages | 7 |
| `cnn-3d/index.html` | Loads trained weights; banner | 8 |
| `index.html` | Third link | 8 |
| `project_notes.md`, `HANDOFF.md`, `CLAUDE.md` | Record the result | 9 |

---

### Task 1: The digit file and the script that makes it

**Files:**
- Create: `tools/make_mnist16.py`
- Create: `data/mnist16.bin` (generated), `data/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `data/mnist16.bin` in the MN16 format (bytes 0–3 `MN16`, 4–5 nTrain uint16 LE, 6–7 nTest uint16 LE, 8 side = 16, 9–15 zero, then train labels, train pixels, test labels, test pixels). 3,000 training and 1,000 test digits, 300 and 100 per class, seed 1. Size 1,028,016 bytes.

- [ ] **Step 1: Write the script**

Create `tools/make_mnist16.py`:

```python
#!/usr/bin/env python3
"""Make data/mnist16.bin: MNIST digits area-resampled to 16x16, 300 training and 100 test digits per class.

Standard library only. The four MNIST files are downloaded once into tools/.cache/ (git-ignored).
Usage: python tools/make_mnist16.py [--cache tools/.cache] [--out data/mnist16.bin]
Format of the output (little-endian): bytes 0-3 'MN16', 4-5 nTrain (uint16), 6-7 nTest (uint16), 8 side (16),
9-15 zero; then nTrain labels (1 byte each), nTrain x side*side pixels (row-major, 0 background .. 255 full ink),
nTest labels, nTest x side*side pixels.
"""
import argparse, gzip, os, random, struct, urllib.request

MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {
    'train_img': 'train-images-idx3-ubyte.gz', 'train_lab': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz', 'test_lab': 't10k-labels-idx1-ubyte.gz',
}
SIDE, PER_CLASS = 16, {'train': 300, 'test': 100}
SEED = 1


def fetch(cache, name):
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        print('downloading', name)
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def read_idx(path):
    """Return (count, rows, cols, data bytes) for an IDX image file, or (count, 1, 1, data) for labels."""
    d = gzip.open(path).read()
    magic, n = struct.unpack('>II', d[:8])
    if magic & 0xff == 3:
        r, c = struct.unpack('>II', d[8:16])
        return n, r, c, d[16:]
    return n, 1, 1, d[8:]


def resample(img, src, side=SIDE):
    """Box-filter a src x src byte image down to side x side; each output cell averages the input area it covers."""
    s = src / side
    out = []
    for i in range(side):
        for j in range(side):
            y0, y1, x0, x1 = i * s, (i + 1) * s, j * s, (j + 1) * s
            acc = 0.0
            for y in range(int(y0), min(src, int(y1) + 1)):
                wy = min(y1, y + 1) - max(y0, y)
                if wy <= 0:
                    continue
                for x in range(int(x0), min(src, int(x1) + 1)):
                    wx = min(x1, x + 1) - max(x0, x)
                    if wx > 0:
                        acc += wy * wx * img[y * src + x]
            out.append(max(0, min(255, int(round(acc / (s * s))))))
    return bytes(out)


def pick(labels, per_class, rng):
    """per_class indices of each digit 0-9, chosen at random, then shuffled so the file is not sorted by class."""
    by_class = {d: [] for d in range(10)}
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    chosen = []
    for d in range(10):
        chosen += rng.sample(by_class[d], per_class)
    rng.shuffle(chosen)
    return chosen


def build(cache, out):
    rng = random.Random(SEED)
    sets = {}
    for split in ('train', 'test'):
        n, r, c, px = read_idx(fetch(cache, FILES[split + '_img']))
        nl, _, _, lab = read_idx(fetch(cache, FILES[split + '_lab']))
        assert n == nl and r == c, 'unexpected MNIST layout'
        idx = pick(lab, PER_CLASS[split], rng)
        labels = bytes(lab[i] for i in idx)
        pixels = b''.join(resample(px[i * r * c:(i + 1) * r * c], r) for i in idx)
        sets[split] = (labels, pixels)
        print(split, len(idx), 'digits')
    header = b'MN16' + struct.pack('<HHB', len(sets['train'][0]), len(sets['test'][0]), SIDE) + bytes(7)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'wb') as f:
        f.write(header + sets['train'][0] + sets['train'][1] + sets['test'][0] + sets['test'][1])
    print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default=os.path.join('tools', '.cache'))
    ap.add_argument('--out', default=os.path.join('data', 'mnist16.bin'))
    a = ap.parse_args()
    build(a.cache, a.out)
```

- [ ] **Step 2: Ignore the download cache**

Append to `.gitignore`:

```
# MNIST downloads used by tools/make_mnist16.py
tools/.cache/
```

- [ ] **Step 3: Run the script**

Run from the repo root: `python -I tools/make_mnist16.py`
Expected: four "downloading" lines (about 12 MB in all), then `train 3000 digits`, `test 1000 digits`, `wrote data/mnist16.bin 1028016 bytes`. Takes under a minute.

- [ ] **Step 4: Check the file**

Run:

```
python -I -c "
import struct, collections
d = open('data/mnist16.bin', 'rb').read()
nTr, nTe, side = struct.unpack('<HHB', d[4:9]); print(d[:4], nTr, nTe, side, len(d))
tr = d[16:16 + nTr]; te = d[16 + nTr + nTr * 256: 16 + nTr + nTr * 256 + nTe]
print('train per class', sorted(collections.Counter(tr).items())); print('test per class', sorted(collections.Counter(te).items()))
px = d[16 + nTr: 16 + nTr + 256]; print('first digit label', tr[0], 'ink pixels', sum(1 for v in px if v > 64), 'max', max(px))
"
```

Expected: `b'MN16' 3000 1000 16 1028016`, each class 300 in train and 100 in test, the first digit with a few dozen ink pixels and max 255.

- [ ] **Step 5: Write the data README**

Create `data/README.md`:

```markdown
# data/

`mnist16.bin`: 3,000 training and 1,000 test MNIST digits (300 and 100 per class), area-resampled from 28×28 to 16×16, the input size of the toy network on every page. Made by `tools/make_mnist16.py` (standard-library Python, seed 1): `python tools/make_mnist16.py` regenerates it byte for byte.

Format, little-endian: bytes 0–3 `MN16`; 4–5 number of training digits (uint16); 6–7 number of test digits (uint16); 8 side length (16); 9–15 zero. Then the training labels (one byte each), the training pixels (256 bytes per digit, row-major, 0 = background, 255 = full ink), the test labels, the test pixels. 1,028,016 bytes. Read it with `core/data.js`.

Source: the MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and Christopher J.C. Burges (http://yann.lecun.com/exdb/mnist/), downloaded from the mirror `https://ossci-datasets.s3.amazonaws.com/mnist/`. MNIST is made available under the Creative Commons Attribution-Share Alike 3.0 licence; this subset is a derived work under the same licence, with the same attribution.
```

- [ ] **Step 6: Commit**

```bash
git add tools/make_mnist16.py data/mnist16.bin data/README.md .gitignore
git commit -m "feat(data): 16x16 MNIST subset and the script that makes it"
```

---

### Task 2: Reading the digit file in JavaScript

**Files:**
- Create: `core/data.js`
- Test: `core/data.test.mjs`

**Interfaces:**
- Produces: `parseMnist16(bytes: Uint8Array | ArrayBuffer) → { N: number, train: Set, test: Set }` with `Set = { n: number, labels: Uint8Array, pixels: Uint8Array }` (views into the input, no copy); `image(set, i) → Float32Array(N*N)` scaled 0–1, a fresh array each call.

- [ ] **Step 1: Write the failing test**

Create `core/data.test.mjs`:

```js
import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

const d = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
assert(d.N === 16 && d.train.n === 3000 && d.test.n === 1000, 'counts');
assert(d.train.labels.length === 3000 && d.train.pixels.length === 3000 * 256, 'train views');
assert(d.test.labels.length === 1000 && d.test.pixels.length === 1000 * 256, 'test views');
for (const set of [d.train, d.test]) {
  const per = new Array(10).fill(0); for (const l of set.labels) { assert(l >= 0 && l <= 9, 'label range'); per[l]++; }
  assert(per.every((v) => v === set.n / 10), 'balanced classes: ' + per);
}
const x = image(d.train, 0);
assert(x instanceof Float32Array && x.length === 256, 'image shape');
assert(x.every((v) => v >= 0 && v <= 1) && Math.max(...x) > 0.9 && x.filter((v) => v === 0).length > 100, 'image scale');
assert(image(d.train, 0) !== x, 'fresh array per call');
let threw = false; try { parseMnist16(new Uint8Array(32)); } catch { threw = true; } assert(threw, 'rejects a non-MN16 buffer');
console.log('core/data.js OK');
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/data.test.mjs`
Expected: `Cannot find module` for `./data.js`.

- [ ] **Step 3: Write the reader**

Create `core/data.js`:

```js
// core/data.js — reader for data/mnist16.bin (format in data/README.md). Pure, no DOM.

/** Parse an MN16 buffer. Returns views into it (no copy): { N, train: { n, labels, pixels }, test: { ... } }. */
export function parseMnist16(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.length < 16 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'MN16') throw new Error('not an MN16 file');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const nTrain = dv.getUint16(4, true), nTest = dv.getUint16(6, true), N = u8[8];
  const px = N * N;
  let off = 16;
  const set = (n) => {
    const labels = u8.subarray(off, off + n); off += n;
    const pixels = u8.subarray(off, off + n * px); off += n * px;
    return { n, labels, pixels };
  };
  const train = set(nTrain), test = set(nTest);
  if (off !== u8.length) throw new Error(`MN16 size mismatch: expected ${off} bytes, got ${u8.length}`);
  return { N, train, test };
}

/** Digit i of a set as a fresh Float32Array scaled to 0–1. */
export function image(set, i) {
  const px = set.pixels.length / set.n;
  const out = new Float32Array(px);
  const base = i * px;
  for (let q = 0; q < px; q++) out[q] = set.pixels[base + q] / 255;
  return out;
}
```

- [ ] **Step 4: Run the test**

Run: `node core/data.test.mjs`
Expected: `core/data.js OK`.

- [ ] **Step 5: Commit**

```bash
git add core/data.js core/data.test.mjs
git commit -m "feat(core): reader for the 16x16 MNIST file"
```

---
### Task 3: Loss and gradients, checked against finite differences

**Files:**
- Create: `core/train.js`
- Test: `core/train.test.mjs`

**Interfaces:**
- Consumes: `forward`, `defaultBank` from `core/cnn.js` (forward state fields `c1, a1, p1, c2, a2, p2, gap, logits, prob, O, Q, Q2`).
- Produces: `mulberry32(seed) → () => number`; `initWeights({ seed = 1, layer1 = 'random' | 'edges' }) → { bank, l2, W, B }`; `crossEntropy(prob, label) → number`; `lossAndGrad(weights, input, label, { poolMode = 'max' }) → { S, loss, grads: { bank, l2, W, B }, back: { scores, gap, pool2, conv2, pool1, conv1 } }`. `grads` have the shapes of the weights as plain arrays; `back.scores` and `back.gap` are plain arrays, the other `back` maps are arrays of `Float32Array` (one per channel), `conv1`/`conv2` being gradients at the pre-ReLU conv outputs.

- [ ] **Step 1: Write the failing test (gradient check)**

Create `core/train.test.mjs`:

```js
import { mulberry32, initWeights, crossEntropy, lossAndGrad } from './train.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

// a small random instance: 8x8 input so the check runs in a second
const N = 8, rnd = mulberry32(7);
const input = Float32Array.from({ length: N * N }, () => rnd());
const label = 3;
const w0 = initWeights({ seed: 3 });
assert(w0.bank.length === 4 && w0.l2.length === 8 && w0.l2[0].length === 4 && w0.W.length === 8 && w0.W[0].length === 10 && w0.B.length === 10, 'weight shapes');
assert(initWeights({ seed: 3 }).bank[0][0] === w0.bank[0][0] && initWeights({ seed: 4 }).bank[0][0] !== w0.bank[0][0], 'seeded init');
assert(initWeights({ seed: 3, layer1: 'edges' }).bank[0].join() === '-1,0,1,-2,0,2,-1,0,1', 'edge-detector start');

assert(Math.abs(crossEntropy([0.1, 0.2, 0.7], 2) + Math.log(0.7)) < 1e-12, 'cross-entropy');

for (const poolMode of ['max', 'avg']) {
  const { loss, grads, back } = lossAndGrad(w0, input, label, { poolMode });
  assert(loss > 0 && Number.isFinite(loss), 'finite loss');
  assert(back.scores.length === 10 && Math.abs(back.scores.reduce((a, b) => a + b, 0)) < 1e-9, 'dL/dscores = p - y sums to 0');
  assert(back.conv1.length === 4 && back.conv1[0].length === N * N && back.pool1[0].length === 16 && back.conv2.length === 8 && back.pool2[0].length === 4, 'backward map shapes');
  // central finite differences on every parameter
  const h = 1e-3; let maxErr = 0; const errs = [];
  const clone = (w) => ({ bank: w.bank.map((k) => k.slice()), l2: w.l2.map((ks) => ks.map((k) => k.slice())), W: w.W.map((r) => r.slice()), B: w.B.slice() });
  const lossAt = (w) => lossAndGrad(w, input, label, { poolMode }).loss;
  const check = (get, set, ana, name) => {
    const wp = clone(w0); set(wp, get(wp) + h); const wm = clone(w0); set(wm, get(wm) - h);
    const num = (lossAt(wp) - lossAt(wm)) / (2 * h);
    const err = Math.abs(num - ana) / (1 + Math.abs(ana)); errs.push(err);
    if (err > maxErr) maxErr = err;
    assert(err < 2e-3, `${name}: numeric ${num} vs analytic ${ana}`);
  };
  w0.bank.forEach((k, i) => k.forEach((_, q) => check((w) => w.bank[i][q], (w, v) => { w.bank[i][q] = v; }, grads.bank[i][q], `bank[${i}][${q}] ${poolMode}`)));
  w0.l2.forEach((ks, i) => ks.forEach((k, s) => k.forEach((_, q) => check((w) => w.l2[i][s][q], (w, v) => { w.l2[i][s][q] = v; }, grads.l2[i][s][q], `l2[${i}][${s}][${q}] ${poolMode}`))));
  w0.W.forEach((r, f) => r.forEach((_, c) => check((w) => w.W[f][c], (w, v) => { w.W[f][c] = v; }, grads.W[f][c], `W[${f}][${c}] ${poolMode}`)));
  w0.B.forEach((_, c) => check((w) => w.B[c], (w, v) => { w.B[c] = v; }, grads.B[c], `B[${c}] ${poolMode}`));
  errs.sort((a, b) => a - b);
  console.log(`gradient check (${poolMode}): ${errs.length} parameters, max normalised error ${maxErr.toExponential(2)}, median ${errs[errs.length >> 1].toExponential(2)}`);
  assert(errs[errs.length >> 1] < 1e-4, 'median gradient error');
}
console.log('core/train.js OK');
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/train.test.mjs`
Expected: `Cannot find module` for `./train.js`.

- [ ] **Step 3: Write the maths**

Create `core/train.js`:

```js
// core/train.js — training maths for the toy CNN: loss, backward pass, SGD. Pure functions, no DOM.
// Shapes follow core/cnn.js: bank 4×9, l2 8×4×9, W 8×10, B 10. Stride 1, padding 1, dilation 1, 3×3 kernels.
import { forward, defaultBank } from './cnn.js';
import { image } from './data.js';

/** Small seeded PRNG (mulberry32); returns a function giving numbers in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random weights, uniform in ±1/√fan_in, biases 0. layer1 'edges' uses the hand-picked bank instead. */
export function initWeights({ seed = 1, layer1 = 'random', C1 = 4, C2 = 8, NC = 10 } = {}) {
  const rnd = mulberry32(seed);
  const uni = (n, s) => Array.from({ length: n }, () => (rnd() * 2 - 1) * s);
  const bank = layer1 === 'edges' ? defaultBank.map((k) => k.slice()) : Array.from({ length: C1 }, () => uni(9, 1 / 3));
  const l2 = Array.from({ length: C2 }, () => Array.from({ length: C1 }, () => uni(9, 1 / 6)));
  const W = Array.from({ length: C2 }, () => uni(NC, 1 / Math.sqrt(C2)));
  const B = Array(NC).fill(0);
  return { bank, l2, W, B };
}

export function crossEntropy(prob, label) { return -Math.log(Math.max(prob[label], 1e-12)); }

/** Backward through a 3×3 conv (stride 1, pad 1): accumulates ∂L/∂kernel into dKer (9) and, if given, ∂L/∂x into dX (n×n). */
function convBackward(x, n, ker, g, dKer, dX) {
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const gi = g[i * n + j]; if (gi === 0) continue;
    for (let a = 0; a < 3; a++) {
      const r = i + a - 1; if (r < 0 || r >= n) continue;
      for (let b = 0; b < 3; b++) {
        const c = j + b - 1; if (c < 0 || c >= n) continue;
        dKer[a * 3 + b] += gi * x[r * n + c];
        if (dX) dX[r * n + c] += gi * ker[a * 3 + b];
      }
    }
  }
}

/** Backward through 2×2 / stride-2 pooling of act (n×n) → grad (Q×Q): max routes to the first maximum, avg spreads. */
function poolBackward(act, n, grad, Q, mode) {
  const out = new Float32Array(n * n);
  for (let i = 0; i < Q; i++) for (let j = 0; j < Q; j++) {
    const g = grad[i * Q + j]; if (g === 0) continue;
    const cells = [[2 * i, 2 * j], [2 * i, 2 * j + 1], [2 * i + 1, 2 * j], [2 * i + 1, 2 * j + 1]].filter(([r, c]) => r < n && c < n);
    if (mode === 'max') {
      let best = cells[0], bv = -Infinity;
      for (const [r, c] of cells) { const v = act[r * n + c]; if (v > bv) { bv = v; best = [r, c]; } }
      out[best[0] * n + best[1]] += g;
    } else for (const [r, c] of cells) out[r * n + c] += g / cells.length;
  }
  return out;
}

const reluGate = (grad, pre) => grad.map((v, q) => (pre[q] > 0 ? v : 0));

/**
 * Loss and gradients for one digit. Returns { S, loss, grads, back }: S the forward state, grads the gradient of the
 * loss for every weight (same shapes as the weights, plain arrays), back the gradient at each activation for drawing
 * (scores = p − y, gap, pool2, conv2 (pre-ReLU), pool1, conv1 (pre-ReLU)).
 */
export function lossAndGrad(weights, input, label, { poolMode = 'max' } = {}) {
  const { bank, l2, W, B } = weights;
  const N = Math.round(Math.sqrt(input.length));
  const S = forward({ input, N, bank, l2, W, B, poolMode });
  const C1 = bank.length, C2 = l2.length;
  const { O, Q, Q2 } = S;
  const loss = crossEntropy(S.prob, label);
  const dScores = S.prob.map((p, c) => p - (c === label ? 1 : 0));
  const dW = W.map((row, f) => row.map((_, c) => S.gap[f] * dScores[c]));
  const dB = dScores.slice();
  const dGap = S.gap.map((_, f) => dScores.reduce((t, g, c) => t + W[f][c] * g, 0));
  const dPool2 = dGap.map((g) => new Float32Array(Q2 * Q2).fill(g / (Q2 * Q2)));
  const dA2 = dPool2.map((g, k) => poolBackward(S.a2[k], Q, g, Q2, poolMode));
  const dC2 = dA2.map((g, k) => reluGate(g, S.c2[k]));
  const dL2 = l2.map((ks) => ks.map(() => new Float64Array(9)));
  const dP1 = S.p1.map(() => new Float32Array(Q * Q));
  for (let k = 0; k < C2; k++) for (let s = 0; s < C1; s++) convBackward(S.p1[s], Q, l2[k][s], dC2[k], dL2[k][s], dP1[s]);
  const dA1 = dP1.map((g, k) => poolBackward(S.a1[k], O, g, Q, poolMode));
  const dC1 = dA1.map((g, k) => reluGate(g, S.c1[k]));
  const dBank = bank.map(() => new Float64Array(9));
  for (let k = 0; k < C1; k++) convBackward(input, N, bank[k], dC1[k], dBank[k], null);
  return {
    S, loss,
    grads: { bank: dBank.map((d) => Array.from(d)), l2: dL2.map((ks) => ks.map((d) => Array.from(d))), W: dW, B: dB },
    back: { scores: dScores, gap: dGap, pool2: dPool2, conv2: dC2, pool1: dP1, conv1: dC1 },
  };
}
```

- [ ] **Step 4: Run the test**

Run: `node core/train.test.mjs`
Expected: two `gradient check` lines with max normalised error below 2e-3 and median below 1e-4, then `core/train.js OK`. If the max error fails on a single parameter sitting on a ReLU or max-pool kink, change the test's seeds (`mulberry32(7)`, `initWeights({ seed: 3 })`) and note the chosen seeds in the commit body; do not loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add core/train.js core/train.test.mjs
git commit -m "feat(core): loss and backward pass for the toy CNN, checked against finite differences"
```

---

### Task 4: SGD step, batches, prediction, a training pass

**Files:**
- Modify: `core/train.js` (append), `core/train.test.mjs` (append)

**Interfaces:**
- Consumes: `lossAndGrad`, `initWeights` from Task 3; `parseMnist16`, `image` from Task 2.
- Produces: `sgdStep(weights, grads, lr) → weights` (fresh arrays); `averageGrads(list) → grads`; `predict(weights, input) → { prob, argmax }`; `accuracy(weights, set, indices) → number` in 0–1; `shuffledOrder(n, seed) → number[]`.

- [ ] **Step 1: Append the failing tests**

Append to `core/train.test.mjs` (before the final `console.log`), and add `sgdStep, averageGrads, predict, accuracy, shuffledOrder` to the import from `./train.js` plus `import { readFileSync } from 'node:fs'; import { parseMnist16, image } from './data.js';` at the top:

```js
// SGD on one real digit: the loss must fall well below its start
const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
{
  let w = initWeights({ seed: 1 });
  const x = image(data.train, 0), y = data.train.labels[0];
  const l0 = lossAndGrad(w, x, y).loss;
  for (let t = 0; t < 50; t++) w = sgdStep(w, lossAndGrad(w, x, y).grads, 0.05);
  const l1 = lossAndGrad(w, x, y).loss;
  assert(l1 < 0.5 * l0, `loss after 50 steps ${l1} vs start ${l0}`);
  assert(w.bank[0] !== initWeights({ seed: 1 }).bank[0] && Number.isFinite(w.W[0][0]), 'fresh arrays');
  console.log(`one-digit SGD: loss ${l0.toFixed(3)} → ${l1.toFixed(3)} after 50 steps`);
}
// averaging two gradients
{
  const w = initWeights({ seed: 1 });
  const g1 = lossAndGrad(w, image(data.train, 0), data.train.labels[0]).grads, g2 = lossAndGrad(w, image(data.train, 1), data.train.labels[1]).grads;
  const g = averageGrads([g1, g2]);
  assert(Math.abs(g.W[2][3] - (g1.W[2][3] + g2.W[2][3]) / 2) < 1e-12 && Math.abs(g.bank[1][4] - (g1.bank[1][4] + g2.bank[1][4]) / 2) < 1e-12 && Math.abs(g.l2[5][2][8] - (g1.l2[5][2][8] + g2.l2[5][2][8]) / 2) < 1e-12 && Math.abs(g.B[9] - (g1.B[9] + g2.B[9]) / 2) < 1e-12, 'averaged gradients');
}
// shuffled order is a permutation and depends on the seed
{
  const o = shuffledOrder(3000, 1);
  assert(o.length === 3000 && new Set(o).size === 3000 && Math.min(...o) === 0 && Math.max(...o) === 2999, 'permutation');
  assert(shuffledOrder(3000, 1).join() === o.join() && shuffledOrder(3000, 2).join() !== o.join(), 'seeded shuffle');
}
// one pass over the training digits must beat 80% on the held-out digits
{
  const testIdx = Array.from({ length: data.test.n }, (_, i) => i);
  let w = initWeights({ seed: 1 });
  const before = accuracy(w, data.test, testIdx);
  const t0 = Date.now();
  for (const i of shuffledOrder(data.train.n, 2)) w = sgdStep(w, lossAndGrad(w, image(data.train, i), data.train.labels[i]).grads, 0.05);
  const ms = Date.now() - t0;
  const after = accuracy(w, data.test, testIdx);
  console.log(`one pass (3000 digits, lr 0.05, batch 1): held-out accuracy ${(before * 100).toFixed(1)}% → ${(after * 100).toFixed(1)}% in ${ms} ms`);
  assert(after > 0.8, 'held-out accuracy after one pass: ' + after);
  const p = predict(w, image(data.test, 0));
  assert(p.prob.length === 10 && p.argmax === p.prob.indexOf(Math.max(...p.prob)), 'predict');
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/train.test.mjs`
Expected: `SyntaxError: The requested module './train.js' does not provide an export named 'sgdStep'`.

- [ ] **Step 3: Append the functions**

Append to `core/train.js`:

```js
/** Element-wise mean of a list of gradient sets. */
export function averageGrads(list) {
  const n = list.length;
  const mean = (pick) => pick(list[0]).map((_, q) => list.reduce((t, g) => t + pick(g)[q], 0) / n);
  return {
    bank: list[0].bank.map((_, i) => mean((g) => g.bank[i])),
    l2: list[0].l2.map((ks, i) => ks.map((_, s) => mean((g) => g.l2[i][s]))),
    W: list[0].W.map((_, f) => mean((g) => g.W[f])),
    B: mean((g) => g.B),
  };
}

/** One SGD update, w − lr · g, returning fresh arrays; the inputs are not touched. */
export function sgdStep(weights, grads, lr) {
  return {
    bank: weights.bank.map((k, i) => k.map((w, q) => w - lr * grads.bank[i][q])),
    l2: weights.l2.map((ks, i) => ks.map((k, s) => k.map((w, q) => w - lr * grads.l2[i][s][q]))),
    W: weights.W.map((row, f) => row.map((w, c) => w - lr * grads.W[f][c])),
    B: weights.B.map((w, c) => w - lr * grads.B[c]),
  };
}

/** Class probabilities and the top class for one digit (forward only). */
export function predict(weights, input, { poolMode = 'max' } = {}) {
  const N = Math.round(Math.sqrt(input.length));
  const { prob } = forward({ input, N, ...weights, poolMode });
  let argmax = 0; for (let c = 1; c < prob.length; c++) if (prob[c] > prob[argmax]) argmax = c;
  return { prob, argmax };
}

/** Fraction of the given digits of a set that the weights classify correctly. */
export function accuracy(weights, set, indices, opts) {
  let ok = 0;
  for (const i of indices) if (predict(weights, image(set, i), opts).argmax === set.labels[i]) ok++;
  return ok / indices.length;
}

/** A seeded permutation of 0..n-1 (Fisher–Yates), one pass over a set. */
export function shuffledOrder(n, seed) {
  const rnd = mulberry32(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
```

- [ ] **Step 4: Run the test**

Run: `node core/train.test.mjs`
Expected: the gradient-check lines, `one-digit SGD: loss … → …`, `one pass … → …% in … ms` with the accuracy above 80%, then `core/train.js OK`. If one pass lands below 80%, do not loosen the test: try lr 0.1 and the init scales ±1/2, ±1/4, ±1/√8 in turn, keep the first combination that passes, and record the final values in the spec's "Core maths" section and in the commit body.

- [ ] **Step 5: Commit**

```bash
git add core/train.js core/train.test.mjs
git commit -m "feat(core): SGD step, batch averaging, prediction and a one-pass training check"
```

---
### Task 5: Shared canvas grid helpers, moved out of the 2D page

**Files:**
- Create: `ui/grid.js`
- Modify: `cnn-2d/index.html` (the helpers block after `// ---------- helpers ----------`, the import line, one `drawGrid` call)

**Interfaces:**
- Produces: `fmt(v) → string`, `isDark() → boolean`, `cssVar(name) → string`, `divColor(v, scale) → css colour`, `grayColor(v) → css colour`, `maxAbs(array) → number ≥ 0.01`, `drawGrid(canvas, data, w, h, options)`, `cellAt(canvas, pointerEvent) → [row, col] | null`. `options`: `mode: 'div' | undefined` (diverging or grey), `scale`, `labels`, `taps: [[r,c],…]` (yellow cells), `cell: [r,c]` (yellow), `rect: [r0,r1,c0,c1]` (green), `outline: boolean` (green canvas outline), `padBorder: boolean` with `pad: number` (dotted border inset by `pad` cells).

- [ ] **Step 1: Create the module**

Create `ui/grid.js` (this is the 2D page's code, unchanged except that `padBorder` reads `o.pad` instead of a page variable):

```js
// ui/grid.js — canvas grid drawing and hit-testing shared by the 2D and training pages.
// Colours come from the page's CSS tokens (--line, --ink, --hl, --hl2) and follow its light/dark theme.
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/** Short number: '0', '1.5', '-0.33'. */
export const fmt = (v) => { if (Math.abs(v) < 1e-9) return '0'; const s = v.toFixed(2); return s.replace(/\.?0+$/, ''); };
export const isDark = () => document.documentElement.dataset.theme === 'dark' || (matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light');
export const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/** Diverging colour for a signed value: blue negative, orange positive, the page's zero tone in between. */
export function divColor(v, scale) {
  const t = clamp(v / scale, -1, 1);
  const neg = [42, 111, 214], pos = [229, 100, 30];
  const zero = isDark() ? [42, 47, 56] : [243, 244, 241];
  const c = t < 0 ? neg : pos, a = Math.abs(t);
  return `rgb(${zero.map((z, i) => Math.round(z + (c[i] - z) * a)).join(',')})`;
}
/** Grey for a 0–1 value (ink on the page background). */
export function grayColor(v) { const g = isDark() ? 40 + v * 200 : 245 - v * 225; return `rgb(${g | 0},${g | 0},${g | 0})`; }
export const maxAbs = (m) => Math.max(0.01, ...Array.from(m, Math.abs));

const layouts = new WeakMap(); // canvas -> {cell, ox, oy, w, h}

/** Draw a w×h grid of values on a canvas, with optional labels and highlights; remembers the layout for cellAt. */
export function drawGrid(cv, data, w, h, o = {}) {
  const ctx = cv.getContext('2d'); const Wd = cv.width, Hd = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, Wd, Hd);
  const cell = Math.floor(Math.min(Wd / w, Hd / h)); const ox = Math.floor((Wd - cell * w) / 2), oy = Math.floor((Hd - cell * h) / 2);
  layouts.set(cv, { cell, ox, oy, w, h });
  const scale = o.scale || 1;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = data[r * w + c];
    ctx.fillStyle = o.mode === 'div' ? divColor(v, scale) : grayColor(v);
    ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
  }
  ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
  for (let r = 0; r <= h; r++) { ctx.beginPath(); ctx.moveTo(ox, oy + r * cell + .5); ctx.lineTo(ox + w * cell, oy + r * cell + .5); ctx.stroke(); }
  for (let c = 0; c <= w; c++) { ctx.beginPath(); ctx.moveTo(ox + c * cell + .5, oy); ctx.lineTo(ox + c * cell + .5, oy + h * cell); ctx.stroke(); }
  if (o.padBorder) { const pad = o.pad || 0; ctx.strokeStyle = 'rgba(128,128,128,.9)'; ctx.setLineDash([3, 3]);
    ctx.strokeRect(ox + pad * cell + .5, oy + pad * cell + .5, (w - 2 * pad) * cell, (h - 2 * pad) * cell); ctx.setLineDash([]); }
  if (o.labels && cell >= 22) {
    ctx.font = `${Math.max(9, cell * 0.36)}px ${getComputedStyle(document.body).fontFamily}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const v = data[r * w + c]; const strong = o.mode === 'div' ? Math.abs(v / scale) > .55 : (isDark() ? v < .5 : v > .5);
      ctx.fillStyle = strong ? '#fff' : cssVar('--ink');
      if (o.mode !== 'div' && !strong && !isDark()) ctx.fillStyle = '#333';
      ctx.fillText(fmt(v), ox + c * cell + cell / 2, oy + r * cell + cell / 2);
    }
  }
  const hl = cssVar('--hl'), hl2 = cssVar('--hl2');
  if (o.taps) { ctx.strokeStyle = hl; ctx.lineWidth = 3; o.taps.forEach(([r, c]) => ctx.strokeRect(ox + c * cell + 1.5, oy + r * cell + 1.5, cell - 3, cell - 3)); }
  if (o.cell) { ctx.strokeStyle = hl; ctx.lineWidth = 3; ctx.strokeRect(ox + o.cell[1] * cell + 1.5, oy + o.cell[0] * cell + 1.5, cell - 3, cell - 3); }
  if (o.rect) { const [r0, r1, c0, c1] = o.rect; ctx.strokeStyle = hl2; ctx.lineWidth = Math.max(2, Math.min(3, cell / 6));
    const lw = ctx.lineWidth / 2; ctx.strokeRect(ox + c0 * cell + lw, oy + r0 * cell + lw, (c1 - c0 + 1) * cell - 2 * lw, (r1 - r0 + 1) * cell - 2 * lw);
    ctx.fillStyle = hl2; ctx.globalAlpha = .18; ctx.fillRect(ox + c0 * cell, oy + r0 * cell, (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell); ctx.globalAlpha = 1; }
  if (o.outline) { cv.style.outline = `3px solid ${hl2}`; cv.style.outlineOffset = '2px'; } else cv.style.outline = '';
}

/** Grid cell under a pointer event, as [row, col], or null outside the grid. */
export function cellAt(cv, ev) {
  const L = layouts.get(cv); if (!L) return null; const rect = cv.getBoundingClientRect(); const sx = rect.width / cv.width;
  const c = Math.floor((ev.clientX - rect.left - L.ox * sx) / (L.cell * sx)), r = Math.floor((ev.clientY - rect.top - L.oy * sx) / (L.cell * sx));
  return (r >= 0 && c >= 0 && r < L.h && c < L.w) ? [r, c] : null;
}
```

- [ ] **Step 2: Point the 2D page at it**

In `cnn-2d/index.html`, three edits, done with a Python replace script run through `python -I` (exact-string replacements, assert each old string occurs once):

1. After the line `import { installTerms } from '../ui/terms.js';` add
   `import { fmt, isDark, cssVar, divColor, grayColor, maxAbs, drawGrid, cellAt } from '../ui/grid.js';`
2. In the `// ---------- helpers ----------` block, delete from the line `const fmt=v=>{ if(Math.abs(v)<1e-9) return '0'; const s=v.toFixed(2); return s.replace(/\.?0+$/,''); };` through the end of `function cellAt(cv, ev){ … }` (that is: `fmt`, `isDark`, `cssVar`, `divColor`, `grayColor`, `maxAbs`, `const layouts = new WeakMap();`, `drawGrid`, `cellAt`). Keep `const clamp=…` and `function hoverable(cv, stage, ch){ … }`. Use marker slicing: cut from `const fmt=v=>` up to `function hoverable(cv, stage, ch){`.
3. Replace `drawGrid($('cPadIn'),S.pin,S.P,S.P,{padBorder:pad>0,taps,rect:R.padIn&&R.padIn.rect});` with `drawGrid($('cPadIn'),S.pin,S.P,S.P,{padBorder:pad>0,pad,taps,rect:R.padIn&&R.padIn.rect});`.

After the edits assert that `function drawGrid`, `function cellAt`, `const layouts`, `function divColor`, `function grayColor`, `const fmt=`, `const isDark=`, `const cssVar=`, `const maxAbs=` are all absent from the page.

- [ ] **Step 3: Syntax check**

Run: `awk '/<script type="module">/{f=1;next} /<\/script>/{f=0} f' cnn-2d/index.html > <scratch>/cnn-2d.mjs && node --check <scratch>/cnn-2d.mjs && node --check ui/grid.js`
Expected: no output (both parse).

- [ ] **Step 4: Prove no change in behaviour**

Stage `old/cnn-2d/index.html` from `git show HEAD:cnn-2d/index.html` and `new/cnn-2d/index.html` from the working tree in a scratch folder that also holds `new/core/cnn.js`, `new/ui/terms.js`, `new/ui/terms.css`, `new/ui/grid.js` (and `old/core`, `old/ui` likewise from HEAD). Inject the seeded `Math.random` shim after `<head>` in both pages: `<script>(function(){var s=123456789;Math.random=function(){s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};})();</script>`. Serve the folder (`python -m http.server 8767 --bind 127.0.0.1 -d <folder>` in the background), dump both DOMs with headless Edge (`msedge --headless=new --disable-gpu --no-first-run --user-data-dir=<scratch profile> --virtual-time-budget=4000 --dump-dom <url>`), strip `<script>…</script>` blocks with `perl -0pe 's/<script\b.*?<\/script>//gs'`, and `diff` them.
Expected: no differences, and no `CONSOLE` errors in the stderr log of the new page.

- [ ] **Step 5: Commit**

```bash
git add ui/grid.js cnn-2d/index.html
git commit -m "refactor: share the canvas grid helpers in ui/grid.js"
```

---
### Task 6: The training page

**Files:**
- Modify: `ui/terms.js` (append nine entries to `TERMS`)
- Create: `cnn-train/index.html`

**Interfaces:**
- Consumes: `footprint` from `core/cnn.js`; `parseMnist16`, `image` from Task 2; `initWeights`, `lossAndGrad`, `sgdStep`, `averageGrads`, `accuracy`, `shuffledOrder` from Tasks 3–4; `installTerms` from `ui/terms.js`; `fmt`, `maxAbs`, `drawGrid`, `cellAt`, `cssVar` from Task 5.
- Produces: the page; a `#handoff` container in the play band that Task 7 fills; module-scope `weights`, `digitsSeen`, `passes`, `lastTest`, `lr` that Task 7's buttons read.

- [ ] **Step 1: Add the training terms**

In `ui/terms.js`, insert these entries at the end of the `TERMS` array (after the `'training'` entry):

```js
  ['loss', /\b(cross-entropy|loss)\b/i, 'Loss (cross-entropy)', 'One number for how wrong a prediction was: minus the log of the probability given to the true class. Zero when the network is certain and right, large when it is confident and wrong. Training pushes it down.'],
  ['gradient', /\bgradients?\b/i, 'Gradient', 'For each weight, how much the loss would change per unit change of that weight, sign included: the direction that makes things worse. Written ∂L/∂w. Training moves every weight a little the other way.'],
  ['backprop', /\bbackpropagation\b/i, 'Backpropagation', 'The bookkeeping that gives every weight its gradient. The error at the output is passed back through the layers one at a time; each layer turns the gradient at its output into the gradient at its input and at its own weights (the chain rule).'],
  ['lr', /\blearning rate\b/i, 'Learning rate', 'The step size: each weight moves by learning rate × its gradient. Too small and training crawls; too large and it overshoots, so the loss jumps around instead of falling.'],
  ['update', /\b(update step|SGD)\b/, 'Update (SGD step)', 'w ← w − learning rate × ∂L/∂w for every weight at once. Stochastic gradient descent: stochastic because each step uses one digit, or a small batch, rather than the whole dataset.'],
  ['batch', /\bbatch\b/i, 'Batch', 'How many digits are seen before one update. Their gradients are averaged, so a bigger batch gives steadier but fewer steps.'],
  ['pass', /\bpass(es)?\b/i, 'Pass (epoch)', 'One trip through the whole training set, in a shuffled order. Real training runs many.'],
  ['heldout', /\bheld-out\b/i, 'Held-out test set', 'Digits the network never trains on, kept only to measure accuracy. Accuracy on the training digits can be flattered by memorising them; held-out accuracy cannot.'],
  ['init', /\binitiali[sz]ation\b/i, 'Initialisation', 'The random starting values of the weights. Small random numbers, not zeros: identical weights would get identical gradients and never become different detectors.'],
```

Check with `node --check ui/terms.js`, then `node -e "import('./ui/terms.js').then(m => console.log(m.TERMS.length))"` from the repo root. Expected: `25`.

- [ ] **Step 2: Create the page**

Create `cnn-train/index.html` with the content below (styles copied from the 2D page where the class names match, so the two pages look alike).

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>CNN explorer, training — how the weights learn</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<link rel="stylesheet" href="../ui/terms.css">
<style>
:root{
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
  --bg:#f7f8f6; --panel:#ffffff; --ink:#1d2430; --ink2:#5b6474; --line:#d9ddd6; --line2:#eef0ec;
  --accent:#2457a8; --accent-soft:#dde7f7;
  --pos:#e5641e; --neg:#2a6fd6; --zero:#f3f4f1;
  --hl:#ffd94a; --hl2:#3fae4b;
  --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a;
}}
:root[data-theme="dark"]{
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a;
}
html{scroll-padding-top:env(safe-area-inset-top,0px);}
*{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
main{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:28px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{margin:0 0 10px;max-width:68ch}
.lede{color:var(--ink2);font-size:15.5px;margin-bottom:10px;max-width:78ch}
.keys{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-bottom:26px}
.keys span{display:inline-flex;align-items:center;gap:6px}
.keys i{display:inline-block;width:14px;height:14px;border-radius:3px;border:3px solid var(--hl2)}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:22px}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.head .n{font-family:var(--mono);color:var(--ink2);font-size:13px}
.head p{color:var(--ink2);margin:0;flex:1 1 100%;font-size:14px}
canvas{display:block;max-width:100%;touch-action:none;border-radius:4px}
.cap{font-size:13px;color:var(--ink2);margin-top:6px;font-variant-numeric:tabular-nums}
.cap b{color:var(--ink);font-weight:600}
.ctrls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
button,select{font:inherit;font-size:13.5px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
label.ctl{display:inline-flex;align-items:center;gap:6px;font-size:13.5px;color:var(--ink2)}
input[type=range]{accent-color:var(--accent)}
.math{font-family:var(--mono);font-size:12.5px;line-height:1.7;background:var(--line2);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre-wrap;min-height:3.2em;margin-top:10px}
.math .sum{color:var(--accent);font-weight:600}
.note{font-size:13.5px;color:var(--ink2);border-left:3px solid var(--accent-soft);padding:4px 12px;margin:0 0 22px;max-width:72ch}
.pipe{display:flex;gap:16px;align-items:flex-start;overflow-x:auto;padding:6px 2px 12px}
.stage{flex:0 0 auto;text-align:center;font-size:12.5px;color:var(--ink2)}
.stage .t{font-weight:500;color:var(--ink);margin-bottom:6px;white-space:nowrap}
.stack{display:grid;gap:6px}
.stack canvas{cursor:crosshair}
.arrow{flex:0 0 auto;align-self:center;color:var(--ink2);font-size:20px;padding-top:18px}
.arrow.msg{font-size:13px;padding-top:0}
.bars{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;min-width:220px;text-align:left}
.bars .lab{font-family:var(--mono);font-size:12.5px;color:var(--ink2)}
.bars .bar{height:14px;background:var(--line2);border-radius:3px;overflow:hidden;cursor:pointer}
.bars .bar i{display:block;height:100%;background:var(--accent);opacity:.75}
.bars .bar.top i{opacity:1}
.bars .bar.true{outline:2px solid var(--ink);outline-offset:1px}
.bars .bar.hov{outline:2px solid var(--hl2);outline-offset:1px}
.bars .v{font-family:var(--mono);font-size:12px;color:var(--ink2);min-width:3.5em;text-align:right}
.phase{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--ink2);margin-right:8px}
.phase b{font-weight:500;padding:2px 9px;border-radius:12px;border:1px solid var(--line);color:var(--ink2)}
.phase b.done{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
.phase b.next{border-style:dashed;color:var(--ink)}
.trips{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;margin-top:6px}
.trip{text-align:center;font-size:12px;color:var(--ink2)}
.trip .t{font-weight:500;color:var(--ink);margin-bottom:4px;white-space:nowrap}
.trip .g{display:inline-flex;gap:8px;align-items:flex-end}
.trip .g span{display:block;font-size:11px;margin-top:2px;font-family:var(--mono)}
.trip canvas{cursor:crosshair;margin:0 auto}
.curves{display:flex;gap:18px;flex-wrap:wrap;margin-top:6px}
.curves > div{flex:1 1 320px;min-width:0}
.curves canvas{width:100%;height:170px;background:var(--bg);border:1px solid var(--line2)}
.snaps{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin-top:6px}
.snaps > div{text-align:center;font-size:12px;color:var(--ink2)}
.snaps canvas{margin:0 auto 4px}
.thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.thumbs > div{text-align:center;font-size:11px;color:var(--ink2);width:34px;font-family:var(--mono)}
.thumbs .ok{color:var(--hl2)} .thumbs .no{color:var(--pos)}
.lrv{font-family:var(--mono);font-size:12.5px;min-width:4em}
.counts{font-size:13px;color:var(--ink2);margin-left:auto;font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion: reduce){ *{transition:none!important} }
</style>
</head>
<body>
<main>
<h1>CNN explorer, training</h1>
<p class="lede">The same toy network as the other two pages, now learning. Real MNIST digits, shrunk to 16×16, come in one at a time: the network guesses, the loss says how wrong the guess was, the error flows back through the layers to give every weight a gradient, and every weight moves a little the other way. That is training; the rest is repetition. Step through one digit below, then play to watch the weights change over thousands. Being a toy of 12 kernels on 16×16 pixels, it tops out around 85 to 92% on digits it has never seen, well short of a real LeNet's 99%, but every mechanism is the real one.</p>
<div class="keys"><span><i></i> receptive field of a hovered cell, and the two sources of a hovered weight's gradient</span></div>
<div class="note" id="status">Loading 4,000 digits (1 MB)…</div>

<section id="s1">
  <div class="head"><h2>One step</h2><span class="n">digit → forward → loss → backward → update</span>
    <p>Step runs one phase at a time; Next digit finishes this digit and brings the next one in. Forward is what the other pages show. Backward is new: the gradient of the loss starts at the class scores as p − y and flows back through the same shapes, and each weight's gradient is what came forward times what came back at that weight. Update nudges every weight by learning rate × gradient, the other way. Hover any cell or weight for its arithmetic.</p></div>
  <div class="ctrls">
    <span class="phase" id="phase"><b data-p="0">digit</b>→<b data-p="1">forward</b>→<b data-p="2">backward</b>→<b data-p="3">update</b></span>
    <button id="stepBtn" class="primary">Step</button><button id="nextBtn">Next digit</button>
    <span class="cap" id="digitInfo"></span>
  </div>
  <div class="cap" style="margin-top:12px">Forward: what came in</div>
  <div class="pipe" id="fwd"></div>
  <div class="cap">Backward: the error flowing back through the same shapes</div>
  <div class="pipe" id="bwd"></div>
  <div class="cap">Weights: w · ∂L/∂w · w′ = w − lr·∂L/∂w</div>
  <div class="trips" id="trips"></div>
  <div class="math" id="readout"></div>
</section>

<section id="s2">
  <div class="head"><h2>Many steps</h2><span class="n">play · curves · kernels over time · held-out accuracy</span>
    <p>Play repeats the step for digit after digit. The loss curve shows the loss per digit, which is noisy, and its running average. Accuracy is measured two ways: on the last 200 training digits, and every 200 digits on 1,000 held-out test digits the network never trains on. The strip of kernels shows layer 1 at fixed milestones: with random initialisation it has to discover edges on its own; "start from edge detectors" begins from the hand-picked ones instead.</p></div>
  <div class="ctrls">
    <button id="playBtn" class="primary">Play</button>
    <label class="ctl">speed <select id="speed"><option value="10">10 / s</option><option value="50" selected>50 / s</option><option value="200">200 / s</option></select></label>
    <label class="ctl">learning rate <input type="range" id="lr" min="0" max="100" value="63" style="width:120px"><span class="lrv" id="lrv">0.05</span></label>
    <label class="ctl">batch <select id="batch"><option>1</option><option>8</option><option>32</option></select></label>
    <button id="resetBtn">Reset</button><button id="edgesBtn">Start layer 1 from edge detectors</button>
    <span class="counts" id="counts"></span>
  </div>
  <div class="curves">
    <div><canvas id="cLoss" width="560" height="170"></canvas><div class="cap">loss per digit (faint) and its running average</div></div>
    <div><canvas id="cAcc" width="560" height="170"></canvas><div class="cap">accuracy: <b>training</b>, last 200 digits (thin) · <b>held-out</b>, 1,000 test digits every 200 (thick)</div></div>
  </div>
  <div class="cap" style="margin-top:12px">Layer-1 kernels over time</div>
  <div class="snaps" id="snaps"></div>
  <div class="cap" style="margin-top:12px">Recent digits, with the guess made at the time</div>
  <div class="thumbs" id="thumbs"></div>
  <div class="ctrls" id="handoff"></div>
</section>

<section id="s3">
  <div class="head"><h2>Terms</h2><span class="n">dotted words above point here</span>
    <p>Everything on this page is built from these. Hover or tap a dotted term anywhere above for the same definition in place.</p></div>
  <div class="gloss" id="gloss"></div>
</section>
</main>
<div id="tip" role="tooltip"></div>

<script type="module">
// (Step 3 below)
</script>
</body>
</html>
```

- [ ] **Step 3: The page's script**

Replace the module script's body (`// (Step 3 below)`) with:

```js
import { footprint } from '../core/cnn.js';
import { parseMnist16, image } from '../core/data.js';
import { initWeights, lossAndGrad, sgdStep, averageGrads, accuracy, shuffledOrder } from '../core/train.js';
import { installTerms } from '../ui/terms.js';
import { fmt, maxAbs, drawGrid, cellAt, cssVar } from '../ui/grid.js';

const N = 16, C1 = 4, C2 = 8, NC = 10;
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const MILESTONES = [0, 100, 300, 1000, 3000, 10000];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);

// ---------- state ----------
let data = null, testIdx = [];
let seed = 1, layer1 = 'random', lr = 0.05, batch = 1, speed = 50;
let weights = null, order = [], orderPos = 0, passes = 0, digitsSeen = 0;
let cur = null;          // { i, input, label, phase 0..3, S, loss, grads, back, pred, applied }
let collected = [];      // gradients of the batch in progress
let shown = null;        // the gradient drawn in the triplets: this digit's, or the batch average so far
let prevWeights = null, anim = null;  // for the update animation
let probe = null;        // hover state, see readout()
let lossRaw = [], lossEma = [], ema = null, trainFlags = [], accTrain = [], accTest = [], lastTest = null, snaps = [], recent = [];
let playing = false, carry = 0, lastFrame = 0, lastDraw = 0;

// ---------- the training step, in phases ----------
function loadDigit() {
  if (orderPos >= order.length) { passes++; order = shuffledOrder(data.train.n, seed * 1000 + passes); orderPos = 0; }
  const i = order[orderPos++];
  cur = { i, input: image(data.train, i), label: data.train.labels[i], phase: 0 };
}
function doForward() {
  const r = lossAndGrad(weights, cur.input, cur.label);
  Object.assign(cur, { S: r.S, loss: r.loss, grads: r.grads, back: r.back, pred: argmax(r.S.prob), phase: 1 });
}
function doBackward() { collected.push(cur.grads); shown = collected.length === 1 ? cur.grads : averageGrads(collected); cur.phase = 2; }
function doUpdate() {
  if (collected.length >= batch) { prevWeights = weights; weights = sgdStep(weights, shown, lr); collected = []; cur.applied = true; }
  else cur.applied = false;
  cur.phase = 3; digitsSeen++; record(cur);
}
function record(d) {
  lossRaw.push([digitsSeen, d.loss]); ema = ema == null ? d.loss : ema + 0.02 * (d.loss - ema); lossEma.push([digitsSeen, ema]);
  if (lossRaw.length > 4000) { lossRaw = lossRaw.filter((_, i) => i % 2 === 0); lossEma = lossEma.filter((_, i) => i % 2 === 0); }
  trainFlags.push(d.pred === d.label); if (trainFlags.length > 200) trainFlags.shift();
  recent.push({ input: d.input, label: d.label, pred: d.pred }); if (recent.length > 20) recent.shift();
  if (digitsSeen % 20 === 0) accTrain.push([digitsSeen, trainFlags.filter(Boolean).length / trainFlags.length]);
  if (digitsSeen % 200 === 0) evalTest();
  if (MILESTONES.includes(digitsSeen)) snaps.push({ at: digitsSeen, bank: weights.bank.map((k) => k.slice()) });
}
function evalTest() { lastTest = accuracy(weights, data.test, testIdx); accTest.push([digitsSeen, lastTest]); }
function trainOne() { loadDigit(); doForward(); doBackward(); doUpdate(); }
function step() {
  if (cur.phase === 3) loadDigit(); else if (cur.phase === 0) doForward(); else if (cur.phase === 1) doBackward(); else { doUpdate(); if (cur.applied && !reduced) startAnim(); }
  renderAll();
}
function nextDigit() {
  if (cur.phase === 0) doForward(); if (cur.phase === 1) doBackward(); if (cur.phase === 2) doUpdate();
  loadDigit(); doForward(); renderAll();
}
function reset({ newSeed = false, layer1: l1 = 'random' } = {}) {
  if (newSeed) seed++; layer1 = l1;
  weights = initWeights({ seed, layer1 }); order = shuffledOrder(data.train.n, seed * 1000); orderPos = 0; passes = 0; digitsSeen = 0;
  collected = []; shown = null; prevWeights = null; anim = null; probe = null;
  lossRaw = []; lossEma = []; ema = null; trainFlags = []; accTrain = []; accTest = []; recent = [];
  snaps = [{ at: 0, bank: weights.bank.map((k) => k.slice()) }];
  evalTest(); loadDigit(); renderAll();
}

// ---------- play loop ----------
function frame(ts) {
  if (!playing) return;
  const dt = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0; lastFrame = ts;
  carry = Math.min(carry + speed * dt, Math.max(1, speed / 10));
  const t0 = performance.now();
  while (carry >= 1 && performance.now() - t0 < 12) { trainOne(); carry -= 1; }
  if (ts - lastDraw > 100) { renderAll(); lastDraw = ts; }
  requestAnimationFrame(frame);
}
function setPlaying(p) { playing = p; $('playBtn').textContent = p ? 'Pause' : 'Play'; if (p) { lastFrame = 0; carry = 0; anim = null; requestAnimationFrame(frame); } else renderAll(); }

// ---------- update animation (weights fade from old to new over 300 ms) ----------
function startAnim() { anim = { t0: performance.now() }; const tick = () => { if (!anim) return; const t = (performance.now() - anim.t0) / 300; if (t >= 1) { anim = null; renderTrips(); return; } anim.t = t; renderTrips(); requestAnimationFrame(tick); }; requestAnimationFrame(tick); }
function shownWeights() { // the weights to draw: interpolated during the animation
  if (!anim || !prevWeights) return weights; const t = anim.t || 0; const mix = (a, b) => a + t * (b - a);
  return { bank: weights.bank.map((k, i) => k.map((w, q) => mix(prevWeights.bank[i][q], w))), l2: weights.l2.map((ks, i) => ks.map((k, s) => k.map((w, q) => mix(prevWeights.l2[i][s][q], w)))), W: weights.W.map((r, f) => r.map((w, c) => mix(prevWeights.W[f][c], w))), B: weights.B.map((w, c) => mix(prevWeights.B[c], w)) };
}

// ---------- rendering helpers ----------
const lit = (R, st, ch) => (R[st] && (R[st].chs === null || R[st].chs.has(ch)) ? R[st].rect : null);
function hoverable(cv, make) {
  cv.addEventListener('pointermove', (ev) => { const p = cellAt(cv, ev); const np = p ? make(p) : null; if (JSON.stringify(np) !== JSON.stringify(probe)) { probe = np; renderAll(); } });
  cv.addEventListener('pointerleave', () => { if (probe) { probe = null; renderAll(); } });
}
function stage(title, sub, cols) { const d = document.createElement('div'); d.className = 'stage'; const t = document.createElement('div'); t.className = 't'; t.textContent = title; d.append(t); const s = document.createElement('div'); s.className = 'stack'; if (cols) s.style.gridTemplateColumns = `repeat(${cols},auto)`; d.append(s); if (sub) { const c = document.createElement('div'); c.textContent = sub; c.style.marginTop = '6px'; d.append(c); } return [d, s]; }
function arrow(txt = '→', msg = false) { const a = document.createElement('div'); a.className = 'arrow' + (msg ? ' msg' : ''); a.textContent = txt; return a; }
function canvas(w, h = w) { const cv = document.createElement('canvas'); cv.width = w; cv.height = h; cv.style.width = w + 'px'; cv.style.height = h + 'px'; return cv; }
const isW = (layer) => probe && probe.kind === 'w' && probe.layer === layer;
const gradShown = () => (cur.phase === 2 ? shown : null);

function renderPhase() {
  $('phase').querySelectorAll('b').forEach((b) => { const p = +b.dataset.p; b.classList.toggle('done', p <= cur.phase); b.classList.toggle('next', p === cur.phase + 1); });
  $('digitInfo').textContent = `training digit ${cur.i} · label ${cur.label}` + (cur.phase >= 1 ? ` · guess ${cur.pred} ${cur.pred === cur.label ? '✓' : '✗'}` : '') + (batch > 1 ? ` · batch ${collected.length}/${batch} collected` : '');
  $('stepBtn').textContent = cur.phase === 3 ? 'Step: load the next digit' : cur.phase === 0 ? 'Step: forward' : cur.phase === 1 ? 'Step: backward' : 'Step: update';
}

function renderForward() {
  const pipe = $('fwd'); pipe.innerHTML = '';
  const R = probe && probe.kind === 'map' && cur.S ? footprint({ stage: probe.stage, rect: [probe.r, probe.r, probe.c, probe.c], ch: probe.ch }, cur.S) : {};
  { const [d, s] = stage('digit', `label ${cur.label}`); const cv = canvas(96); drawGrid(cv, cur.input, N, N, { rect: R.input && R.input.rect, outline: isW('bank') }); s.append(cv); pipe.append(d); }
  if (cur.phase < 1) { pipe.append(arrow('→  Step runs the forward pass', true)); return; }
  const S = cur.S; pipe.append(arrow());
  const mapsCol = (title, maps, n, st, sub, cols, px, srcOf) => { const [d, s] = stage(title, sub, cols); const sc = Math.max(...maps.map(maxAbs));
    maps.forEach((m, idx) => { const cv = canvas(px); drawGrid(cv, m, n, n, { mode: 'div', scale: sc, rect: lit(R, st, idx), outline: !!(srcOf && srcOf(idx)) }); hoverable(cv, ([r, c]) => ({ kind: 'map', stage: st, r, c, ch: idx })); s.append(cv); }); return d; };
  pipe.append(mapsCol('conv 1 + ReLU', S.a1, S.O, 'conv1', `4 maps ${S.O}×${S.O}`, 2, 56), arrow());
  pipe.append(mapsCol('pool 1', S.p1, S.Q, 'pool1', `4 maps ${S.Q}×${S.Q}`, 2, 40, (i) => isW('l2') && probe.s === i), arrow());
  pipe.append(mapsCol('conv 2 + ReLU', S.a2, S.Q, 'conv2', `8 maps ${S.Q}×${S.Q}`, 4, 40), arrow());
  pipe.append(mapsCol('pool 2', S.p2, S.Q2, 'pool2', `8 maps ${S.Q2}×${S.Q2}`, 4, 32, (i) => probe && probe.kind === 'gap' && probe.f === i), arrow());
  { const [d, s] = stage('global average', 'vector of 8'); const cv = canvas(36, 8 * 36);
    const f = probe && (probe.kind === 'gap' || isW('W')) ? probe.f : null;
    drawGrid(cv, S.gap, 1, C2, { mode: 'div', scale: maxAbs(S.gap), labels: true, rect: f != null ? [f, f, 0, 0] : probe && (probe.kind === 'bar' || isW('B')) ? [0, C2 - 1, 0, 0] : null });
    hoverable(cv, ([r]) => ({ kind: 'gap', f: r })); s.append(cv); pipe.append(d, arrow()); }
  { const [d, s] = stage('FC weights', '8 × 10, plus 10 biases'); const w = shownWeights(); const flat = new Float32Array(C2 * NC); for (let f = 0; f < C2; f++) for (let c = 0; c < NC; c++) flat[f * NC + c] = w.W[f][c];
    const cv = canvas(NC * 20, C2 * 20);
    const rect = isW('W') ? [probe.f, probe.f, probe.c, probe.c] : probe && (probe.kind === 'bar' || isW('B')) ? [0, C2 - 1, probe.c, probe.c] : probe && probe.kind === 'gap' ? [probe.f, probe.f, 0, NC - 1] : null;
    drawGrid(cv, flat, NC, C2, { mode: 'div', scale: maxAbs(flat), rect }); hoverable(cv, ([f, c]) => ({ kind: 'w', layer: 'W', f, c })); s.append(cv); pipe.append(d, arrow()); }
  { const [d, s] = stage('scores → softmax', `loss = −log p[${cur.label}] = ${fmt(cur.loss)}`); s.append(bars(S.prob, cur.label, cur.pred)); pipe.append(d); }
}
function bars(prob, label, top) {
  const el = document.createElement('div'); el.className = 'bars';
  for (let c = 0; c < NC; c++) {
    const lab = document.createElement('div'); lab.className = 'lab'; lab.textContent = `class ${c}${c === label ? ' (true)' : ''}`;
    const hov = probe && ((probe.kind === 'bar' && probe.c === c) || ((isW('W') || isW('B')) && probe.c === c));
    const bar = document.createElement('div'); bar.className = 'bar' + (c === top ? ' top' : '') + (c === label ? ' true' : '') + (hov ? ' hov' : '');
    const i = document.createElement('i'); i.style.width = (prob[c] * 100) + '%'; bar.append(i);
    const v = document.createElement('div'); v.className = 'v'; v.textContent = (prob[c] * 100).toFixed(1) + '%';
    [lab, bar, v].forEach((x) => x.addEventListener('pointerenter', () => { if (!(probe && probe.kind === 'bar' && probe.c === c)) { probe = { kind: 'bar', c }; renderAll(); } }));
    el.append(lab, bar, v);
  }
  el.addEventListener('pointerleave', () => { probe = null; renderAll(); });
  return el;
}

function renderBackward() {
  const pipe = $('bwd'); pipe.innerHTML = '';
  if (cur.phase < 2) { pipe.append(arrow(cur.phase < 1 ? 'after the forward pass' : '←  Step runs the backward pass', true)); return; }
  const B = cur.back, S = cur.S; const isSrc = (layer, idx) => isW(layer) && probe.k === idx;
  const mapsCol = (title, maps, n, st, sub, cols, px, srcOf) => { const [d, s] = stage(title, sub, cols); const sc = Math.max(...maps.map(maxAbs));
    maps.forEach((m, idx) => { const cv = canvas(px); const own = probe && probe.kind === 'back' && probe.stage === st && probe.ch === idx;
      drawGrid(cv, m, n, n, { mode: 'div', scale: sc, outline: !!(srcOf && srcOf(idx)), rect: own ? [probe.r, probe.r, probe.c, probe.c] : null }); hoverable(cv, ([r, c]) => ({ kind: 'back', stage: st, r, c, ch: idx })); s.append(cv); }); return d; };
  pipe.append(mapsCol('∂L/∂conv 1', B.conv1, S.O, 'conv1', 'gated by ReLU', 2, 56, (i) => isSrc('bank', i)), arrow('←'));
  pipe.append(mapsCol('∂L/∂pool 1', B.pool1, S.Q, 'pool1', 'summed over 8 kernels', 2, 40), arrow('←'));
  pipe.append(mapsCol('∂L/∂conv 2', B.conv2, S.Q, 'conv2', 'gated by ReLU', 4, 40, (i) => isSrc('l2', i)), arrow('←'));
  pipe.append(mapsCol('∂L/∂pool 2', B.pool2, S.Q2, 'pool2', 'through the max of each 2×2', 4, 32), arrow('←'));
  { const [d, s] = stage('∂L/∂gap', 'a 16th to each cell'); const cv = canvas(36, 8 * 36); drawGrid(cv, B.gap, 1, C2, { mode: 'div', scale: maxAbs(B.gap), labels: true, rect: probe && probe.kind === 'gap' ? [probe.f, probe.f, 0, 0] : null }); hoverable(cv, ([r]) => ({ kind: 'gap', f: r })); s.append(cv); pipe.append(d, arrow('←')); }
  { const [d, s] = stage('∂L/∂scores = p − y', 'where the backward pass starts'); const cv = canvas(NC * 30, 30);
    const c = probe && (probe.kind === 'bar' || isW('W') || isW('B')) ? probe.c : null;
    drawGrid(cv, B.scores, NC, 1, { mode: 'div', scale: 1, labels: true, rect: c != null ? [0, 0, c, c] : null }); hoverable(cv, ([, c]) => ({ kind: 'bar', c }));
    const lab = document.createElement('div'); lab.className = 'cap'; lab.textContent = 'classes 0 to 9'; s.append(cv, lab); pipe.append(d); }
}

function renderTrips() {
  const el = $('trips'); el.innerHTML = '';
  const g = gradShown(), w = shownWeights();
  const group = (title) => { const d = document.createElement('div'); d.className = 'trip'; const t = document.createElement('div'); t.className = 't'; t.textContent = title; d.append(t); const gr = document.createElement('div'); gr.className = 'g'; d.append(gr); el.append(d); return gr; };
  const cell = (gr, label, data, cols, rows, scale, px, make, hl) => { const wrap = document.createElement('div'); const cv = canvas(cols * px, rows * px); drawGrid(cv, data, cols, rows, { mode: 'div', scale, labels: px >= 22, rect: hl }); hoverable(cv, make); const s = document.createElement('span'); s.textContent = label; wrap.append(cv, s); gr.append(wrap); };
  const triplet = (gr, wData, gData, cols, rows, px, make, hl, sw, sg) => { cell(gr, 'w', wData, cols, rows, sw, px, make, hl); if (g) { cell(gr, '∂L/∂w', gData, cols, rows, sg, px, make, hl); cell(gr, 'w′', wData.map((v, q) => v - lr * gData[q]), cols, rows, sw, px, make, hl); } };
  // layer 1: four 3×3 kernels
  const sB = Math.max(...w.bank.map(maxAbs)), gB = g ? Math.max(...g.bank.map(maxAbs)) : 1;
  w.bank.forEach((k, i) => { const hl = isW('bank') && probe.k === i ? [Math.floor(probe.q / 3), Math.floor(probe.q / 3), probe.q % 3, probe.q % 3] : null;
    triplet(group(`layer 1, kernel ${i + 1}`), k, g && g.bank[i], 3, 3, 24, ([a, b]) => ({ kind: 'w', layer: 'bank', k: i, q: a * 3 + b }), hl, sB, gB); });
  // layer 2: each kernel as its four 3×3 slices side by side (12 × 3 cells)
  const flat2 = (ks) => { const out = new Float32Array(12 * 3); ks.forEach((k, s) => { for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) out[a * 12 + s * 3 + b] = k[a * 3 + b]; }); return out; };
  const sL = Math.max(...w.l2.flat().map(maxAbs)), gL = g ? Math.max(...g.l2.flat().map(maxAbs)) : 1;
  w.l2.forEach((ks, i) => { const hl = isW('l2') && probe.k === i ? [Math.floor(probe.q / 3), Math.floor(probe.q / 3), probe.s * 3 + probe.q % 3, probe.s * 3 + probe.q % 3] : null;
    triplet(group(`layer 2, kernel ${i + 1} (4 slices)`), flat2(ks), g && flat2(g.l2[i]), 12, 3, 12, ([a, col]) => ({ kind: 'w', layer: 'l2', k: i, s: Math.floor(col / 3), q: a * 3 + col % 3 }), hl, sL, gL); });
  // FC: W as 10 × 8, B as 10 × 1
  const flatW = (M) => { const out = new Float32Array(C2 * NC); for (let f = 0; f < C2; f++) for (let c = 0; c < NC; c++) out[f * NC + c] = M[f][c]; return out; };
  const sW = maxAbs(flatW(w.W)), gW = g ? maxAbs(flatW(g.W)) : 1;
  triplet(group('FC weights W (rows: averages, cols: classes)'), flatW(w.W), g && flatW(g.W), NC, C2, 14, ([f, c]) => ({ kind: 'w', layer: 'W', f, c }), isW('W') ? [probe.f, probe.f, probe.c, probe.c] : null, sW, gW);
  triplet(group('FC biases b'), Float32Array.from(w.B), g && Float32Array.from(g.B), NC, 1, 14, ([, c]) => ({ kind: 'w', layer: 'B', c }), isW('B') ? [0, 0, probe.c, probe.c] : null, maxAbs(w.B), g ? maxAbs(g.B) : 1);
}

function readout() {
  const ro = $('readout'); const p = probe, S = cur.S, g = gradShown();
  const upd = (wv, gv) => `w′ = w − lr·∂L/∂w = ${fmt(wv)} − ${fmt(lr)}×${fmt(gv)} = <span class="sum">${fmt(wv - lr * gv)}</span>`;
  const avgNote = batch > 1 && collected.length > 1 ? ` (gradient averaged over ${collected.length} digits so far)` : '';
  if (!p) {
    ro.innerHTML = cur.phase === 0 ? 'A digit is loaded. Step runs the forward pass.'
      : cur.phase === 1 ? `Forward done: the network says ${cur.pred}, the label is ${cur.label}. loss = −log p[${cur.label}] = −log ${fmt(S.prob[cur.label])} = <span class="sum">${fmt(cur.loss)}</span>. Step runs the backward pass.`
      : cur.phase === 2 ? `Backward done: every weight has its gradient${avgNote}. Step applies w′ = w − ${fmt(lr)}·∂L/∂w to all ${4 * 9 + 8 * 4 * 9 + 8 * 10 + 10} weights at once.`
      : cur.applied ? 'Updated: the weights moved. Step loads the next digit.' : `Gradient collected (batch ${collected.length}/${batch}); the weights move when the batch is complete. Step loads the next digit.`;
    return;
  }
  if (p.kind === 'map') { const name = p.stage.replace(/([0-9])/, ' $1'); const maps = { conv1: S.a1, pool1: S.p1, conv2: S.a2, pool2: S.p2 }[p.stage]; const n = p.stage === 'conv1' ? S.O : p.stage === 'pool2' ? S.Q2 : S.Q;
    const R = footprint({ stage: p.stage, rect: [p.r, p.r, p.c, p.c], ch: p.ch }, S); const [r0, r1, c0, c1] = R.input.rect;
    ro.innerHTML = `<b>${name} map ${p.ch + 1}</b> cell (${p.r}, ${p.c}) = ${fmt(maps[p.ch][p.r * n + p.c])}. Sees input rows ${r0}–${r1}, cols ${c0}–${c1}: a <b>${r1 - r0 + 1}×${c1 - c0 + 1}</b> patch.`; return; }
  if (p.kind === 'back') { const name = p.stage.replace(/([0-9])/, ' $1'); const n = p.stage === 'conv1' ? S.O : p.stage === 'pool2' ? S.Q2 : S.Q; const v = cur.back[p.stage][p.ch][p.r * n + p.c];
    ro.innerHTML = `<b>∂L/∂${name} map ${p.ch + 1}</b> cell (${p.r}, ${p.c}) = ${fmt(v)}: how much the loss would change per unit change of that activation. ${v === 0 ? 'Zero: ReLU was off there, or max-pool did not pick it.' : ''}`; return; }
  if (p.kind === 'gap') { ro.innerHTML = `<b>global average ${p.f + 1}</b> = mean of pool-2 map ${p.f + 1} = ${fmt(S.gap[p.f])}` + (cur.phase >= 2 ? `. ∂L/∂gap[${p.f + 1}] = Σ_c W[${p.f + 1},c]·(p[c] − y[c]) = <span class="sum">${fmt(cur.back.gap[p.f])}</span>, spread as a 16th to each cell of the map.` : '.'); return; }
  if (p.kind === 'bar') { const y = p.c === cur.label ? 1 : 0; ro.innerHTML = `<b>class ${p.c}</b>${y ? ' (the true class)' : ''}: p = ${fmt(S.prob[p.c])}` + (cur.phase >= 2 ? `. ∂L/∂score[${p.c}] = p − y = ${fmt(S.prob[p.c])} − ${y} = <span class="sum">${fmt(S.prob[p.c] - y)}</span>` : '') + '.'; return; }
  if (p.kind === 'w') {
    const w = weights;
    if (p.layer === 'W') { const wv = w.W[p.f][p.c]; const y = p.c === cur.label ? 1 : 0; ro.innerHTML = `<b>W[row ${p.f + 1}, class ${p.c}]</b> = ${fmt(wv)}: links global average ${p.f + 1} to class ${p.c}.` + (g ? ` ∂L/∂W = gap[${p.f + 1}] × (p[${p.c}] − y[${p.c}]) = ${fmt(S.gap[p.f])} × (${fmt(S.prob[p.c])} − ${y}) = <span class="sum">${fmt(g.W[p.f][p.c])}</span>${avgNote}. ${upd(wv, g.W[p.f][p.c])}` : ''); return; }
    if (p.layer === 'B') { const wv = w.B[p.c]; const y = p.c === cur.label ? 1 : 0; ro.innerHTML = `<b>b[${p.c}]</b> = ${fmt(wv)}.` + (g ? ` ∂L/∂b[${p.c}] = p[${p.c}] − y[${p.c}] = ${fmt(S.prob[p.c])} − ${y} = <span class="sum">${fmt(g.B[p.c])}</span>${avgNote}. ${upd(wv, g.B[p.c])}` : ''); return; }
    const a = Math.floor(p.q / 3), b = p.q % 3;
    if (p.layer === 'bank') { const wv = w.bank[p.k][p.q]; ro.innerHTML = `<b>layer-1 kernel ${p.k + 1}</b>, cell (${a}, ${b}) = ${fmt(wv)}.` + (g ? ` ∂L/∂w = Σ over ${S.O * S.O} positions of ∂L/∂conv1_${p.k + 1}[i,j] × input[i+${a}−1, j+${b}−1] (the two outlined maps) = <span class="sum">${fmt(g.bank[p.k][p.q])}</span>${avgNote}. ${upd(wv, g.bank[p.k][p.q])}` : ''); return; }
    const wv = w.l2[p.k][p.s][p.q]; ro.innerHTML = `<b>layer-2 kernel ${p.k + 1}, slice ${p.s + 1}</b>, cell (${a}, ${b}) = ${fmt(wv)}.` + (g ? ` ∂L/∂w = Σ over ${S.Q * S.Q} positions of ∂L/∂conv2_${p.k + 1}[i,j] × pool1_${p.s + 1}[i+${a}−1, j+${b}−1] (the two outlined maps) = <span class="sum">${fmt(g.l2[p.k][p.s][p.q])}</span>${avgNote}. ${upd(wv, g.l2[p.k][p.s][p.q])}` : '');
  }
}

// ---------- play band ----------
const niceStep = (xmax) => { const raw = Math.max(1, xmax / 5); const p = Math.pow(10, Math.floor(Math.log10(raw))); const m = raw / p; return (m < 2 ? 2 : m < 5 ? 5 : 10) * p; };
function drawCurve(cv, series, { ymin, ymax, yTicks, yfmt, xmax }) {
  const ctx = cv.getContext('2d'); const W = cv.width, H = cv.height; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
  const L = 44, Rm = 8, T = 8, Bm = 22; const x = (v) => L + (v / Math.max(1, xmax)) * (W - L - Rm), y = (v) => T + (1 - (clamp(v, ymin, ymax) - ymin) / (ymax - ymin)) * (H - T - Bm);
  ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`; ctx.fillStyle = cssVar('--ink2');
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const t of yTicks) { ctx.strokeStyle = cssVar('--line2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, y(t)); ctx.lineTo(W - Rm, y(t)); ctx.stroke(); ctx.fillText(yfmt(t), L - 5, y(t)); }
  ctx.strokeStyle = cssVar('--line'); ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, H - Bm); ctx.lineTo(W - Rm, H - Bm); ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top'; const st = niceStep(xmax); for (let v = 0; v <= xmax; v += st) ctx.fillText(String(v), x(v), H - Bm + 4);
  for (const s of series) { if (!s.pts.length) continue; ctx.globalAlpha = s.alpha ?? 1; ctx.strokeStyle = ctx.fillStyle = s.color; ctx.lineWidth = s.width ?? 1.5;
    if (s.dots) for (const [px, py] of s.pts) ctx.fillRect(x(px) - 1, y(py) - 1, 2, 2);
    else { ctx.beginPath(); s.pts.forEach(([px, py], i) => (i ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py)))); ctx.stroke(); }
    ctx.globalAlpha = 1; }
}
function renderPlay() {
  $('counts').textContent = `digits seen ${digitsSeen} · pass ${passes + 1} · seed ${seed} · held-out accuracy ${lastTest == null ? 'n/a' : (lastTest * 100).toFixed(1) + '%'}`;
  const xmax = Math.max(200, digitsSeen); const accent = cssVar('--accent'), ink = cssVar('--ink');
  drawCurve($('cLoss'), [{ pts: lossRaw, color: accent, dots: true, alpha: .35 }, { pts: lossEma, color: ink, width: 2 }], { ymin: 0, ymax: 4, yTicks: [0, 1, 2, 3, 4], yfmt: String, xmax });
  drawCurve($('cAcc'), [{ pts: accTrain, color: accent, width: 1.2 }, { pts: accTest, color: ink, width: 2.5 }], { ymin: 0, ymax: 1, yTicks: [0, .5, 1], yfmt: (v) => (v * 100) + '%', xmax });
  const sn = $('snaps'); sn.innerHTML = ''; const all = [...snaps, { at: 'now', bank: weights.bank }]; const sc = Math.max(...all.flatMap((s) => s.bank.map(maxAbs)));
  for (const s of all) { const d = document.createElement('div'); d.textContent = s.at === 'now' ? 'now' : `${s.at} digits`; s.bank.forEach((k) => { const cv = canvas(30); drawGrid(cv, k, 3, 3, { mode: 'div', scale: sc }); d.prepend(cv); }); sn.append(d); }
  const th = $('thumbs'); th.innerHTML = '';
  for (const r of recent) { const d = document.createElement('div'); const cv = canvas(32); drawGrid(cv, r.input, N, N, {}); const t = document.createElement('div'); const ok = r.pred === r.label; t.className = ok ? 'ok' : 'no'; t.textContent = `${r.label}${ok ? '✓' : '✗'}`; t.title = `label ${r.label}, guessed ${r.pred}`; d.append(cv, t); th.append(d); }
}

function renderAll() { renderPhase(); renderForward(); renderBackward(); renderTrips(); readout(); renderPlay(); }

// ---------- controls ----------
const lrOf = (v) => 0.001 * Math.pow(500, v / 100);
$('stepBtn').onclick = () => { if (playing) setPlaying(false); step(); };
$('nextBtn').onclick = () => { if (playing) setPlaying(false); nextDigit(); };
$('playBtn').onclick = () => setPlaying(!playing);
$('speed').onchange = (e) => { speed = +e.target.value; };
$('lr').oninput = (e) => { lr = +lrOf(+e.target.value).toPrecision(2); $('lrv').textContent = String(lr); if (!playing) renderAll(); };
$('batch').onchange = (e) => { batch = +e.target.value; collected = []; shown = null; if (cur && cur.phase === 2) cur.phase = 1; renderAll(); };
$('resetBtn').onclick = () => { setPlaying(false); reset({ newSeed: true }); };
$('edgesBtn').onclick = () => { setPlaying(false); reset({ layer1: 'edges' }); };
document.addEventListener('keydown', (e) => { if (!data || e.target.closest('input,select,button,textarea')) return; if (e.code === 'Space') { e.preventDefault(); $('stepBtn').click(); } else if (e.code === 'Enter') { e.preventDefault(); $('nextBtn').click(); } });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => data && renderAll());
installTerms({ selectors: '.lede, .head p, .note', tip: $('tip'), gloss: $('gloss') });

// ---------- init ----------
(async () => {
  try { const res = await fetch('../data/mnist16.bin'); if (!res.ok) throw new Error('HTTP ' + res.status); data = parseMnist16(await res.arrayBuffer()); }
  catch (e) { $('status').textContent = `Could not load data/mnist16.bin (${e.message}). Serve the repo root over http; the file is 1 MB.`; return; }
  testIdx = Array.from({ length: data.test.n }, (_, i) => i);
  $('status').remove();
  reset();
})();
```

- [ ] **Step 4: Check it parses and loads**

Run: `awk '/<script type="module">/{f=1;next} /<\/script>/{f=0} f' cnn-train/index.html > <scratch>/cnn-train.mjs && node --check <scratch>/cnn-train.mjs`
Expected: no output.

Then serve the repo root (`python -m http.server 8768 --bind 127.0.0.1` in the background) and dump the page with headless Edge (`--virtual-time-budget=8000`, the data fetch needs it). Expected in the DOM: no `CONSOLE` errors in stderr; `#status` gone; `#digitInfo` reads `training digit N · label d`; the forward pipe holds one canvas (the digit) and the message "Step runs the forward pass"; `#counts` contains `digits seen 0 · pass 1 · seed 1 · held-out accuracy` with a percentage near 10%; `#snaps` has one column "0 digits"; `#gloss` has 25 entries; dotted terms in the lede include "loss", "gradient", "learning rate".

- [ ] **Step 5: Drive the step band headless**

Make a scratch copy of the page with the seeded `Math.random` shim and, appended after `reset();` inside the init function, `step(); step(); step();` (forward, backward, update). Dump it. Expected: the phase strip's four `b` elements all carry `done`; `#readout` says "Updated: the weights moved"; the forward pipe has 1 + 4 + 4 + 8 + 8 + 1 + 1 canvases plus 10 bars with one `.true`; the backward pipe has 4 + 4 + 8 + 8 + 1 + 1 canvases; `#trips` has 4 + 8 + 2 groups, each with one canvas (the gradient is cleared after the update); `#counts` says `digits seen 1`; `#thumbs` has one thumbnail.

Make a second copy appending `step(); step();` only (forward, backward) and `probe = { kind: 'w', layer: 'W', f: 2, c: 3 }; renderAll();`. Expected: `#readout` contains `W[row 3, class 3]`, `∂L/∂W = gap[3] × (p[3] − y[3])` and `w′ = w − lr·∂L/∂w`; every `.trip` group has three canvases; the bar for class 3 has class `hov`.

Make a third copy appending `for (let i = 0; i < 300; i++) trainOne(); renderAll();`. Expected: `#counts` says `digits seen 300` and a held-out accuracy well above 10%; the last entry of `lossEma` is below the first (expose it for the check by appending `document.body.dataset.ema = lossEma[0][1] + ',' + lossEma[lossEma.length - 1][1];`); `#snaps` has columns for 0 and 100 and 300 digits plus "now"; `#thumbs` has 20 thumbnails.

Take screenshots of both bands at 1400 px and 1000 px widths with the step-band copy and the 300-digit copy; look at them.

- [ ] **Step 6: Commit**

```bash
git add ui/terms.js cnn-train/index.html
git commit -m "feat: training page: one step in full, then many steps with curves and kernels over time"
```

---
### Task 7: Hand-off of trained weights to the explorer pages

**Files:**
- Create: `ui/trained.js`
- Modify: `cnn-train/index.html` (the `#handoff` container and two handlers), `cnn-2d/index.html`, `cnn-3d/index.html`, `index.html`

**Interfaces:**
- Produces: `KEY = 'dl-explorer.weights.v1'`; `saveTrained(obj)`; `loadTrained() → obj | null` (null when absent, unparsable, or not `{ version: 1, N: 16, bank, l2, W, B, meta }`); `clearTrained()`; `showTrainedBanner(anchorElement, obj)` inserts a `.banner` div after the anchor. The stored object: `{ version: 1, N: 16, bank, l2, W, B, meta: { digits, passes, testAcc, lr, date } }`.

- [ ] **Step 1: The shared module**

Create `ui/trained.js`:

```js
// ui/trained.js — hand-off of weights trained on the training page to the explorer pages, through local storage.
export const KEY = 'dl-explorer.weights.v1';

export function saveTrained(o) { localStorage.setItem(KEY, JSON.stringify(o)); }

/** The stored weights, or null when there are none or they are not in the expected shape. */
export function loadTrained() {
  try {
    const s = localStorage.getItem(KEY); if (!s) return null;
    const o = JSON.parse(s);
    return o && o.version === 1 && o.N === 16 && Array.isArray(o.bank) && Array.isArray(o.l2) && Array.isArray(o.W) && Array.isArray(o.B) && o.meta ? o : null;
  } catch { return null; }
}

export function clearTrained() { try { localStorage.removeItem(KEY); } catch { /* storage blocked: nothing to clear */ } }

/** A banner after `anchor` saying the page uses trained weights; its link removes them and reloads. */
export function showTrainedBanner(anchor, o) {
  const d = document.createElement('div'); d.className = 'banner';
  const acc = o.meta.testAcc == null ? '' : `, held-out accuracy ${Math.round(o.meta.testAcc * 100)}%`;
  d.innerHTML = `Using weights trained on the <a href="../cnn-train/">training page</a> (${o.meta.digits} digits${acc}). <a href="#" data-clear>Back to the hand-picked kernels</a>.`;
  d.querySelector('[data-clear]').onclick = (e) => { e.preventDefault(); clearTrained(); location.reload(); };
  anchor.after(d);
}
```

- [ ] **Step 2: Buttons on the training page**

In `cnn-train/index.html`, replace `<div class="ctrls" id="handoff"></div>` with:

```html
  <div class="ctrls" id="handoff"><button id="useBtn">Use these weights in the 2D / 3D explorers</button><button id="dlBtn">Download JSON</button><span class="cap" id="useMsg"></span></div>
```

Add `import { saveTrained } from '../ui/trained.js';` after the other imports, and these handlers before `// ---------- init ----------`:

```js
const bundle = () => ({ version: 1, N, bank: weights.bank, l2: weights.l2, W: weights.W, B: weights.B, meta: { digits: digitsSeen, passes, testAcc: lastTest, lr, date: new Date().toISOString() } });
$('useBtn').onclick = () => { try { saveTrained(bundle()); $('useMsg').innerHTML = `Saved in this browser. Open the <a href="../cnn-2d/">2D</a> or <a href="../cnn-3d/">3D</a> explorer: it will use these weights until you choose "back to the hand-picked kernels" there.`; } catch (e) { $('useMsg').textContent = 'Could not save in this browser: ' + e.message; } };
$('dlBtn').onclick = () => { const blob = new Blob([JSON.stringify(bundle(), null, 1)], { type: 'application/json' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'cnn16-trained.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); };
```

- [ ] **Step 3: The 2D page loads them**

In `cnn-2d/index.html`:

1. Add to the `<style>` block, after the `.note{…}` rule: `.banner{font-size:13.5px;background:var(--accent-soft);border:1px solid var(--accent);border-radius:8px;padding:8px 12px;margin:0 0 14px;max-width:78ch}` and `.banner a{color:var(--accent)}`.
2. Add `import { loadTrained, showTrainedBanner } from '../ui/trained.js';` after the `ui/grid.js` import.
3. In stage 5's `.head p`, wrap the sentence `Layer-2 and FC weights here are random, so the prediction is arbitrary — training is what moves these weights until the right bar wins.` in `<span id="wnote">…</span>`.
4. Replace the init line `inPresets['7'](); rerollL2(); rerollFC(); syncKgrid(); renderAll();` with:

```js
const trained = loadTrained();
if (trained) { trained.bank.forEach((k, i) => { bank[i] = k.slice(); }); l2 = trained.l2; W = trained.W; B = trained.B; showTrainedBanner(document.querySelector('h1'), trained);
  $('wnote').textContent = `Layer-2 and FC weights here were trained on the training page (${trained.meta.digits} digits), so the prediction means something for digits; the mechanics are the same either way.`; }
else { rerollL2(); rerollFC(); }
inPresets['7'](); syncKgrid(); renderAll();
```

- [ ] **Step 4: The 3D page loads them**

In `cnn-3d/index.html`:

1. Add the same two `.banner` rules to the `<style>` block after `.hint{…}`.
2. Add `import { loadTrained, showTrainedBanner } from '../ui/trained.js';` after the `ui/terms.js` import.
3. Wrap the text of the weights hint, `Layer-1 kernels are Sobel-x, Sobel-y, Laplacian, diagonal. Layer 2 and FC are random: the prediction is arbitrary, the mechanics are real.`, in `<span id="wnote">…</span>`.
4. Replace the init line `inPresets['7'](); rerollL2(); rerollFC(); setSweep('off'); resize(); recompute(true); applyCam(); loop();` with:

```js
const trained = loadTrained();
if (trained) { trained.bank.forEach((k, i) => { bank[i] = k.slice(); }); l2 = trained.l2; W = trained.W; B = trained.B; showTrainedBanner(document.querySelector('h1'), trained);
  $('wnote').textContent = `All weights were trained on the training page (${trained.meta.digits} digits): the prediction means something for digits, and the mechanics are the same either way.`; }
else { rerollL2(); rerollFC(); }
inPresets['7'](); setSweep('off'); resize(); recompute(true); applyCam(); loop();
```

- [ ] **Step 5: The landing page**

In `index.html`, add after the 3D item:

```html
<li><a href="cnn-train/">CNN, training</a> — real digits stream in; one step in full (forward, loss, backward, update), then the arc of many: curves, kernels learning edges, accuracy on held-out digits. Trained weights load into the other two pages.</li>
```

- [ ] **Step 6: Check**

`node --check ui/trained.js`, then the three module scripts extracted with `awk` and `node --check`.

Headless, serving the repo root: (a) the 2D and 3D pages with no stored key: dump with the seeded shim and compare to HEAD's copies with scripts stripped; the only differences allowed are the `<span id="wnote">` wrapper tags. (b) A scratch copy of each explorer page with `<script>localStorage.setItem('dl-explorer.weights.v1', JSON.stringify({version:1,N:16,bank:[[1,0,-1,2,0,-2,1,0,-1],[0,1,0,1,-4,1,0,1,0],[1,1,1,0,0,0,-1,-1,-1],[1,0,0,0,1,0,0,0,1]],l2:Array.from({length:8},()=>Array.from({length:4},()=>Array(9).fill(0.1))),W:Array.from({length:8},()=>Array(10).fill(0.2)),B:Array(10).fill(0),meta:{digits:1234,passes:0,testAcc:0.87,lr:0.05,date:'2026-10-12'}}))</script>` injected after `<head>`: expected a `.banner` div after the `h1` containing `1234 digits` and `87%`, and `#wnote` starting with "Layer-2 and FC weights here were trained" (2D) or "All weights were trained" (3D); no console errors. (c) The training page with `$('useBtn').click(); document.body.dataset.saved = String(!!localStorage.getItem('dl-explorer.weights.v1'));` appended inside the init after `reset();`: expected `data-saved="true"` on the body and `#useMsg` containing "Saved in this browser".

- [ ] **Step 7: Commit**

```bash
git add ui/trained.js cnn-train/index.html cnn-2d/index.html cnn-3d/index.html index.html
git commit -m "feat: trained weights load into the 2D and 3D explorers"
```

---

### Task 8: Whole-site check, documentation, deploy

**Files:**
- Modify: `project_notes.md`, `HANDOFF.md`, `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-10-12-cnn-training-page-design.md` only if a measured value (init scales, learning rate, gradient-check tolerance, one-pass accuracy) differs from what it states.

- [ ] **Step 1: Everything from one server**

Serve the repo root and dump all four pages (landing, 2D, 3D, training) headless with console logging. Expected: no page errors; the landing page has three links; the training page reports a held-out accuracy and 25 glossary entries. Run `node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs`: all three print their OK line.

- [ ] **Step 2: Screenshots**

With the step-band copy from Task 6 (forward + backward done, an FC weight probed) and the 300-digit play copy, take screenshots at 1400×1600 and 1000×1600 and look at them: the forward and backward rows line up, the triplets are readable, the curves have axes and ticks, the kernel strip shows 0, 100, 300 and now, nothing overflows the page horizontally (the pipes may scroll inside their own box), and the dark theme (add `data-theme="dark"` on `<html>` in a copy) has no unreadable text.

- [ ] **Step 3: Project notes, handoff, conventions**

Use the `project-notes` skill (Update job). As built: Overview (three pages now), Method (the training page: what each band shows; `core/train.js` and the formulas in one line each; `core/data.js` and the MN16 file; `ui/grid.js`, `ui/trained.js`; the hand-off), Results (the gradient-check numbers, the one-digit loss drop, the one-pass held-out accuracy and its run time, the browser checks and widths), Decisions (the five from the spec's "Decisions taken in the brainstorm", with Why lines), Map (every new path), Status (HANDOFF step 3's "trained weights" and "shared state" ideas done, in this form; what is left: 28×28, MNIST presets on the explorer pages, the arrow overlay). Keep Intent untouched. In `HANDOFF.md`, mark under "Ideas discussed, not built" that trained LeNet-style weights and shared state arrived as live in-browser training on 2026-10-12. In `CLAUDE.md`, add `data/`, `tools/` and `docs/` to the Layout list and change "Trained weights are a planned addition, not a given" to say weights trained on the training page can be loaded into the explorer pages and the pages say which they use.

- [ ] **Step 4: Commit and deploy**

```bash
git add project_notes.md HANDOFF.md CLAUDE.md docs/superpowers/specs/2026-10-12-cnn-training-page-design.md
git commit -m "docs: record the training page in project notes, handoff and conventions"
git push origin main
```

Then `gh run watch` the Pages deploy and fetch `https://smooth-cactus0.github.io/Visualising-AI/cnn-train/` and `.../data/mnist16.bin` with a cache-busting query: expected HTTP 200 for both and 1,028,016 bytes for the data file. Report the push and the URL in the final message.

---

## Plan self-review

- Spec coverage: files (Tasks 1–7), data format (1, 2), core maths and every gradient formula (3), SGD, batches, predict, accuracy, shuffle (4), the step band with its phases, rows, triplets, hover readouts, keyboard, reduced motion (6), the play band with controls, curves, snapshots, thumbnails (6), terms (6), hand-off and the explorer banners and wording (7), checks (3, 4, 5, 6, 7, 8), docs (8). Left-for-later items have no task, as intended.
- Placeholders: none; every code step carries its code.
- Names used across tasks: `parseMnist16`, `image` (2 → 4, 6); `initWeights`, `lossAndGrad`, `sgdStep`, `averageGrads`, `accuracy`, `shuffledOrder` (3–4 → 6); `drawGrid`, `cellAt`, `fmt`, `maxAbs`, `cssVar` (5 → 6); `saveTrained`, `loadTrained`, `showTrainedBanner` (7); page state `weights`, `digitsSeen`, `passes`, `lastTest`, `lr`, `N`, `$` (6 → 7).
