# ViT Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A page, `vit/`, that shows what a Vision Transformer computes, on the CNN pages' 16×16 digits: patches become tokens, position is added, one attention head is written out, then the MLP and residuals, and the tokens are averaged into a prediction. It runs on trained weights, its green hover is graded by attention, and an overview line stays pinned at the top.

**Architecture:** The model's maths is pure JS in `core/vit.js`, tested in Node against finite differences: forward, backward, updates, and `vitTrace`, the rule that decides the green. A Node script trains the model on `data/mnist16.bin` and writes `models/vit16.json`, which the page fetches. The page `vit/index.html` builds the overview and the five section pictures from one small column builder over `ui/grid.js` canvases, all redrawn by one `render()`. Spec: `docs/superpowers/specs/2026-10-28-vit-explorer-design.md`.

**Tech Stack:** Plain HTML + ES modules (no build step), Node 24 (`node file.mjs`), Python 3 `http.server` to serve pages, headless Edge for the page checks.

## Global Constraints

- No build step: plain HTML with `<script type="module">`. No external scripts; the Google Fonts stylesheet only, as on the other pages.
- `core/` stays DOM-free; `ui/` holds the shared DOM helpers. After the `ui/` changes the three CNN pages must render identically (seeded DOM dump, Task 4).
- The model is fixed:
  - 16×16 input, 4×4 patches → 16 tokens × 8 numbers;
  - pre-LayerNorm, 2 heads of 4, MLP 8 → 32 → 8 with GELU (tanh form), one encoder block;
  - average of the 16 tokens → LayerNorm → 8×10 → softmax;
  - 1,242 weights, and no class token.
- Colour grammar:
  - yellow = the window being computed (a patch, a score);
  - green = computed from, graded after attention: the hovered token's own patch at 1, every other token at 0.9 × its attention ÷ the largest other attention;
  - learned weights are tinted (`--wtint`, accent border), with each bias as a row beneath its matrix.
- Wherever a prediction appears, the page says whether the weights are trained (with the held-out accuracy read from the weights file) or random.
- Theme-aware (the CNN pages' colour tokens) and responsive: the overview is not pinned below 700 px, and the page never overflows sideways. Reduced motion is respected. Canvases are sized in CSS as well as in pixels.
- Every technical term gets a dotted definition (`VIT_TERMS`) and an entry in the Terms section. No tables on the page.
- NZ English in all copy.
- Files are LF; ignore Git's CRLF warnings. In this environment, heredocs lose backslashes: write files with the Write tool, never with `cat <<EOF`.
- Conventional Commits, with one local commit per task. The push to `main` happens once, in Task 6, after the checks; pushing `main` is authorised for this repo.
- Check scripts and their outputs live in `.superpowers/check/`, which is git-ignored, not in the repo.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `core/vit.js` | Constants and `SHAPES`, `init`, `forward`, `tensor`, loss and gradients, SGD and Adam, `predict`, `accuracy` | 1 |
| `core/vit.js` (end) | `vitTrace`, the green | 2 |
| `core/vit.test.mjs` | Shapes, attention rows, LayerNorm, patch convolution, shuffling, finite differences, updates; then `vitTrace` | 1, 2 |
| `tools/train_vit.mjs` | Training runs and the weights file | 3 |
| `models/vit16.json` | The shipped weights and how they were made | 3 |
| `ui/grid.js` | Options `glow`, `win`, `blocks`, `lines: false` | 4 |
| `ui/terms.js` | Per-page term lists; `VIT_TERMS` | 4 |
| `.superpowers/check/domcheck.mjs` | The CNN pages' DOM, before and after | 4 |
| `vit/index.html` | The page | 5 |
| `.superpowers/check/vitcheck.mjs`, `scen_*.js` | Page checks in headless Edge | 5 |
| `index.html` | Card, badge, chip, lede, schematic, green key | 6 |
| `CLAUDE.md`, `HANDOFF.md`, `project_notes.md` | Record the result | 6 |

**Context for the executor.** Every file in this plan was prototyped and run in a scratch folder on 2026-10-28, before the plan was written; the code below is that code. What the prototype measured:
- **Tests:** they pass. The gradient check's maximum normalised error was 1.5e-8 with the default settings and 1.7e-6 with position off and the patches shuffled.
- **Training** (10 passes, seed 1, batch 1, held-out accuracy):
  - SGD: lr 0.003 → 82.0%, 0.01 → 80.2%, 0.03 → 73.4%, 0.1 → 29.7%, 0.3 → 15.6% (diverged);
  - Adam: lr 0.001 → 83.4%, 0.003 → 84.9%, 0.01 → 70.2%;
  - about 2 to 4 s a pass.
- **The page** passed the Task 5 scenarios in light and dark, at 1400, 1000 and 500 px, and with the weights file missing.
- **The CNN pages** gave an identical DOM before and after the Task 4 changes (30, 4 and 18 dotted terms).

---

### Task 1: The model: forward pass, gradients, updates

**Files:**
- Create: `core/vit.js`
- Test: `core/vit.test.mjs`

**Interfaces:**
- Consumes: `mulberry32(seed)` and `shuffledOrder(n, seed)` from `core/train.js`; `parseMnist16(bytes)` and `image(set, i)` from `core/data.js`.
- Produces (all exported from `core/vit.js`):
  - `N = 16, P = 4, T = 16, D = 8, H = 2, DH = 4, HID = 32, NC = 10`.
  - `SHAPES`: `{ key: [rows, cols] }` for every drawn tensor. The activation keys are `digit xp z0 h1 q k v att0 att1 o z1 h2 hid z2 avg avgNorm prob`. The weight keys are `Wp bp pos ln1 Wq bq Wk bk Wv bv Wo bo ln2 W1 b1 W2 b2 lnf Wh bh`; LayerNorm keys are 2×8 (row 0 the scale, row 1 the shift), and biases are 1×n.
  - `init(seed = 1)` returns weights `{ Wp: 8×16, bp: 8, pos: 16×8, ln1: { g: 8, b: 8 }, Wq, Wk, Wv, Wo: 8×8, bq, bk, bv, bo: 8, ln2, W1: 8×32, b1: 32, W2: 32×8, b2: 8, lnf, Wh: 8×10, bh: 10 }`, as plain nested arrays. Matrices are (in × out), except `Wp`, which holds 8 kernels of 16 values.
  - `forward(w, img, { position = true, order = null })` returns `S` with:
    - `img` (16×16, as the model sees it), `patches` (16×16), `xp`, `z0`, `h1`, `q`, `k`, `v` (16×8), `scores` and `att` (2 × 16×16), `heads` (2 × 16×4), `o`, `z1`, `h2` (16×8), `pre`, `hid` (16×32), `z2` (16×8), `avg`, `avgNorm` (8), `logits`, `prob` (10);
    - the internals `n1`, `n2`, `nf`, `mixed`, `mlp`;
    - `position` and `order`.
  - `tensor(key, S, w)` returns any `SHAPES` key as rows of numbers.
  - `softmax(v)`, `gelu(x)`, `patchify(img, order)`, `paramCount(w)`, `isWeights(o)`, `crossEntropy(prob, label)`.
  - `lossAndGrad(w, img, label, opts)` returns `{ S, loss, grads }`, where `grads` has the same structure as `w`.
  - `sgdStep(w, g, lr)` returns new weights. `adamInit(w)` returns `{ t, m, v }`. `adamStep(w, g, state, lr)` returns `{ w, state }`. `averageGrads(list)` returns the element-wise mean.
  - `predict(w, img, opts)` returns `{ prob, argmax }`. `accuracy(w, set, indices, opts)` returns the fraction correct.

- [ ] **Step 1: Write the failing test**

Create `core/vit.test.mjs`:

```js
import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { shuffledOrder } from './train.js';
import { N, P, T, D, H, DH, HID, NC, SHAPES, init, forward, tensor, paramCount, isWeights, crossEntropy, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, predict, accuracy } from './vit.js';
const assert = (c, m) => { if (!c) throw new Error(m); };
const close = (a, b, tol, m) => assert(Math.abs(a - b) <= tol, `${m}: ${a} vs ${b}`);

const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
const img = image(data.test, 0), label = data.test.labels[0];
const w = init(3);

// ---- weights ----
assert(paramCount(w) === 1242, `1,242 weights, got ${paramCount(w)}`);
assert(isWeights(w) && !isWeights({ ...w, Wh: w.Wh.slice(1) }) && !isWeights(null), 'isWeights');
assert(init(3).Wq[2][5] === w.Wq[2][5] && init(4).Wq[2][5] !== w.Wq[2][5], 'seeded init');
assert(w.ln1.g.every((x) => x === 1) && w.bq.every((x) => x === 0), 'LayerNorm scale 1, biases 0');

// ---- forward: shapes ----
const S = forward(w, img);
for (const [key, [R, C]] of Object.entries(SHAPES)) {
  const M = tensor(key, S, w);
  assert(M.length === R && M.every((row) => row.length === C), `${key} is ${R}×${C}`);
}
assert(S.patches.length === 16 && S.patches[0].length === 16 && S.att.length === H, 'patches and heads');
close(S.prob.reduce((a, b) => a + b, 0), 1, 1e-12, 'probabilities sum to 1');

// ---- attention rows sum to 1 ----
for (const A of S.att) for (const r of A) close(r.reduce((a, b) => a + b, 0), 1, 1e-12, 'attention row sum');

// ---- LayerNorm: mean 0, variance 1 per token before scale and shift (var / (var + ε) exactly, ε = 1e-5) ----
S.n1.forEach((n, i) => {
  const mean = (x) => x.reduce((a, b) => a + b, 0) / D, vr = (x) => { const m = mean(x); return x.reduce((a, b) => a + (b - m) ** 2, 0) / D; };
  const raw = vr(S.z0[i]);
  close(mean(n.xhat), 0, 1e-12, 'LayerNorm mean'); close(vr(n.xhat), raw / (raw + 1e-5), 1e-12, 'LayerNorm variance');
});

// ---- patch embedding = a 4×4 convolution with stride 4 ----
for (let d = 0; d < D; d++) for (let orow = 0; orow < 4; orow++) for (let ocol = 0; ocol < 4; ocol++) {
  let s = w.bp[d];
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) s += img[(orow * 4 + a) * N + ocol * 4 + b] * w.Wp[d][a * 4 + b];
  close(S.xp[orow * 4 + ocol][d], s, 1e-12, `patch conv d${d} (${orow}, ${ocol})`);
}

// ---- shuffling: no change without position, a change with it ----
const perm = shuffledOrder(16, 9);
const maxDiff = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
const off0 = forward(w, img, { position: false }), off1 = forward(w, img, { position: false, order: perm });
assert(maxDiff(off0.prob, off1.prob) < 1e-12, `position off: shuffle changes nothing (${maxDiff(off0.prob, off1.prob)})`);
const on1 = forward(w, img, { order: perm });
assert(maxDiff(S.prob, on1.prob) > 1e-6, 'position on: shuffle changes the prediction');
for (let s = 0; s < 16; s++) assert(on1.img[(s >> 2) * 4][(s & 3) * 4] === img[(perm[s] >> 2) * 64 + (perm[s] & 3) * 4], 'shuffled picture');
console.log(`forward: shapes, attention rows, LayerNorm, patch conv, shuffle ok (position off ${maxDiff(off0.prob, off1.prob).toExponential(1)}, on ${maxDiff(S.prob, on1.prob).toExponential(1)})`);

// ---- gradients against central finite differences, every weight ----
const paths = [];
(function walk(t, p) { if (typeof t === 'number') paths.push(p); else if (Array.isArray(t)) t.forEach((x, i) => walk(x, [...p, i])); else Object.keys(t).forEach((key) => walk(t[key], [...p, key])); })(w, []);
const get = (t, p) => p.reduce((o, key) => o[key], t);
const set = (t, p, v) => { get(t, p.slice(0, -1))[p[p.length - 1]] = v; };
for (const opts of [{}, { position: false, order: perm }]) {
  const { loss, grads } = lossAndGrad(w, img, label, opts);
  close(loss, crossEntropy(forward(w, img, opts).prob, label), 1e-12, 'loss');
  const h = 1e-5, errs = [];
  for (const p of paths) {
    const wp = structuredClone(w); set(wp, p, get(w, p) + h);
    const wm = structuredClone(w); set(wm, p, get(w, p) - h);
    const num = (crossEntropy(forward(wp, img, opts).prob, label) - crossEntropy(forward(wm, img, opts).prob, label)) / (2 * h);
    const ana = get(grads, p), err = Math.abs(num - ana) / (1 + Math.abs(ana));
    assert(err < 1e-5, `${p.join('.')}: numeric ${num} vs analytic ${ana}`);
    errs.push(err);
  }
  errs.sort((a, b) => a - b);
  console.log(`gradient check (${opts.order ? 'position off, shuffled' : 'default'}): ${errs.length} weights, max normalised error ${errs[errs.length - 1].toExponential(2)}, median ${errs[errs.length >> 1].toExponential(2)}`);
}

// ---- updates lower the loss on one digit ----
{
  let ws = w; const l0 = lossAndGrad(w, img, label).loss;
  for (let s = 0; s < 50; s++) ws = sgdStep(ws, lossAndGrad(ws, img, label).grads, 0.1);
  let wa = w, st = adamInit(w);
  for (let s = 0; s < 50; s++) ({ w: wa, state: st } = adamStep(wa, lossAndGrad(wa, img, label).grads, st, 0.01));
  const ls = lossAndGrad(ws, img, label).loss, la = lossAndGrad(wa, img, label).loss;
  assert(ls < 0.5 * l0 && la < 0.5 * l0, `50 steps on one digit: ${l0} → SGD ${ls}, Adam ${la}`);
  const avg = averageGrads([lossAndGrad(w, img, label).grads, lossAndGrad(w, image(data.test, 1), data.test.labels[1]).grads]);
  close(avg.Wh[0][0], (lossAndGrad(w, img, label).grads.Wh[0][0] + lossAndGrad(w, image(data.test, 1), data.test.labels[1]).grads.Wh[0][0]) / 2, 1e-15, 'averageGrads');
  assert(predict(w, img).prob.length === NC && accuracy(w, data.test, [0, 1, 2]) >= 0, 'predict, accuracy');
  console.log(`updates: one digit, 50 steps, loss ${l0.toFixed(3)} → SGD ${ls.toFixed(4)}, Adam ${la.toFixed(4)}`);
}

console.log('core/vit.test.mjs: all checks passed');
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/vit.test.mjs`
Expected: FAIL, `Error [ERR_MODULE_NOT_FOUND]: Cannot find module '…/core/vit.js'`.

- [ ] **Step 3: Write the model**

Create `core/vit.js`:

```js
// core/vit.js — the toy Vision Transformer: forward pass, backward pass, updates and the green trace. Pure, no DOM.
// 16×16 digit → 16 patches of 4×4 → 8 numbers each, plus position → one encoder block (LayerNorm, 2 heads of 4,
// Wo, residual; LayerNorm, MLP 8 → 32 → 8 with GELU, residual) → average of the 16 tokens → LayerNorm → 10 classes.
// A token is a row and matrices are (in × out), so q = h · Wq + bq. The patch kernels Wp are stored as 8 rows of 16
// (kernel d = row d), like the CNN's bank. Token i is patch i, in reading order.
import { mulberry32 } from './train.js';
import { image } from './data.js';

export const N = 16, P = 4, T = 16, D = 8, H = 2, DH = 4, HID = 32, NC = 10;

/** Rows × columns of every tensor the page draws, by key. Weights come with their biases (1 row) and LayerNorm
 *  parameters (row 0 scale, row 1 shift). */
export const SHAPES = {
  digit: [N, N], xp: [T, D], z0: [T, D], h1: [T, D], q: [T, D], k: [T, D], v: [T, D], att0: [T, T], att1: [T, T],
  o: [T, D], z1: [T, D], h2: [T, D], hid: [T, HID], z2: [T, D], avg: [1, D], avgNorm: [1, D], prob: [1, NC],
  Wp: [D, P * P], bp: [1, D], pos: [T, D], ln1: [2, D], Wq: [D, D], bq: [1, D], Wk: [D, D], bk: [1, D], Wv: [D, D], bv: [1, D],
  Wo: [D, D], bo: [1, D], ln2: [2, D], W1: [D, HID], b1: [1, HID], W2: [HID, D], b2: [1, D], lnf: [2, D], Wh: [D, NC], bh: [1, NC],
};

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
/** X (n×a) · W (a×b) + bias (b). */
function linear(X, W, b) {
  return X.map((x) => b.map((bc, c) => { let s = bc; for (let a = 0; a < x.length; a++) s += x[a] * W[a][c]; return s; }));
}
export function softmax(v) { const m = Math.max(...v); const e = v.map((x) => Math.exp(x - m)); const s = e.reduce((t, x) => t + x, 0); return e.map((x) => x / s); }
const EPS = 1e-5;
function layerNorm(x, { g, b }) {
  const n = x.length, mean = x.reduce((t, v) => t + v, 0) / n, vr = x.reduce((t, v) => t + (v - mean) ** 2, 0) / n;
  const rstd = 1 / Math.sqrt(vr + EPS), xhat = x.map((v) => (v - mean) * rstd);
  return { y: xhat.map((v, d) => g[d] * v + b[d]), xhat, rstd };
}
const K0 = Math.sqrt(2 / Math.PI);
/** GELU in its tanh form. */
export const gelu = (x) => 0.5 * x * (1 + Math.tanh(K0 * (x + 0.044715 * x ** 3)));
function geluGrad(x) { const t = Math.tanh(K0 * (x + 0.044715 * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * K0 * (1 + 3 * 0.044715 * x * x); }

/** The digit cut into 16 patches of 16 pixels (row-major inside the patch). order[s] is the patch placed in slot s. */
export function patchify(img, order = null) {
  return Array.from({ length: T }, (_, s) => {
    const p = order ? order[s] : s, pr = p >> 2, pc = p & 3, out = new Array(P * P);
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) out[a * P + b] = img[(pr * P + a) * N + pc * P + b];
    return out;
  });
}
/** The 16×16 picture the model sees: slot s drawn at block s. */
function unpatch(patches) {
  const img = zeros(N, N);
  patches.forEach((px, s) => { for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) img[(s >> 2) * P + a][(s & 3) * P + b] = px[a * P + b]; });
  return img;
}

/** Seeded starting weights: matrices uniform ±√(6/(fan_in + fan_out)) (Glorot), position ±0.1, LayerNorm scale 1 and
 *  shift 0, biases 0. */
export function init(seed = 1) {
  const rnd = mulberry32(seed);
  const mat = (r, c, s) => Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() * 2 - 1) * s));
  const glorot = (a, b) => mat(a, b, Math.sqrt(6 / (a + b)));
  const zero = (n) => new Array(n).fill(0), ln = () => ({ g: new Array(D).fill(1), b: zero(D) });
  return {
    Wp: mat(D, P * P, Math.sqrt(6 / (P * P + D))), bp: zero(D), pos: mat(T, D, 0.1),
    ln1: ln(), Wq: glorot(D, D), bq: zero(D), Wk: glorot(D, D), bk: zero(D), Wv: glorot(D, D), bv: zero(D), Wo: glorot(D, D), bo: zero(D),
    ln2: ln(), W1: glorot(D, HID), b1: zero(HID), W2: glorot(HID, D), b2: zero(D),
    lnf: ln(), Wh: glorot(D, NC), bh: zero(NC),
  };
}

/** Every intermediate of one forward pass. position false leaves the position table out; order shuffles the patches. */
export function forward(w, img, { position = true, order = null } = {}) {
  const patches = patchify(img, order);
  const xp = patches.map((px) => w.Wp.map((k, d) => dot(px, k) + w.bp[d]));
  const z0 = xp.map((t, i) => t.map((v, d) => v + (position ? w.pos[i][d] : 0)));
  const n1 = z0.map((t) => layerNorm(t, w.ln1)), h1 = n1.map((n) => n.y);
  const q = linear(h1, w.Wq, w.bq), k = linear(h1, w.Wk, w.bk), v = linear(h1, w.Wv, w.bv);
  const cols = (X, h) => X.map((r) => r.slice(h * DH, h * DH + DH));
  const scores = [], att = [], heads = [];
  for (let h = 0; h < H; h++) {
    const qh = cols(q, h), kh = cols(k, h), vh = cols(v, h);
    const s = qh.map((qi) => kh.map((kj) => dot(qi, kj) / Math.sqrt(DH)));
    const a = s.map(softmax);
    scores.push(s); att.push(a);
    heads.push(a.map((ai) => Array.from({ length: DH }, (_, c) => ai.reduce((t, aij, j) => t + aij * vh[j][c], 0))));
  }
  const o = q.map((_, i) => heads.flatMap((hh) => hh[i]));
  const mixed = linear(o, w.Wo, w.bo);
  const z1 = z0.map((t, i) => t.map((x, d) => x + mixed[i][d]));
  const n2 = z1.map((t) => layerNorm(t, w.ln2)), h2 = n2.map((n) => n.y);
  const pre = linear(h2, w.W1, w.b1), hid = pre.map((r) => r.map(gelu));
  const mlp = linear(hid, w.W2, w.b2);
  const z2 = z1.map((t, i) => t.map((x, d) => x + mlp[i][d]));
  const avg = Array.from({ length: D }, (_, d) => z2.reduce((t, r) => t + r[d], 0) / T);
  const nf = layerNorm(avg, w.lnf), avgNorm = nf.y;
  const logits = w.bh.map((b, c) => avgNorm.reduce((t, x, d) => t + x * w.Wh[d][c], b));
  const prob = softmax(logits);
  return { img: unpatch(patches), patches, xp, z0, n1, h1, q, k, v, scores, att, heads, o, mixed, z1, n2, h2, pre, hid, mlp, z2, avg, nf, avgNorm, logits, prob, position, order };
}

/** Any drawn tensor as rows of numbers, by its SHAPES key. */
export function tensor(key, S, w) {
  switch (key) {
    case 'digit': return S.img;
    case 'att0': return S.att[0];
    case 'att1': return S.att[1];
    case 'avg': case 'avgNorm': case 'prob': return [S[key]];
    case 'ln1': case 'ln2': case 'lnf': return [w[key].g, w[key].b];
    case 'bp': case 'bq': case 'bk': case 'bv': case 'bo': case 'b1': case 'b2': case 'bh': return [w[key]];
    default: return key in w ? w[key] : S[key];
  }
}

/** A structure shaped like t (nested arrays and objects of numbers), each number replaced by fn(that number, the
 *  numbers at the same place in more...). */
function zipTree(fn, t, ...more) {
  if (typeof t === 'number') return fn(t, ...more);
  if (Array.isArray(t)) return t.map((x, i) => zipTree(fn, x, ...more.map((m) => m[i])));
  return Object.fromEntries(Object.keys(t).map((key) => [key, zipTree(fn, t[key], ...more.map((m) => m[key]))]));
}
export function paramCount(w) { let n = 0; zipTree(() => { n++; return 0; }, w); return n; }
/** True when o has exactly the structure of the model's weights, with finite numbers. */
export function isWeights(o) {
  const same = (a, b) => typeof a === 'number' ? typeof b === 'number' && Number.isFinite(b)
    : Array.isArray(a) ? Array.isArray(b) && a.length === b.length && a.every((x, i) => same(x, b[i]))
      : b != null && typeof b === 'object' && Object.keys(a).every((key) => same(a[key], b[key]));
  return same(init(1), o);
}

export const crossEntropy = (prob, label) => -Math.log(Math.max(prob[label], 1e-12));
/** dL/dx through LayerNorm from dL/dy; adds dL/dscale and dL/dshift into dg, db. */
function lnBack(dy, { xhat, rstd }, g, dg, db) {
  const n = dy.length, dxhat = dy.map((v, d) => v * g[d]);
  let m1 = 0, m2 = 0;
  for (let d = 0; d < n; d++) { dg[d] += dy[d] * xhat[d]; db[d] += dy[d]; m1 += dxhat[d]; m2 += dxhat[d] * xhat[d]; }
  m1 /= n; m2 /= n;
  return dxhat.map((v, d) => rstd * (v - m1 - xhat[d] * m2));
}
/** Backward through Y = X · W + b: adds dL/dW and dL/db into dW, db; returns dL/dX. */
function linearBack(X, W, dY, dW, db) {
  for (let i = 0; i < X.length; i++) for (let c = 0; c < dY[i].length; c++) {
    const g = dY[i][c]; if (g === 0) continue;
    db[c] += g; for (let a = 0; a < X[i].length; a++) dW[a][c] += X[i][a] * g;
  }
  return dY.map((dy) => W.map((row) => dot(row, dy)));
}

/** Loss and the gradient of every weight (same structure as the weights) for one digit. */
export function lossAndGrad(w, img, label, opts = {}) {
  const S = forward(w, img, opts), g = zipTree(() => 0, w);
  const loss = crossEntropy(S.prob, label);
  const dLogits = S.prob.map((p, c) => p - (c === label ? 1 : 0));
  // head: average → LayerNorm → W_head
  const [dAvgNorm] = linearBack([S.avgNorm], w.Wh, [dLogits], g.Wh, g.bh);
  const dAvg = lnBack(dAvgNorm, S.nf, w.lnf.g, g.lnf.g, g.lnf.b);
  const dZ2 = Array.from({ length: T }, () => dAvg.map((x) => x / T));
  // MLP and its residual
  const dHid = linearBack(S.hid, w.W2, dZ2, g.W2, g.b2);
  const dPre = dHid.map((r, i) => r.map((x, f) => x * geluGrad(S.pre[i][f])));
  const dH2 = linearBack(S.h2, w.W1, dPre, g.W1, g.b1);
  const dZ1 = dZ2.map((r, i) => { const dx = lnBack(dH2[i], S.n2[i], w.ln2.g, g.ln2.g, g.ln2.b); return r.map((x, d) => x + dx[d]); });
  // attention and its residual
  const dO = linearBack(S.o, w.Wo, dZ1, g.Wo, g.bo);
  const dQ = zeros(T, D), dK = zeros(T, D), dV = zeros(T, D), sc = 1 / Math.sqrt(DH);
  for (let h = 0; h < H; h++) {
    const A = S.att[h], off = h * DH;
    for (let i = 0; i < T; i++) {
      const dA = new Array(T);
      for (let j = 0; j < T; j++) {
        let s = 0;
        for (let c = 0; c < DH; c++) { s += dO[i][off + c] * S.v[j][off + c]; dV[j][off + c] += A[i][j] * dO[i][off + c]; }
        dA[j] = s;
      }
      const m = A[i].reduce((t, a, j) => t + a * dA[j], 0);
      for (let j = 0; j < T; j++) {
        const dS = A[i][j] * (dA[j] - m) * sc;
        for (let c = 0; c < DH; c++) { dQ[i][off + c] += dS * S.k[j][off + c]; dK[j][off + c] += dS * S.q[i][off + c]; }
      }
    }
  }
  const dHq = linearBack(S.h1, w.Wq, dQ, g.Wq, g.bq), dHk = linearBack(S.h1, w.Wk, dK, g.Wk, g.bk), dHv = linearBack(S.h1, w.Wv, dV, g.Wv, g.bv);
  const dZ0 = dZ1.map((r, i) => {
    const dx = lnBack(dHq[i].map((x, d) => x + dHk[i][d] + dHv[i][d]), S.n1[i], w.ln1.g, g.ln1.g, g.ln1.b);
    return r.map((x, d) => x + dx[d]);
  });
  // position and patch embedding
  if (S.position) dZ0.forEach((r, i) => r.forEach((x, d) => { g.pos[i][d] += x; }));
  for (let p = 0; p < T; p++) for (let d = 0; d < D; d++) {
    const x = dZ0[p][d]; g.bp[d] += x;
    for (let q2 = 0; q2 < P * P; q2++) g.Wp[d][q2] += x * S.patches[p][q2];
  }
  return { S, loss, grads: g };
}

/** One SGD step, w − lr · g, as fresh arrays. */
export function sgdStep(w, g, lr) { return zipTree((x, gx) => x - lr * gx, w, g); }
export function adamInit(w) { return { t: 0, m: zipTree(() => 0, w), v: zipTree(() => 0, w) }; }
/** One Adam step; returns the new weights and optimiser state. */
export function adamStep(w, g, st, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const t = st.t + 1;
  const m = zipTree((mx, gx) => b1 * mx + (1 - b1) * gx, st.m, g);
  const v = zipTree((vx, gx) => b2 * vx + (1 - b2) * gx * gx, st.v, g);
  const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
  return { w: zipTree((x, mx, vx) => x - lr * (mx / c1) / (Math.sqrt(vx / c2) + eps), w, m, v), state: { t, m, v } };
}
/** Element-wise mean of a list of gradient sets. */
export function averageGrads(list) { return zipTree((...xs) => xs.reduce((a, b) => a + b, 0) / xs.length, ...list); }

/** Class probabilities and the top class for one digit. */
export function predict(w, img, opts) {
  const { prob } = forward(w, img, opts);
  let argmax = 0; for (let c = 1; c < NC; c++) if (prob[c] > prob[argmax]) argmax = c;
  return { prob, argmax };
}
/** Fraction of the given digits of a set that the weights classify correctly. */
export function accuracy(w, set, indices, opts) {
  let ok = 0;
  for (const i of indices) if (predict(w, image(set, i), opts).argmax === set.labels[i]) ok++;
  return ok / indices.length;
}
```

- [ ] **Step 4: Run the test**

Run: `node core/vit.test.mjs`
Expected (about 2 s):
```text
forward: shapes, attention rows, LayerNorm, patch conv, shuffle ok (position off 5.6e-17, on 2.0e-1)
gradient check (default): 1242 weights, max normalised error 1.51e-8, median 1.05e-11
gradient check (position off, shuffled): 1242 weights, max normalised error 1.68e-6, median 8.10e-12
updates: one digit, 50 steps, loss 2.070 → SGD 0.0090, Adam 0.0052
core/vit.test.mjs: all checks passed
```
The exact small numbers may differ in the last digits. Any `Error:` line means a failure: fix the code, not the test.

- [ ] **Step 5: Run the other tests, so nothing shared broke**

Run: `node core/cnn.test.mjs && node core/train.test.mjs && node core/data.test.mjs`
Expected: each passes, as before.

- [ ] **Step 6: Commit**

```bash
git add core/vit.js core/vit.test.mjs
git commit -m "feat(core): the toy ViT's forward pass, gradients and updates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `vitTrace`, the graded green

**Files:**
- Modify: `core/vit.js` (append at the end)
- Modify: `core/vit.test.mjs` (the import line, and a block before the last line)

**Interfaces:**
- Consumes: `SHAPES`, `P`, `T`, `H`, `DH` and a forward result `S` from Task 1.
- Produces: `vitTrace(S, probe)`, where `probe` is `{ t, r, c }` (`t` a `SHAPES` key, and `r`, `c` a cell of that tensor) or null. It returns `{ key: Float32Array(rows × cols) }` with a strength from 0 to 1 per cell; keys with nothing lit are absent, and the hovered cell itself is 1. The rules are in the function's comment. Task 5 calls it on every render, and with `{ t: 'o', r: i, c: head * 4 }` to shade the digit by token i's attention.

- [ ] **Step 1: Write the failing test**

In `core/vit.test.mjs`, replace the import line
```js
import { N, P, T, D, H, DH, HID, NC, SHAPES, init, forward, tensor, paramCount, isWeights, crossEntropy, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, predict, accuracy } from './vit.js';
```
with
```js
import { N, P, T, D, H, DH, HID, NC, SHAPES, init, forward, tensor, paramCount, isWeights, crossEntropy, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, predict, accuracy, vitTrace } from './vit.js';
```
and insert this block just before the last line (`console.log('core/vit.test.mjs: all checks passed');`):

```js
// ---- the green: vitTrace ----
{
  const at = (tr, key, r, c) => (tr[key] ? tr[key][r * SHAPES[key][1] + c] : 0);
  // before attention: on/off, exactly token 5's patch (rows 4–7, cols 4–7)
  const tq = vitTrace(S, { t: 'q', r: 5, c: 2 });
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) assert(at(tq, 'digit', r, c) === (r >= 4 && r < 8 && c >= 4 && c < 8 ? 1 : 0), `q probe: digit (${r}, ${c})`);
  assert(!tq.k && !tq.v && !tq.att0 && at(tq, 'Wq', 3, 2) === 1 && at(tq, 'Wq', 3, 1) === 0 && at(tq, 'bq', 0, 2) === 1, 'q probe: its weights, no attention');
  // after attention, one head: own patch 1, others 0.9 × A / max A (k ≠ i), only that head's columns
  const gradeCheck = (tr, i, A, heads) => {
    let m = 0; A[i].forEach((a, j) => { if (j !== i && a > m) m = a; });
    for (let j = 0; j < T; j++) {
      const want = j === i ? 1 : 0.9 * A[i][j] / m;
      close(at(tr, 'digit', (j >> 2) * 4 + 1, (j & 3) * 4 + 2), want, 1e-6, `token ${j} patch strength`);
      close(at(tr, 'z0', j, 3), want, 1e-6, `token ${j} z0 strength`);
      for (let c = 0; c < D; c++) close(at(tr, 'v', j, c), heads.includes(Math.floor(c / DH)) ? want : 0, 1e-6, `value ${j}, ${c}`);
    }
    const others = Array.from({ length: T }, (_, j) => j).filter((j) => j !== i).map((j) => at(tr, 'z0', j, 0));
    close(Math.max(...others), Math.fround(0.9), 1e-7, 'most-attended other token at 0.9');
  };
  gradeCheck(vitTrace(S, { t: 'o', r: 11, c: 1 }), 11, S.att[0], [0]);
  gradeCheck(vitTrace(S, { t: 'o', r: 11, c: 6 }), 11, S.att[1], [1]);
  const mean = S.att[0].map((r, a) => r.map((x, b) => (x + S.att[1][a][b]) / 2));
  const tz = vitTrace(S, { t: 'z1', r: 3, c: 0 });
  gradeCheck(tz, 3, mean, [0, 1]);
  assert(at(tz, 'Wo', 5, 0) === 1 && at(tz, 'Wo', 5, 1) === 0 && at(tz, 'z1', 3, 0) === 1, 'z1 probe: Wo column and the cell');
  // attention weight: query i and key j in full, other keys at 0.2
  const ta = vitTrace(S, { t: 'att1', r: 2, c: 9 });
  assert(at(ta, 'q', 2, 4) === 1 && at(ta, 'q', 3, 4) === 0 && at(ta, 'k', 9, 5) === 1 && Math.abs(at(ta, 'k', 0, 5) - 0.2) < 1e-6 && at(ta, 'k', 9, 0) === 0, 'attention probe');
  // the head: direct links only
  assert(Object.keys(vitTrace(S, { t: 'prob', r: 0, c: 7 })).sort().join() === 'Wh,avgNorm,bh,prob', 'class: direct link only');
  const tavg = vitTrace(S, { t: 'avg', r: 0, c: 2 });
  assert(Object.keys(tavg).sort().join() === 'avg,z2' && [...Array(T).keys()].every((r) => at(tavg, 'z2', r, 2) === 1 && at(tavg, 'z2', r, 3) === 0), 'average: its column of z2');
  // a weight: the column it reads and the column it makes
  const tw = vitTrace(S, { t: 'Wq', r: 1, c: 6 });
  assert(at(tw, 'h1', 7, 1) === 1 && at(tw, 'h1', 7, 2) === 0 && at(tw, 'q', 7, 6) === 1 && at(tw, 'q', 7, 5) === 0 && !tw.digit, 'weight probe');
  assert(Object.keys(vitTrace(S, null)).length === 0, 'no probe, nothing lit');
  console.log('vitTrace: before attention, one head, both heads, attention weight, head, weights ok');
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/vit.test.mjs`
Expected: FAIL, `SyntaxError: The requested module './vit.js' does not provide an export named 'vitTrace'`.

- [ ] **Step 3: Append `vitTrace` to `core/vit.js`**

Add at the end of `core/vit.js`, after `accuracy`:

```js
/**
 * The green: for a hovered value probe { t, r, c } (t a SHAPES key), a strength from 0 to 1 for every cell it is
 * computed from, as { key: Float32Array(rows × cols) }; keys with nothing lit are absent. The hovered cell is 1.
 * Before attention a value is computed from its own token only: 1 for those cells, 0 elsewhere. After attention the
 * hovered token's own row and patch are 1 and every other token j gets 0.9 · A[i][j] / (largest A[i][k], k ≠ i): a
 * head's output follows that head's map, values after the heads are joined follow the mean of the two. An attention
 * weight A[i][j] lights query i and key j in full and every other key at 0.2, since the softmax divides by all of
 * them. A weight lights the column it reads and the column it makes. The head shows only its direct links.
 */
export function vitTrace(S, probe) {
  const out = {};
  if (!probe) return out;
  const put = (key, r, c, s) => {
    const [, C] = SHAPES[key], a = out[key] || (out[key] = new Float32Array(SHAPES[key][0] * C));
    if (s > a[r * C + c]) a[r * C + c] = s;
  };
  const row = (key, r, s, c0 = 0, c1 = SHAPES[key][1] - 1) => { for (let c = c0; c <= c1; c++) put(key, r, c, s); };
  const col = (key, c, s) => { for (let r = 0; r < SHAPES[key][0]; r++) put(key, r, c, s); };
  const block = (j, s) => { for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) put('digit', (j >> 2) * P + a, (j & 3) * P + b, s); };
  const token = (j, s) => { block(j, s); row('xp', j, s); row('pos', j, s); row('z0', j, s); row('h1', j, s); };
  const grades = (i, A) => {
    let m = 0; A[i].forEach((a, j) => { if (j !== i && a > m) m = a; });
    return A[i].map((a, j) => (j === i ? 1 : m > 0 ? 0.9 * a / m : 0));
  };
  const attended = (i, h) => { // h: one head, or null for the mean of both
    const A = h == null ? S.att[0].map((r, a) => r.map((x, b) => S.att.reduce((t, M) => t + M[a][b], 0) / H)) : S.att[h];
    const gr = grades(i, A);
    for (const hh of h == null ? [0, 1] : [h]) {
      const c0 = hh * DH, c1 = c0 + DH - 1;
      row('att' + hh, i, 1); row('q', i, 1, c0, c1);
      gr.forEach((s, j) => { row('k', j, s, c0, c1); row('v', j, s, c0, c1); });
    }
    gr.forEach((s, j) => token(j, s));
  };
  const { t, r, c } = probe;
  put(t, r, c, 1);
  switch (t) {
    case 'xp': block(r, 1); row('Wp', c, 1); put('bp', 0, c, 1); break;
    case 'z0': block(r, 1); put('xp', r, c, 1); put('pos', r, c, 1); row('Wp', c, 1); put('bp', 0, c, 1); break;
    case 'h1': token(r, 1); col('ln1', c, 1); break;
    case 'q': case 'k': case 'v': token(r, 1); col('W' + t, c, 1); put('b' + t, 0, c, 1); break;
    case 'att0': case 'att1': {
      const h = +t[3], c0 = h * DH, c1 = c0 + DH - 1;
      for (let j = 0; j < T; j++) { row('k', j, j === c ? 1 : 0.2, c0, c1); token(j, j === r || j === c ? 1 : 0.2); }
      row('q', r, 1, c0, c1); break;
    }
    case 'o': attended(r, Math.floor(c / DH)); break;
    case 'z1': attended(r, null); row('o', r, 1); col('Wo', c, 1); put('bo', 0, c, 1); break;
    case 'h2': attended(r, null); row('o', r, 1); row('z1', r, 1); col('ln2', c, 1); break;
    case 'hid': attended(r, null); row('o', r, 1); row('z1', r, 1); row('h2', r, 1); col('W1', c, 1); put('b1', 0, c, 1); break;
    case 'z2': attended(r, null); row('o', r, 1); row('z1', r, 1); row('h2', r, 1); row('hid', r, 1); col('W2', c, 1); put('b2', 0, c, 1); break;
    case 'avg': col('z2', c, 1); break;
    case 'avgNorm': row('avg', 0, 1); col('lnf', c, 1); break;
    case 'prob': row('avgNorm', 0, 1); col('Wh', c, 1); put('bh', 0, c, 1); break;
    case 'Wp': for (let j = 0; j < T; j++) put('digit', (j >> 2) * P + (c >> 2), (j & 3) * P + (c & 3), 1); col('xp', r, 1); break;
    case 'bp': col('xp', c, 1); break;
    case 'pos': put('z0', r, c, 1); break;
    case 'ln1': col('h1', c, 1); break;
    case 'ln2': col('h2', c, 1); break;
    case 'lnf': put('avgNorm', 0, c, 1); break;
    case 'Wq': case 'Wk': case 'Wv': col('h1', r, 1); col(t[1], c, 1); break;
    case 'bq': case 'bk': case 'bv': col(t[1], c, 1); break;
    case 'Wo': col('o', r, 1); col('z1', c, 1); break;
    case 'bo': col('z1', c, 1); break;
    case 'W1': col('h2', r, 1); col('hid', c, 1); break;
    case 'b1': col('hid', c, 1); break;
    case 'W2': col('hid', r, 1); col('z2', c, 1); break;
    case 'b2': col('z2', c, 1); break;
    case 'Wh': put('avgNorm', 0, r, 1); put('prob', 0, c, 1); break;
    case 'bh': put('prob', 0, c, 1); break;
    default: break; // digit: the input itself
  }
  return out;
}
```

- [ ] **Step 4: Run the test**

Run: `node core/vit.test.mjs`
Expected: the Task 1 lines, then `vitTrace: before attention, one head, both heads, attention weight, head, weights ok`, then `core/vit.test.mjs: all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add core/vit.js core/vit.test.mjs
git commit -m "feat(core): vitTrace, the graded green of the ViT page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Training script, the optimiser measurement, the shipped weights

**Files:**
- Create: `tools/train_vit.mjs`
- Create: `models/vit16.json` (generated)

**Interfaces:**
- Consumes: `init`, `paramCount`, `lossAndGrad`, `sgdStep`, `adamInit`, `adamStep`, `averageGrads`, `accuracy` (Task 1); `shuffledOrder` (`core/train.js`); `parseMnist16`, `image` (`core/data.js`).
- Produces: `models/vit16.json`, of the form `{ version: 1, arch: 'vit16', meta: { seed, optimiser: 'sgd' | 'adam', lr, batch, passes, digits, trainAcc, testAcc, made, date }, weights }`. `weights` has the structure of `init()`, with numbers rounded to 6 significant digits. `testAcc` and `trainAcc` are measured on the rounded weights. Task 5 reads `meta.testAcc`, `meta.passes` and `meta.optimiser`.

- [ ] **Step 1: Write the script**

Create `tools/train_vit.mjs`:

```js
// tools/train_vit.mjs — train the toy ViT of core/vit.js on data/mnist16.bin; with --out, write the weights the
// ViT page loads. Prints held-out accuracy after every pass over the 3,000 training digits.
//   node tools/train_vit.mjs --opt sgd --lr 0.1 --passes 10
//   node tools/train_vit.mjs --opt adam --lr 0.003 --passes 10 --out models/vit16.json
// Options: --opt sgd|adam, --lr, --passes, --batch (default 1), --seed (default 1), --out (a path from the repo root).
import { readFileSync, writeFileSync } from 'node:fs';
import { parseMnist16, image } from '../core/data.js';
import { shuffledOrder } from '../core/train.js';
import { init, paramCount, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, accuracy } from '../core/vit.js';

const argv = process.argv.slice(2), arg = (name, dflt) => { const i = argv.indexOf('--' + name); return i < 0 ? dflt : argv[i + 1]; };
const opt = arg('opt', 'sgd'), lr = +arg('lr', opt === 'adam' ? 0.003 : 0.1), passes = +arg('passes', 10);
const batch = +arg('batch', 1), seed = +arg('seed', 1), out = arg('out', null);
if (!['sgd', 'adam'].includes(opt)) throw new Error('--opt is sgd or adam');

const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
const every = (set) => Array.from({ length: set.n }, (_, i) => i);
const pct = (x) => (100 * x).toFixed(1) + '%';
let w = init(seed), st = adamInit(w);
console.log(`ViT, ${paramCount(w)} weights; ${opt}, lr ${lr}, batch ${batch}, seed ${seed}`);
console.log(`pass 0: held-out ${pct(accuracy(w, data.test, every(data.test)))}`);
const t0 = Date.now();
for (let pass = 1; pass <= passes; pass++) {
  const order = shuffledOrder(data.train.n, seed * 1000 + pass);
  let lossSum = 0;
  for (let b = 0; b < order.length; b += batch) {
    const gs = order.slice(b, b + batch).map((i) => { const r = lossAndGrad(w, image(data.train, i), data.train.labels[i]); lossSum += r.loss; return r.grads; });
    const g = gs.length === 1 ? gs[0] : averageGrads(gs);
    if (opt === 'adam') ({ w, state: st } = adamStep(w, g, st, lr)); else w = sgdStep(w, g, lr);
  }
  console.log(`pass ${pass}: mean loss ${(lossSum / order.length).toFixed(3)}, held-out ${pct(accuracy(w, data.test, every(data.test)))} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

if (out) {
  const weights = JSON.parse(JSON.stringify(w), (k, v) => (typeof v === 'number' ? +v.toPrecision(6) : v));
  const meta = {
    seed, optimiser: opt, lr, batch, passes, digits: passes * data.train.n,
    trainAcc: +accuracy(weights, data.train, every(data.train)).toFixed(4), testAcc: +accuracy(weights, data.test, every(data.test)).toFixed(4),
    made: `node tools/train_vit.mjs ${argv.join(' ')}`, date: new Date().toISOString().slice(0, 10),
  };
  writeFileSync(new URL('../' + out, import.meta.url), JSON.stringify({ version: 1, arch: 'vit16', meta, weights }) + '\n');
  console.log(`wrote ${out}: held-out ${pct(meta.testAcc)}, training ${pct(meta.trainAcc)} (rounded weights)`);
}
```

- [ ] **Step 2: Smoke run**

Run: `node tools/train_vit.mjs --opt sgd --lr 0.01 --passes 1`
Expected: three lines: `ViT, 1242 weights; sgd, lr 0.01, batch 1, seed 1`, then `pass 0: held-out 10.1%`, then `pass 1: mean loss 2.177, held-out 35.4% (… s)`.

- [ ] **Step 3: Measure SGD and Adam, 10 passes each**

Run (Git Bash; the six runs go in parallel and take about a minute):
```bash
mkdir -p .superpowers/check
for c in "sgd 0.003" "sgd 0.01" "sgd 0.03" "adam 0.001" "adam 0.003" "adam 0.01"; do set -- $c; node tools/train_vit.mjs --opt $1 --lr $2 --passes 10 > .superpowers/check/run_$1_$2.txt 2>&1 & done; wait
for f in .superpowers/check/run_*.txt; do echo "$f: $(grep 'pass 10:' $f)"; done
```
Expected: six lines. The prototype's figures are in the context note above; for example `run_sgd_0.003.txt: pass 10: mean loss 0.609, held-out 82.0% (…)` and `run_adam_0.003.txt: pass 10: mean loss 0.448, held-out 84.9% (…)`.

- [ ] **Step 4: Choose the optimiser (the spec's rule)**

Take the best pass-10 held-out accuracy for SGD and for Adam:
- **SGD** if Adam's best is at most 3 points higher. Use SGD at its best learning rate. This keeps the later training page's update row w′ = w − lr·∂L/∂w.
- **Adam** otherwise, at its best learning rate.

With the prototype's figures, this is SGD at lr 0.003 (82.0% against 84.9%). Write the six figures down for Task 6.

- [ ] **Step 5: Write the weights file**

Run (with the chosen optimiser and rate, here SGD 0.003):
`node tools/train_vit.mjs --opt sgd --lr 0.003 --passes 10 --out models/vit16.json`
Expected: the pass lines, then `wrote models/vit16.json: held-out 82.0%, training …% (rounded weights)`.

- [ ] **Step 6: Check the file**

Run:
```bash
node -e "import('./core/vit.js').then(async (v) => { const j = JSON.parse(require('fs').readFileSync('models/vit16.json', 'utf8')); console.log(require('fs').statSync('models/vit16.json').size, 'bytes', j.version, j.arch, v.isWeights(j.weights), JSON.stringify(j.meta)); })"
```
Expected: about 12,000 bytes, then `1 vit16 true` and the meta, with `testAcc` equal to the accuracy Step 5 printed.

- [ ] **Step 7: Commit**

```bash
git add tools/train_vit.mjs models/vit16.json
git commit -m "feat(tools): train the toy ViT and ship its weights" -m "Held-out accuracy after 10 passes, seed 1, batch 1: <the six figures from Step 3>. Chosen: <optimiser, lr> by the spec's 3-point rule." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Fill in the two angle-bracket parts with the measured figures and the choice. They are the only free text in this plan.

---

### Task 4: Shared UI: graded green, yellow window, per-page terms

**Files:**
- Create: `.superpowers/check/domcheck.mjs` (git-ignored)
- Modify: `ui/grid.js` (whole file below)
- Modify: `ui/terms.js` (whole file below)

**Interfaces:**
- Produces:
  - `drawGrid(cv, data, w, h, o)` gains four options that the CNN pages do not use:
    - `glow`: an array of w × h strengths from 0 to 1, drawn as a green fill; cells at full strength get one green outline around them as a region;
    - `win: [r0, r1, c0, c1]`: a yellow rectangle;
    - `blocks: n`: dashed accent lines every n cells;
    - `lines: false`: the outline only, without the inner grid lines.
  - `installTerms({ selectors, tip, gloss, terms = TERMS })`; `linkTerms(selectors, terms)`, `bindTooltip(tip, terms)` and `renderGlossary(el, terms)` take the list too.
  - `VIT_TERMS`, the ViT page's 18 terms.

- [ ] **Step 1: Write the DOM check**

Create `.superpowers/check/domcheck.mjs`:

```js
// domcheck.mjs — dump the three CNN pages' DOM in headless Edge, with a seeded Math.random and scripts stripped, so that
// a dump before and a dump after a change to the shared ui/ modules can be compared byte for byte.
//   node domcheck.mjs <root> <outdir>
// Serves <root> on 127.0.0.1:8124, writes <page>/_dom.html (index.html with the shim after <head>) for each page,
// dumps it to <outdir>/<page>.html, then deletes the copies. Compare two outdirs with `diff -r`.
import { readFileSync, writeFileSync, rmSync, mkdirSync, mkdtempSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const [root, outdir] = process.argv.slice(2).map((p) => resolve(p));
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', PORT = 8124, PAGES = ['cnn-2d', 'cnn-3d', 'cnn-train'];
const SHIM = '<script>(() => { let s = 12345; Math.random = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();</script>';
mkdirSync(outdir, { recursive: true });
for (const p of PAGES) writeFileSync(join(root, p, '_dom.html'), readFileSync(join(root, p, 'index.html'), 'utf8').replace('<head>', '<head>' + SHIM));
const server = spawn('python', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', root], { stdio: 'ignore' });
try {
  for (let k = 0; ; k++) { try { const r = await fetch(`http://127.0.0.1:${PORT}/`); await r.arrayBuffer(); if (r.ok) break; } catch { /* not up yet */ } if (k > 50) throw new Error('server did not start'); await new Promise((r) => setTimeout(r, 100)); }
  for (const p of PAGES) {
    const profile = mkdtempSync(join(tmpdir(), 'domcheck-'));
    const out = await new Promise((res) => {
      const e = spawn(EDGE, ['--headless=new', '--disable-gpu', '--enable-unsafe-swiftshader', '--no-first-run', `--user-data-dir=${profile}`, '--virtual-time-budget=5000', '--dump-dom', `http://127.0.0.1:${PORT}/${p}/_dom.html`]);
      let s = ''; e.stdout.on('data', (d) => { s += d; }); e.on('close', () => res(s));
    });
    const dom = out.replace(/<script[\s\S]*?<\/script>/g, '');
    writeFileSync(join(outdir, p + '.html'), dom);
    console.log(`${p}: ${dom.length} characters, ${(dom.match(/<dfn/g) || []).length} dotted terms`);
  }
} finally { server.kill(); for (const p of PAGES) rmSync(join(root, p, '_dom.html'), { force: true }); }
```

- [ ] **Step 2: Dump the CNN pages before the change**

Run: `node .superpowers/check/domcheck.mjs . .superpowers/check/dom-before`
Expected:
```text
cnn-2d: 28809 characters, 30 dotted terms
cnn-3d: 15162 characters, 4 dotted terms
cnn-train: 45940 characters, 18 dotted terms
```
The character counts may differ slightly with fonts or Edge's version; what matters is that the before and after dumps match. If `cnn-3d` shows 0 dotted terms, three.js did not load from cdnjs: rerun.

- [ ] **Step 3: Replace `ui/grid.js`**

The new parts are the `lines`, `blocks`, `glow` and `win` blocks and the two comments; everything else is unchanged.

```js
// ui/grid.js — canvas grid drawing and hit-testing shared by the 2D, training and ViT pages.
// Colours come from the page's CSS tokens (--line, --ink, --hl, --hl2) and follow its light/dark theme.
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/** Short number: '0', '1.5', '-0.33'. */
export const fmt = (v) => { if (Math.abs(v) < 1e-9) return '0'; const s = v.toFixed(2).replace(/\.?0+$/, ''); return s === '-0' ? '0' : s; }; // no "-0" for tiny negatives
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

/** Draw a w×h grid of values on a canvas, with optional labels and highlights; remembers the layout for cellAt.
 *  Options used by the ViT page only: glow (a 0–1 green strength per cell), win ([r0, r1, c0, c1], a yellow
 *  rectangle), blocks (dashed lines every n cells) and lines: false (no inner grid lines). */
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
  if (o.lines === false) ctx.strokeRect(ox + .5, oy + .5, w * cell - 1, h * cell - 1); // small cells: the outline only
  else {
    for (let r = 0; r <= h; r++) { ctx.beginPath(); ctx.moveTo(ox, oy + r * cell + .5); ctx.lineTo(ox + w * cell, oy + r * cell + .5); ctx.stroke(); }
    for (let c = 0; c <= w; c++) { ctx.beginPath(); ctx.moveTo(ox + c * cell + .5, oy); ctx.lineTo(ox + c * cell + .5, oy + h * cell); ctx.stroke(); }
  }
  if (o.blocks) { // dashed lines every `blocks` cells: the patch grid
    ctx.strokeStyle = cssVar('--accent'); ctx.setLineDash([3, 2]); ctx.beginPath();
    for (let k = o.blocks; k < w; k += o.blocks) { ctx.moveTo(ox + k * cell + .5, oy); ctx.lineTo(ox + k * cell + .5, oy + h * cell); }
    for (let k = o.blocks; k < h; k += o.blocks) { ctx.moveTo(ox, oy + k * cell + .5); ctx.lineTo(ox + w * cell, oy + k * cell + .5); }
    ctx.stroke(); ctx.setLineDash([]);
  }
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
  if (o.glow) { // graded green: a 0–1 strength per cell, the full-strength cells outlined as one region
    const g = o.glow, full = (r, c) => r >= 0 && c >= 0 && r < h && c < w && g[r * w + c] >= 0.999;
    ctx.fillStyle = hl2;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const s = g[r * w + c]; if (s > 0) { ctx.globalAlpha = 0.12 + 0.38 * Math.min(1, s); ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell); }
    }
    ctx.globalAlpha = 1; ctx.strokeStyle = hl2; ctx.lineWidth = cell >= 8 ? 2 : 1.5; const e = ctx.lineWidth / 2; ctx.beginPath();
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) if (full(r, c)) {
      const x = ox + c * cell, y = oy + r * cell;
      if (!full(r - 1, c)) { ctx.moveTo(x, y + e); ctx.lineTo(x + cell, y + e); }
      if (!full(r + 1, c)) { ctx.moveTo(x, y + cell - e); ctx.lineTo(x + cell, y + cell - e); }
      if (!full(r, c - 1)) { ctx.moveTo(x + e, y); ctx.lineTo(x + e, y + cell); }
      if (!full(r, c + 1)) { ctx.moveTo(x + cell - e, y); ctx.lineTo(x + cell - e, y + cell); }
    }
    ctx.stroke();
  }
  if (o.taps) { ctx.strokeStyle = hl; ctx.lineWidth = 3; o.taps.forEach(([r, c]) => ctx.strokeRect(ox + c * cell + 1.5, oy + r * cell + 1.5, cell - 3, cell - 3)); }
  if (o.cell) { ctx.strokeStyle = hl; ctx.lineWidth = 3; ctx.strokeRect(ox + o.cell[1] * cell + 1.5, oy + o.cell[0] * cell + 1.5, cell - 3, cell - 3); }
  if (o.win) { const [r0, r1, c0, c1] = o.win; ctx.strokeStyle = hl; ctx.lineWidth = cell >= 8 ? 3 : 2; const lw = ctx.lineWidth / 2; // yellow window
    ctx.strokeRect(ox + c0 * cell + lw, oy + r0 * cell + lw, (c1 - c0 + 1) * cell - 2 * lw, (r1 - r0 + 1) * cell - 2 * lw); }
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

- [ ] **Step 4: Replace `ui/terms.js`**

What changes: `TERMS` stays as it is; `VIT_TERMS` is added after it; the module-level `termById` moves into `bindTooltip`; the four functions take a `terms` list.

```js
// ui/terms.js — dotted inline definitions (tooltip on hover, focus or tap) and the Terms section.
// Shared by every page, each with its own list of terms (TERMS for the CNN pages, VIT_TERMS for the ViT page). A page links ../ui/terms.css, has <div id="tip" role="tooltip"></div> and an empty
// .gloss element, and calls installTerms once its static text is in the DOM. Dynamic text (readouts, hints
// rewritten on every step) is left unlinked on purpose.

export const TERMS = [
  ['kernel', /\bkernel\b/i, 'Kernel (filter)', 'A small grid of weights, 3×3 here, that is slid over the input. Its values are learned parameters: training changes them. Each kernel produces one feature map.'],
  ['weights', /\bweights\b/i, 'Weights / parameters', 'The numbers a network learns: every kernel value and every FC entry. Nothing else on this page is learned — ReLU, pooling, global averaging and softmax are fixed arithmetic.'],
  ['feature map', /\bfeature map\b/i, 'Feature map', 'The output grid of one kernel: at each position, how strongly the kernel\'s pattern matches the input there. Positive = looks like the pattern, negative = looks like its opposite.'],
  ['channel', /\bchannels?\b/i, 'Channel', 'One of the stacked maps at a given stage. An RGB image has 3 channels; after a layer with 8 kernels there are 8. The "C" in H×W×C.'],
  ['stride', /\bstride\b/i, 'Stride', 'How many pixels the window moves between neighbouring outputs. Stride 2 halves the output size and makes each output cell cover twice the input width.'],
  ['padding', /\bpadding\b/i, 'Padding', 'A ring of zeros added around the input so the window can be centred on edge pixels. Padding 1 with a 3×3 kernel keeps the output the same size as the input.'],
  ['dilation', /\bdilation\b/i, 'Dilation', 'Spacing between the kernel taps. Dilation 2 reads every second pixel, so a 3×3 kernel spans a 5×5 patch with the same 9 weights.'],
  ['relu', /\bReLU\b/, 'ReLU', 'Rectified linear unit: max(0, x). Keeps positive responses, replaces negative ones with zero. This is the nonlinearity that stops stacked layers from collapsing into one.'],
  ['pool', /\b(max-?pool(ing)?|pooling)\b/i, 'Pooling', 'Shrinks a map by summarising each 2×2 block with one number (its max or mean). Halves height and width; no learned parameters. Keeps the strongest response, drops exactly where it was.'],
  ['gap', /\bglobal average(?: pooling)?\b/i, 'Global average pooling', 'Averages each channel over all positions, turning an H×W×C block into a vector of C numbers: "how much of each pattern, anywhere". Removes spatial layout entirely.'],
  ['fc', /\b(fully connected(?: \(FC\))?(?: layer)?|FC)\b/, 'Fully connected (FC) layer', 'Also called dense or linear. Every input value connects to every output value through its own weight: 8 inputs × 10 outputs = 80 weights plus 10 biases. Output = input · W + b. The weights are learned parameters, like kernels; on this page they are random.'],
  ['bias', /\bbias(es)?\b/i, 'Bias', 'A constant added to each output of a layer: one per output, learned. It shifts a score up or down regardless of the input.'],
  ['score', /\bclass scores?\b/i, 'Score (logit)', 'The raw output of the FC layer for one class, before softmax. Can be any real number; higher means the network favours that class.'],
  ['softmax', /\bsoftmax\b/i, 'Softmax', 'Turns scores into probabilities: p_k = exp(s_k) / Σ exp(s_j). All outputs are positive and sum to 1; the largest score gets the largest share, and gaps between scores are exaggerated.'],
  ['rf', /\breceptive field\b/i, 'Receptive field', 'The patch of the input that can influence one value in a later map. Grows with every conv and every pool; the green boxes trace it.'],
  ['training', /\btraining\b/i, 'Training', 'Showing the network many labelled images and nudging every weight so the correct class score rises. The training page does it live; the explorer pages skip it, so their weights are random or hand-picked unless trained ones are loaded, but the forward pass is identical.'],
  ['loss', /\b(cross-entropy|loss)\b/i, 'Loss (cross-entropy)', 'One number for how wrong a prediction was: minus the log of the probability given to the true class. Zero when the network is certain and right, large when it is confident and wrong. Training pushes it down.'],
  ['gradient', /\bgradients?\b/i, 'Gradient', 'For each weight, how much the loss would change per unit change of that weight, sign included: the direction that makes things worse. Written ∂L/∂w. Training moves every weight a little the other way.'],
  ['backprop', /\bbackpropagation\b/i, 'Backpropagation', 'The bookkeeping that gives every weight its gradient. The error at the output is passed back through the layers one at a time; each layer turns the gradient at its output into the gradient at its input and at its own weights (the chain rule).'],
  ['lr', /\blearning rate\b/i, 'Learning rate', 'The step size: each weight moves by learning rate × its gradient. Too small and training crawls; too large and it overshoots, so the loss jumps around instead of falling.'],
  ['update', /\b(update step|SGD)\b/, 'Update (SGD step)', 'w ← w − learning rate × ∂L/∂w for every weight at once. Stochastic gradient descent: stochastic because each step uses one digit, or a small batch, rather than the whole dataset.'],
  ['batch', /\bbatch\b/i, 'Batch', 'How many digits are seen before one update. Their gradients are averaged, so a bigger batch gives steadier but fewer steps.'],
  ['pass', /\bpass(es)?\b/i, 'Pass (epoch)', 'One trip through the whole training set, in a shuffled order. Real training runs many.'],
  ['heldout', /\bheld-out\b/i, 'Held-out test set', 'Digits the network never trains on, kept only to measure accuracy. Accuracy on the training digits can be flattered by memorising them; held-out accuracy cannot.'],
  ['init', /\binitiali[sz]ation\b/i, 'Initialisation', 'The random starting values of the weights. Small random numbers, not zeros: identical weights would get identical gradients and never become different detectors.'],
];

/** The ViT page's terms. Longer phrases come before the words inside them ("patch embedding" before "patch"), so that
 *  at the same position the longer one is linked. */
export const VIT_TERMS = [
  ['patchemb', /\bpatch embedding\b/i, 'Patch embedding', 'The first layer of a ViT: each 4×4 patch of pixels becomes 8 numbers through 8 learned kernels. It is a convolution whose stride equals its size, so the windows never overlap.'],
  ['patch', /\bpatch(es)?\b/i, 'Patch', 'A square tile of the image, 4×4 pixels here. The 16×16 digit is cut into 16 of them, numbered in reading order.'],
  ['position', /\bposition (embedding|table)s?\b/i, 'Position embedding', 'A learned row of 8 numbers per patch slot, added to the token in that slot. Attention by itself treats the tokens as a set; this table is the only way the model can know where a patch sat.'],
  ['token', /\btokens?\b/i, 'Token', 'One vector per patch, 8 numbers here, carried through the network as one row. Every step except attention works on each token separately.'],
  ['head', /\b(attention )?heads?\b/i, 'Attention head', 'One complete attention pattern with its own queries, keys and values. This model has two, each 4 numbers wide; their outputs are joined back into 8 numbers and mixed by Wo.'],
  ['attention', /\b(self-)?attention\b/i, 'Attention', 'Each token builds its output as a weighted average of every token\'s value. The weights come from how well its query matches each key, through a softmax, so they are positive and sum to 1. They are computed from the image: where a token looks is not fixed, unlike a kernel\'s window.'],
  ['qkv', /\bquer(y|ies)\b/i, 'Query, key, value', 'Three learned projections of each token. The query is what the token looks for, the key what it offers to others, the value what it hands over. Score(i, j) = query i · key j / √4.'],
  ['layernorm', /\bLayerNorm\b/, 'LayerNorm', 'Rescales each token on its own to mean 0 and variance 1 across its 8 numbers, then applies a learned scale and shift per column. It keeps the numbers in range from step to step.'],
  ['residual', /\bresidual( connection)?s?\b/i, 'Residual connection', 'A block\'s input is added back to its output: z₁ = z₀ + attention, z₂ = z₁ + MLP. The block only has to learn a correction, and each token keeps its own information, which is why its own patch stays full green.'],
  ['mlp', /\bMLP\b/, 'MLP', 'Multilayer perceptron: two fully connected layers, 8 → 32 → 8, with GELU between them, applied to each token separately. In CNN terms, two 1×1 convolutions.'],
  ['gelu', /\bGELU\b/, 'GELU', 'Gaussian error linear unit, a smooth cousin of ReLU: close to 0 for large negative inputs, close to x for large positive ones, curved in between. Computed here in its tanh form.'],
  ['avg', /\baverage of the tokens\b/i, 'Average of the tokens', 'The 16 tokens averaged into one vector of 8 numbers, like the CNN\'s global average pooling, then turned into 10 class scores. The original ViT reads an extra class token instead; with a single block, only that token\'s row would reach the answer.'],
  ['kernel', /\bkernels?\b/i, 'Kernel (filter)', 'A small grid of learned weights, 4×4 here. In the patch embedding each of the 8 kernels reads one patch at a time and gives one of the token\'s 8 numbers.'],
  ['stride', /\bstride\b/i, 'Stride', 'How far the window jumps between outputs. With stride 4 and a 4×4 kernel the windows tile the image exactly: that is what cutting it into patches means.'],
  ['softmax', /\bsoftmax\b/i, 'Softmax', 'Turns scores into weights that are positive and sum to 1: exp(s) / Σ exp. Used twice here: across each row of an attention map, and over the 10 classes.'],
  ['rf', /\breceptive field\b/i, 'Receptive field', 'The part of the input a value can depend on. Before attention a token\'s receptive field is its own patch; after one attention step it is the whole image, which the green shades by attention weight.'],
  ['weights', /\bweights\b/i, 'Weights / parameters', 'The numbers a network learns: the patch kernels, the position table, every matrix and bias, the LayerNorm scales and shifts. Tinted wherever they sit. Softmax, GELU, the average and the residual additions are fixed arithmetic.'],
  ['heldout', /\bheld-out\b/i, 'Held-out test set', 'Digits the network never trained on, kept only to measure accuracy.'],
];

/** Wrap the first mention of each term inside every element matching `selectors` in a <dfn>.
 *  Terms are taken in order of appearance in the text, so a term is never skipped because another one
 *  listed earlier in TERMS happens to occur later in the same paragraph. */
export function linkTerms(selectors, terms = TERMS) {
  document.querySelectorAll(selectors).forEach((el) => {
    const seen = new Set();
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.parentNode.closest('dfn,.n')) return;
      let text = node.nodeValue;
      for (;;) {
        let best = null;
        for (const [id, re] of terms) { if (seen.has(id)) continue; const m = re.exec(text); if (m && (!best || m.index < best.m.index)) best = { id, m }; }
        if (!best) break;
        const { id, m } = best; seen.add(id);
        const before = document.createTextNode(text.slice(0, m.index));
        const d = document.createElement('dfn'); d.dataset.t = id; d.tabIndex = 0; d.textContent = m[0];
        const after = document.createTextNode(text.slice(m.index + m[0].length));
        node.parentNode.insertBefore(before, node); node.parentNode.insertBefore(d, node); node.parentNode.replaceChild(after, node);
        node = after; text = after.nodeValue;
      }
    });
  });
}

/** Show a <dfn>'s definition in `tip` on hover and keyboard focus; a click or tap toggles it. */
export function bindTooltip(tip, terms = TERMS) {
  const termById = Object.fromEntries(terms.map((t) => [t[0], t]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const show = (d) => {
    const t = termById[d.dataset.t]; tip.innerHTML = `<b>${t[2]}</b>${t[3]}`; tip.style.display = 'block';
    const r = d.getBoundingClientRect(); const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = clamp(r.left, 8, innerWidth - w - 8), y = r.bottom + 8; if (y + h > innerHeight - 8) y = r.top - h - 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };
  const dfnOf = (e) => e.target.closest && e.target.closest('dfn');
  document.addEventListener('pointerover', (e) => { const d = dfnOf(e); if (d) show(d); });
  document.addEventListener('pointerout', (e) => { if (dfnOf(e)) hide(); });
  document.addEventListener('focusin', (e) => { const d = dfnOf(e); if (d) show(d); });
  document.addEventListener('focusout', (e) => { if (dfnOf(e)) hide(); });
  document.addEventListener('click', (e) => {
    const d = dfnOf(e);
    if (d) { if (tip.style.display === 'block' && tip.dataset.for === d.dataset.t) { hide(); tip.dataset.for = ''; } else { show(d); tip.dataset.for = d.dataset.t; } }
    else { hide(); tip.dataset.for = ''; }
  });
}

/** Fill the Terms section with every definition. */
export function renderGlossary(el, terms = TERMS) {
  el.innerHTML = terms.map((t) => `<div><b>${t[2]}</b> — ${t[3]}</div>`).join('');
}

/** Link terms in the elements matching `selectors`, bind the tooltip element, fill the glossary element.
 *  `terms` is the page's list: TERMS for the CNN pages (the default), VIT_TERMS for the ViT page. */
export function installTerms({ selectors, tip, gloss, terms = TERMS }) {
  linkTerms(selectors, terms);
  if (tip) bindTooltip(tip, terms);
  if (gloss) renderGlossary(gloss, terms);
}
```

- [ ] **Step 5: Dump again and compare**

Run:
```bash
node .superpowers/check/domcheck.mjs . .superpowers/check/dom-after
diff -r .superpowers/check/dom-before .superpowers/check/dom-after && echo IDENTICAL
```
Expected: the same three lines as Step 2, then `IDENTICAL`.

- [ ] **Step 6: Commit**

```bash
git add ui/grid.js ui/terms.js
git commit -m "feat(ui): graded green, yellow window and per-page terms for the ViT page" -m "The CNN pages pass no new options and keep TERMS; their DOM is identical before and after (seeded dump)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The page

**Files:**
- Create: `.superpowers/check/vitcheck.mjs`, `.superpowers/check/scen_basic.js`, `.superpowers/check/scen_hover.js`, `.superpowers/check/scen_none.js` (git-ignored)
- Create: `vit/index.html`

**Interfaces:**
- Consumes: `SHAPES`, `P`, `T`, `D`, `DH`, `NC`, `init`, `forward`, `tensor`, `vitTrace`, `isWeights`, `paramCount` (`core/vit.js`); `parseMnist16`, `image` (`core/data.js`); `shuffledOrder` (`core/train.js`); `drawGrid`, `cellAt`, `fmt`, `maxAbs`, `cssVar` (`ui/grid.js`); `installTerms`, `VIT_TERMS` (`ui/terms.js`); `models/vit16.json` (Task 3).
- Produces: the page. Module-scope names that the check scenarios rely on: `ready` (the start-up promise), `views` (with `kOf(v)`), `setProbe(p)`, `render()`, `S`, `wTrained`, `meta`, `sweep`, `walk`, `$`.

How the page works, to read before the code:
- **Views.** `view(key, cell, o)` makes a canvas showing all or part of one `SHAPES` tensor. Its `map(r, c)` takes a view cell to a tensor cell; its `yl()` gives its yellow; `own()` replaces the hover's green.
- **Lines.** `line(box, items, cell)` lays columns out on a three-row grid: keys and values above, the token rows, then labels. The overview and every section are lines.
- **Rendering.** `render()` runs `forward`, computes `vitTrace(S, probe)` once and redraws every view, then updates the readouts. Hovering any canvas sets `probe` to a tensor cell, so every copy of that tensor lights together.

- [ ] **Step 1: Write the check harness and the scenarios**

Create `.superpowers/check/vitcheck.mjs`:

```js
// vitcheck.mjs — drive vit/index.html in headless Edge and print what a scenario recorded.
//   node vitcheck.mjs <root> <scenario.js> [--shot out.png] [--width 1400] [--height 1000] [--dark] [--no-weights]
// Serves <root> on 127.0.0.1:8123 with Python, writes <root>/vit/_check.html: the page with the scenario appended
// inside its module (so it sees the module's variables; it should `await ready` and fill the object R), then dumps the
// DOM with headless Edge and prints R (from body[data-check]) and the page's console errors. --dark sets
// data-theme="dark"; --no-weights points the page at a missing weights file. _check.html is deleted afterwards.
import { readFileSync, writeFileSync, rmSync, mkdtempSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const argv = process.argv.slice(2), flag = (n) => argv.includes('--' + n), opt = (n, d) => { const i = argv.indexOf('--' + n); return i < 0 ? d : argv[i + 1]; };
const root = resolve(argv[0]), scenario = readFileSync(argv[1], 'utf8');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', PORT = 8123;

let html = readFileSync(join(root, 'vit', 'index.html'), 'utf8');
const end = html.lastIndexOf('</script>');
html = html.slice(0, end) + `\n;(async () => { const R = {}; try { ${scenario}\n } catch (e) { R.error = String(e && e.stack || e); } document.body.dataset.check = JSON.stringify(R); })();\n` + html.slice(end);
if (flag('dark')) html = html.replace('<html lang="en">', '<html lang="en" data-theme="dark">');
if (flag('no-weights')) html = html.replace("'../models/vit16.json'", "'../models/missing.json'");
const page = join(root, 'vit', '_check.html');
writeFileSync(page, html);

const server = spawn('python', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', root], { stdio: 'ignore' });
const url = `http://127.0.0.1:${PORT}/vit/_check.html`;
try {
  for (let k = 0; ; k++) { try { const r = await fetch(url); await r.arrayBuffer(); if (r.ok) break; } catch { /* not up yet */ } if (k > 50) throw new Error('server did not start'); await new Promise((r) => setTimeout(r, 100)); }
  const profile = mkdtempSync(join(tmpdir(), 'vitcheck-'));
  const args = ['--headless=new', '--disable-gpu', '--no-first-run', `--user-data-dir=${profile}`, '--enable-logging=stderr', '--v=0', '--virtual-time-budget=8000', '--hide-scrollbars', `--window-size=${opt('width', '1400')},${opt('height', '1000')}`];
  const shot = opt('shot', null);
  const run = (extra) => new Promise((res) => { const p = spawn(EDGE, [...args, ...extra, url]); let out = '', err = ''; p.stdout.on('data', (d) => { out += d; }); p.stderr.on('data', (d) => { err += d; }); p.on('close', () => res({ out, err })); });
  const { out, err } = await run(['--dump-dom']);
  const m = out.match(/data-check="([^"]*)"/);
  const R = m ? JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')) : { error: 'no data-check in the DOM' };
  const errors = err.split('\n').filter((l) => /CONSOLE/.test(l) && /_check\.html|\/core\/|\/ui\//.test(l) && !/Tracking Prevention/.test(l));
  console.log(JSON.stringify({ ...R, consoleErrors: errors }, null, 1));
  if (shot) { await run([`--screenshot=${resolve(shot)}`]); console.log('screenshot:', resolve(shot)); }
} finally { server.kill(); rmSync(page, { force: true }); }
```

Create `.superpowers/check/scen_basic.js`:

```js
await ready;
R.views = views.length;
R.trained = !!wTrained; R.testAcc = meta && meta.testAcc;
R.wsrc = $('wsrc').textContent;
R.overflow = document.documentElement.scrollWidth - innerWidth;
R.ovSticky = getComputedStyle($('ov')).position;
R.ovHeight = $('ov').offsetHeight;
R.glossary = $('gloss').children.length; R.dfn = document.querySelectorAll('dfn').length;
R.arcs = !!document.querySelector('#ovLine svg.arcs');
// hover sync: a cell of z1 lights the same cells in every z1 view (overview and section 4)
setProbe({ t: 'z1', r: 11, c: 2 });
const z1views = views.filter((v) => kOf(v) === 'z1');
R.z1views = z1views.length;
R.readout = $('readout').textContent.slice(0, 120);
// sweep visits 16 positions
const seen = new Set(); for (let k = 0; k < 16; k++) { $('n1').click(); seen.add(sweep); } R.sweepPositions = seen.size;
// walk completes the row
for (let k = 0; k < 17; k++) $('n3').click(); R.walkStep = walk.step; R.m3 = $('m3').textContent.slice(0, 60);
// shuffle
$('pOff').click(); const p0 = S.prob.slice(); $('shuf').click(); const p1 = S.prob.slice();
R.offDiff = Math.max(...p0.map((x, c) => Math.abs(x - p1[c])));
$('pOn').click(); const p2 = S.prob.slice(); $('unshuf').click(); const p3 = S.prob.slice();
R.onDiff = Math.max(...p2.map((x, c) => Math.abs(x - p3[c])));
R.m2 = $('m2').textContent.slice(0, 80);
// weights switch
$('wRa').click(); R.randomNote = $('wnote').textContent.slice(0, 40); $('wTr').click(); R.trainedNote = $('wnote').textContent.slice(0, 40);
R.banner = !!document.querySelector('.banner');
setProbe(null);
```

Create `.superpowers/check/scen_hover.js`:

```js
await ready;
setProbe({ t: 'o', r: 6, c: 1 });
const lit = views.filter((v) => kOf(v) === 'digit').length; R.digitViews = lit;
R.readout = $('readout').textContent;
```

Create `.superpowers/check/scen_none.js`:

```js
await ready;
R.overflow = document.documentElement.scrollWidth - innerWidth;
R.ovSticky = getComputedStyle($('ov')).position;
R.banner = document.querySelector('.banner') && document.querySelector('.banner').textContent;
R.trainedDisabled = $('wTr').disabled; R.wsrc = $('wsrc').textContent;
```

- [ ] **Step 2: Run the basic scenario to see it fail**

Run: `node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_basic.js`
Expected: FAIL, `ENOENT: no such file or directory, open '…\vit\index.html'`.

- [ ] **Step 3: Write the page**

Create `vit/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ViT explorer — patches, tokens and attention</title>
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
  --hl:#ffd94a; --hl2:#3fae4b; --wtint:rgba(36,87,168,.07);
  --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
}}
:root[data-theme="dark"]{
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
}
html{scroll-padding-top:env(safe-area-inset-top,0px);}
*{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
main{max-width:1240px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:28px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{margin:0 0 10px;max-width:68ch}
.lede{color:var(--ink2);font-size:15.5px;margin-bottom:10px;max-width:78ch}
.keys{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-bottom:18px}
.keys span{display:inline-flex;align-items:center;gap:6px}
.keys i{display:inline-block;width:14px;height:14px;border-radius:3px;border:3px solid var(--hl)}
.keys i.g{border-color:var(--hl2);background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--hl2) 45%,transparent))}
.keys i.wt{border:1px solid var(--accent);background:var(--wtint)}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:22px;scroll-margin-top:var(--ovh,0px)}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.head .n{font-family:var(--mono);color:var(--ink2);font-size:13px}
.head p{color:var(--ink2);margin:0;flex:0 0 100%;font-size:14px;max-width:none;padding-right:max(0px,calc(100% - 92ch))}
canvas{display:block;touch-action:none;border-radius:3px}
.ctrls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
button,select{font:inherit;font-size:13.5px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.sel{background:var(--accent-soft);border-color:var(--accent)}
label.ctl{display:inline-flex;align-items:center;gap:6px;font-size:13.5px;color:var(--ink2)}
input[type=range]{accent-color:var(--accent)}
.toggle{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.toggle button{border:0;border-radius:0}
.toggle button+button{border-left:1px solid var(--line)}
.math{font-family:var(--mono);font-size:12.5px;line-height:1.7;background:var(--line2);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre-wrap;min-height:3.2em;margin-top:10px}
.math .sum{color:var(--accent);font-weight:600}
.note{font-size:13.5px;color:var(--ink2);border-left:3px solid var(--accent-soft);padding:4px 12px;margin-top:10px;max-width:78ch}
.banner{font-size:13.5px;background:var(--accent-soft);border:1px solid var(--accent);border-radius:8px;padding:8px 12px;margin:0 0 14px;max-width:78ch}
.crumb{font-size:13px;color:var(--ink2);margin:0 0 10px}.crumb a{color:var(--accent);text-decoration:none}.crumb a:hover{text-decoration:underline}
/* the pinned overview */
.ov{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin:0 -20px 22px;padding:8px 20px 6px}
.ov .ctrls{margin:0 0 6px;gap:6px}
.ov button{font-size:12.5px;padding:3px 8px}
.ov .digits button{min-width:26px;padding:3px 0}
.wsrc{font-size:12.5px;color:var(--ink2)}
.wsrc b{color:var(--ink)}
.rbar{display:flex;gap:6px 18px;flex-wrap:wrap;align-items:baseline;margin-top:2px}
.readout{flex:1 1 480px;font-size:13px;color:var(--ink2);min-height:1.5em}
.readout b{color:var(--ink);font-weight:600}
@media (max-width:700px){ .ov{position:static} }
/* a line of tensors: storey (keys and values) over the token rows, labels under */
.wrap{overflow-x:auto;padding-bottom:4px}
.line{display:grid;grid-template-rows:auto auto auto;column-gap:8px;row-gap:3px;width:max-content;align-items:start;position:relative;padding-bottom:14px}
.line .up{grid-row:1;align-self:end;justify-self:center;display:grid;gap:3px}
.line .at{grid-row:2;display:grid;gap:3px;justify-items:start;justify-self:center}
.line .at.w{background:var(--wtint);border:1px solid var(--accent);border-radius:4px;padding:2px}
.line .lb{grid-row:3;font-size:11.5px;color:var(--ink2);text-align:center;white-space:nowrap;line-height:1.25}
.line .op{grid-row:2;margin-top:var(--mid);font-family:var(--mono);font-size:12px;color:var(--ink2);white-space:nowrap}
.line .sl{font-size:11px;color:var(--ink2)}
.line [data-sec]{cursor:pointer}
.kg{display:grid;gap:3px}
.kg canvas.pick{cursor:pointer}
svg.arcs{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
@media (prefers-reduced-motion: reduce){ *{transition:none!important;scroll-behavior:auto!important} }
</style>
</head>
<body>
<main>
<nav class="crumb"><a href="../">Visualising AI</a> › Vision Transformer, explorer</nav>
<h1>ViT explorer</h1>
<p class="lede">A ViT diagram says "patch embedding → + position → encoder (Norm → multi-head attention → + → Norm → MLP → +) → classifier". This page runs one small enough that every number fits on screen: a 16×16 digit cut into 16 patches, 8 numbers per token, one encoder block with two attention heads, <span id="pc">1,242</span> weights. The line below is the whole forward pass, token i always on row i; the sections zoom into one part at a time.</p>
<div class="keys"><span><i></i> the window being computed: one patch, or one score</span><span><i class="g"></i> what a hovered value is computed from; after attention, shaded by attention weight</span><span><i class="wt"></i> learned weights, where they act</span></div>

<div class="ov" id="ov">
  <div class="ctrls">
    <span class="digits" id="digits"></span><button id="another">another</button><span class="wsrc" id="dlab"></span>
    <span class="toggle"><button id="wTr" class="sel">trained</button><button id="wRa">random</button></span>
    <span class="toggle"><button id="pOn" class="sel">position on</button><button id="pOff">off</button></span>
    <button id="shuf">Shuffle patches</button><button id="unshuf" disabled>Restore order</button>
  </div>
  <div class="wrap"><div id="ovLine"></div></div>
  <div class="rbar"><div class="readout" id="readout"></div><div class="wsrc" id="wsrc"></div></div>
</div>

<section id="s1">
  <div class="head"><h2>1 · Patches become tokens</h2><span class="n">4×4 kernels, stride 4 → 16 tokens × 8</span>
    <p>The patch embedding is a convolution whose kernel is as big as its stride: 4×4 kernels jumping 4 pixels at a time, so the windows tile the digit without overlapping. Each of the 8 kernels gives one number per patch, so each patch becomes a token of 8 numbers, one row. It is the CNN page's kernel, jumping instead of sliding. Click a kernel to write its column out.</p></div>
  <div class="wrap"><div id="l1"></div></div>
  <div class="ctrls"><button id="p1">← prev</button><button id="play1" class="primary">Play</button><button id="n1">next →</button>
    <input type="range" id="sl1" min="0" max="15" value="0">
    <label class="ctl">speed <select id="sp1"><option value="500">slow</option><option value="220" selected>normal</option><option value="80">fast</option></select></label></div>
  <div class="math" id="m1"></div>
</section>

<section id="s2">
  <div class="head"><h2>2 · Position</h2><span class="n">z₀ = tokens + position table</span>
    <p>Attention by itself does not know where a patch came from: shuffle the patches and, without position, every token keeps its content and the average of the tokens comes out the same. So a learned position table, one row of 8 numbers per slot, is added to the tokens. Try "Shuffle patches" at the top, with position on and then off.</p></div>
  <div class="wrap"><div id="l2"></div></div>
  <div class="math" id="m2"></div>
</section>

<section id="s3">
  <div class="head"><h2>3 · One attention head, written out</h2><span class="n">A = softmax(Q·Kᵀ / 2), output = A·V</span>
    <p>Each token makes a query, a key and a value from its LayerNorm output: three learned 8×8 matrices, of which each head uses 4 columns. Token i's score for token j is query i · key j / 2; a softmax along the row turns the scores into weights that sum to 1; token i's output is the weighted sum of the values. Keys and values stand as columns above the map, one per token, so each row of the map weights the columns directly above it.</p></div>
  <div class="ctrls"><span class="toggle"><button id="hd0" class="sel">head 1</button><button id="hd1">head 2</button></span>
    <label class="ctl">query token <input type="range" id="qi" min="0" max="15" value="0"><b id="qil">0</b></label></div>
  <div class="wrap"><div id="l3"></div></div>
  <div class="ctrls"><button id="p3">← prev</button><button id="play3" class="primary">Play</button><button id="n3">next →</button>
    <input type="range" id="sl3" min="0" max="17" value="0">
    <label class="ctl">speed <select id="sp3"><option value="500">slow</option><option value="220" selected>normal</option><option value="80">fast</option></select></label></div>
  <div class="math" id="m3"></div>
  <div class="note">Before this step a token had seen only its own patch; after it, every patch. The attention map is a receptive field that is global in one step, and computed from the image rather than fixed, unlike a kernel's window. The price: 16 tokens make 256 scores per head, and the count grows with the square of the number of patches, which is why real ViTs use large patches.</div>
</section>

<section id="s4">
  <div class="head"><h2>4 · Mix the heads, residual, MLP</h2><span class="n">z₁ = z₀ + heads·Wo, z₂ = z₁ + MLP(LayerNorm(z₁))</span>
    <p>The two heads' outputs are joined back into 8 numbers and mixed by Wo, and the block's input z₀ is added back, a residual connection, giving z₁. Then LayerNorm and an MLP, 8 → 32 → 8 with GELU, and z₁ is added back again. Every step here works on one token at a time, like a 1×1 convolution; attention was the only step where tokens met. A real ViT repeats this block 12 times or more.</p></div>
  <div class="wrap"><div id="l4"></div></div>
</section>

<section id="s5">
  <div class="head"><h2>5 · Average the tokens, then predict</h2><span class="n">mean of 16 tokens → LayerNorm → W head → softmax</span>
    <p>The 16 tokens are averaged into 8 numbers, as the CNN's global average does with its maps; LayerNorm and W head turn them into 10 scores, and softmax into probabilities. The original ViT reads an extra class token instead; with a single block only that token's row would reach the answer, so this toy uses the average of the tokens, as many current ViTs do.</p></div>
  <div class="wrap"><div id="l5"></div></div>
  <div class="math" id="m5"></div>
  <div class="note" id="wnote"></div>
</section>

<section id="s6">
  <div class="head"><h2>Terms</h2><span class="n">dotted words above point here</span>
    <p>Hover or tap a dotted term anywhere above for the same definition in place.</p></div>
  <div class="gloss" id="gloss"></div>
</section>
</main>
<div id="tip" role="tooltip"></div>

<script type="module">
import { SHAPES, P, T, D, DH, NC, init, forward, tensor, vitTrace, isWeights, paramCount } from '../core/vit.js';
import { parseMnist16, image } from '../core/data.js';
import { shuffledOrder } from '../core/train.js';
import { drawGrid, cellAt, fmt, maxAbs, cssVar } from '../ui/grid.js';
import { installTerms, VIT_TERMS } from '../ui/terms.js';
const $ = (id) => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const pct = (x) => (100 * x).toFixed(1) + '%';

// ---------- state ----------
const wRandom = init(1);
let wTrained = null, meta = null, useTrained = true;
let data = null, byClass = [], pick = { cls: 7, n: 0 }, img = null;
let position = true, order = null, shuffles = 0;
let probe = null, head = 0, kSel = 0, sweep = 0, walk = { i: 0, step: 0 };
let S = null;
const W = () => (useTrained && wTrained ? wTrained : wRandom);

// ---------- views: one canvas showing (part of) a tensor, by its SHAPES key ----------
const views = []; // { cv, key, rows, cols, cell, map, yl, own, bars, blocks }
const kOf = (v) => (typeof v.key === 'function' ? v.key() : v.key);
const el = (cls, text) => { const d = document.createElement('div'); d.className = cls; if (text != null) d.textContent = text; return d; };
/** A canvas for tensor `key` at `cell` px a cell. o.rows, o.cols and o.map(r, c) → [row, col] in the tensor show a part
 *  of it; o.yl() gives the yellow ({ win, cell }); o.own() replaces the hover's green; o.bars draws probabilities. */
function view(key, cell, o = {}) {
  const [R, C] = SHAPES[typeof key === 'function' ? key() : key];
  const v = { key, cell, rows: o.rows ?? R, cols: o.cols ?? C, map: o.map || ((r, c) => [r, c]), yl: o.yl, own: o.own, bars: o.bars, blocks: o.blocks };
  const cv = document.createElement('canvas'); v.cv = cv;
  cv.width = o.px ? o.px[0] : v.cols * cell; cv.height = o.px ? o.px[1] : v.rows * cell;
  cv.style.width = cv.width + 'px'; cv.style.height = cv.height + 'px';
  cv.addEventListener('pointermove', (ev) => { const p = v.bars ? barAt(v, ev) : cellAt(cv, ev); setProbe(p ? { t: kOf(v), ...rc(v.map(p[0], p[1])) } : null); });
  cv.addEventListener('pointerleave', () => setProbe(null));
  views.push(v); return cv;
}
const rc = ([r, c]) => ({ r, c });
function setProbe(p) { if (JSON.stringify(p) !== JSON.stringify(probe)) { probe = p; render(); } }

/** Lay out columns on a grid: { op, id } is an operator; { v: [elements], up, lb, w, grid, sec, id } a column, `up` above
 *  the token rows (keys and values), `w` tinted as learned, `grid` its number of sub-columns, `sec` the section a click
 *  on it scrolls to. */
function line(box, items, cell) {
  box.className = 'line'; box.innerHTML = ''; box.style.setProperty('--mid', Math.max(0, T * cell / 2 - 9) + 'px');
  items.forEach((it, x) => {
    const col = String(x + 1);
    if (it.op != null) { const o = el('op', it.op); o.style.gridColumn = col; if (it.id) o.dataset.id = it.id; box.append(o); return; }
    const parts = [];
    if (it.up) { const u = el('up'); u.append(...it.up); parts.push(u); }
    const a = el('at' + (it.w ? ' w' : '')); if (it.grid) a.style.gridTemplateColumns = `repeat(${it.grid}, auto)`; a.append(...it.v); if (it.id) a.dataset.id = it.id; parts.push(a);
    parts.push(el('lb', it.lb || ''));
    for (const p of parts) { p.style.gridColumn = col; if (it.sec) p.dataset.sec = it.sec; box.append(p); }
    a.style.marginTop = Math.max(0, (T * cell - a.offsetHeight) / 2) + 'px'; // shorter columns sit level with the middle of the tokens
  });
}
/** Dashed residual arcs under a line's labels, from a tensor's column to the + that adds it back. */
function arcs(box, pairs) {
  box.querySelector('svg.arcs')?.remove();
  const b = box.getBoundingClientRect(); if (!b.width) return;
  const labs = [...box.querySelectorAll('.lb')].map((l) => l.getBoundingClientRect().bottom - b.top);
  const y = Math.max(...labs) + 6; let d = '';
  for (const [from, to] of pairs) {
    const A = box.querySelector(`[data-id="${from}"]`).getBoundingClientRect(), B = box.querySelector(`[data-id="${to}"]`).getBoundingClientRect();
    const xa = A.left + A.width / 2 - b.left, xb = B.left + B.width / 2 - b.left;
    d += `M${xa} ${y - 5}V${y}H${xb}V${B.bottom - b.top + 2}`;
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.classList.add('arcs');
  svg.setAttribute('width', b.width); svg.setAttribute('height', y + 4);
  svg.innerHTML = `<path d="${d}" fill="none" stroke="var(--ink2)" stroke-width="1" stroke-dasharray="3 2"/>`;
  box.append(svg);
}
const kernels = (cell, cols, pickable) => {
  const g = el('kg'); g.style.gridTemplateColumns = `repeat(${cols}, auto)`;
  for (let d = 0; d < D; d++) {
    const cv = view('Wp', cell, { rows: P, cols: P, map: (r, c) => [d, r * P + c], yl: pickable ? () => (d === kSel ? { win: [0, P - 1, 0, P - 1] } : {}) : null });
    if (pickable) { cv.classList.add('pick'); cv.title = `kernel ${d + 1}: write out column ${d}`; cv.onclick = () => { kSel = d; render(); }; }
    g.append(cv);
  }
  return g;
};
const headCols = (key, cell, h, yl) => view(key, cell, { cols: DH, map: (r, c) => [r, h() * DH + c], yl });
const headRows = (key, cell, h, yl) => view(key, cell, { rows: DH, cols: T, map: (r, c) => [c, h() * DH + r], yl });
const bias = (key, cell, h) => view(key, cell, h ? { cols: DH, map: (r, c) => [0, h() * DH + c] } : {});
const bars = (wd, ht) => view('prob', 1, { rows: NC, cols: 1, map: (r) => [0, r], bars: true, px: [wd, ht] });
const blockWin = (s) => [(s >> 2) * P, (s >> 2) * P + P - 1, (s & 3) * P, (s & 3) * P + P - 1];

// ---------- the lines: the overview and the five sections ----------
function build() {
  const A = 4, Wc = 3, fixed = (h) => () => h;
  const ovHead = (h) => [
    { v: [headCols('q', A, fixed(h))], lb: `Q${h + 1}`, sec: 's3' }, { op: '·' },
    { up: [headRows('k', A, fixed(h)), headRows('v', A, fixed(h))], v: [view('att' + h, A)], lb: `head ${h + 1}`, sec: 's3' }, { op: '=' },
    { v: [headCols('o', A, fixed(h))], lb: 'A·V', sec: 's3' }];
  line($('ovLine'), [
    { v: [view('digit', A, { blocks: P })], lb: 'digit', sec: 's1' }, { op: '⋆' },
    { v: [kernels(Wc, 2, false), view('bp', Wc)], w: true, lb: 'kernels', sec: 's1' }, { op: '=' },
    { v: [view('xp', A)], lb: 'tokens', sec: 's1' }, { op: '+' },
    { v: [view('pos', A)], w: true, lb: 'position', sec: 's2' }, { op: '=' },
    { v: [view('z0', A)], lb: 'z₀', sec: 's2', id: 'z0' }, { op: 'LN' },
    { v: [view('ln1', Wc)], w: true, lb: 'LN₁', sec: 's3' }, { op: '·' },
    { v: [view('Wq', Wc), view('Wk', Wc), view('Wv', Wc), view('bq', Wc), view('bk', Wc), view('bv', Wc)], grid: 3, w: true, lb: 'Wq Wk Wv', sec: 's3' }, { op: '=' },
    ...ovHead(0), { op: '' }, ...ovHead(1), { op: '·' },
    { v: [view('Wo', Wc), view('bo', Wc)], w: true, lb: 'Wo', sec: 's4' }, { op: '+', id: 'add1' },
    { v: [view('z1', A)], lb: 'z₁', sec: 's4', id: 'z1' }, { op: 'LN' },
    { v: [view('ln2', Wc)], w: true, lb: 'LN₂', sec: 's4' }, { op: '·' },
    { v: [view('W1', Wc), view('b1', Wc)], w: true, lb: 'W₁', sec: 's4' }, { op: '=' },
    { v: [view('hid', A)], lb: 'hidden, GELU', sec: 's4' }, { op: '·' },
    { v: [view('W2', 2), view('b2', 2)], w: true, lb: 'W₂', sec: 's4' }, { op: '+', id: 'add2' },
    { v: [view('z2', A)], lb: 'z₂', sec: 's4' }, { op: 'mean' },
    { v: [view('avg', A)], lb: 'average', sec: 's5' }, { op: 'LN' },
    { v: [view('lnf', Wc)], w: true, lb: 'LN', sec: 's5' }, { op: '=' },
    { v: [view('avgNorm', A)], lb: '', sec: 's5' }, { op: '·' },
    { v: [view('Wh', Wc), view('bh', Wc)], w: true, lb: 'W head', sec: 's5' }, { op: '=' },
    { v: [bars(60, T * A)], lb: 'classes', sec: 's5' },
  ], A);

  line($('l1'), [
    { v: [view('digit', 16, { blocks: P, yl: () => ({ win: blockWin(sweep) }) })], lb: 'digit, cut into 16 patches' }, { op: '⋆' },
    { v: [kernels(12, 4, true), view('bp', 12, { yl: () => ({ cell: [0, kSel] }) })], w: true, lb: '8 kernels and their biases' }, { op: '=' },
    { v: [view('xp', 14, { yl: () => ({ win: [sweep, sweep, 0, D - 1], cell: [sweep, kSel] }) })], lb: 'patch tokens, 16 × 8' },
  ], 14);

  line($('l2'), [
    { v: [view('xp', 13)], lb: 'patch tokens' }, { op: '+' },
    { v: [view('pos', 13)], w: true, lb: 'position table' }, { op: '=' },
    { v: [view('z0', 13)], lb: 'z₀' },
  ], 13);

  const hd = () => head, j = () => walk.step, i = () => walk.i, last = T + 1;
  line($('l3'), [
    { v: [view('h1', 13)], lb: 'LayerNorm(z₀)' }, { op: '·' },
    { v: [headCols('Wq', 9, hd), bias('bq', 9, hd), headCols('Wk', 9, hd), bias('bk', 9, hd), headCols('Wv', 9, hd), bias('bv', 9, hd)], w: true, lb: 'this head’s 4 columns' }, { op: '=' },
    { v: [headCols('q', 14, hd, () => ({ win: [i(), i(), 0, DH - 1] }))], lb: 'Q' }, { op: '·' },
    { up: [el('sl', 'Kᵀ: one key per token'), headRows('k', 14, hd, () => (j() < T ? { win: [0, DH - 1, j(), j()] } : {})),
      el('sl', 'Vᵀ: one value per token'), headRows('v', 14, hd, () => (j() === last ? { win: [0, DH - 1, 0, T - 1] } : {}))],
    v: [view(() => 'att' + head, 14, { yl: () => (j() < T ? { cell: [i(), j()] } : { win: [i(), i(), 0, T - 1] }) })], lb: 'A = softmax(Q·Kᵀ / 2)' }, { op: '=' },
    { v: [headCols('o', 14, hd, () => (j() === last ? { win: [i(), i(), 0, DH - 1] } : {}))], lb: 'A·V' }, { op: '' },
    { v: [view('digit', 9, { blocks: P, own: () => vitTrace(S, { t: 'o', r: walk.i, c: head * DH }).digit })], lb: 'where this token looked' },
  ], 14);

  line($('l4'), [
    { v: [view('o', 11)], lb: 'heads joined' }, { op: '·' },
    { v: [view('Wo', 9), view('bo', 9)], w: true, lb: 'Wo' }, { op: '+ z₀' },
    { v: [view('z1', 11)], lb: 'z₁' }, { op: 'LN' },
    { v: [view('ln2', 9)], w: true, lb: 'LN₂' }, { op: '·' },
    { v: [view('W1', 7), view('b1', 7)], w: true, lb: 'W₁, 8 → 32' }, { op: '=' },
    { v: [view('hid', 8)], lb: 'hidden, after GELU' }, { op: '·' },
    { v: [view('W2', 7), view('b2', 7)], w: true, lb: 'W₂, 32 → 8' }, { op: '+ z₁' },
    { v: [view('z2', 11)], lb: 'z₂' },
  ], 11);

  line($('l5'), [
    { v: [view('z2', 11)], lb: 'z₂' }, { op: 'mean' },
    { v: [view('avg', 18)], lb: 'average' }, { op: 'LN' },
    { v: [view('lnf', 18)], w: true, lb: 'scale, shift' }, { op: '=' },
    { v: [view('avgNorm', 18)], lb: 'normalised' }, { op: '·' },
    { v: [view('Wh', 18), view('bh', 18)], w: true, lb: 'W head, bias' }, { op: '=' },
    { v: [bars(240, 220)], lb: 'class probabilities' },
  ], 11);

  $('ovLine').addEventListener('click', (e) => { const s = e.target.closest('[data-sec]'); if (s) $(s.dataset.sec).scrollIntoView({ behavior: reduce ? 'auto' : 'smooth' }); });
}
const layout = () => { arcs($('ovLine'), [['z0', 'add1'], ['z1', 'add2']]); document.documentElement.style.setProperty('--ovh', ($('ov').offsetHeight + 10) + 'px'); };

// ---------- probabilities as bars ----------
function drawBars(v, vals, glow) {
  const { cv } = v, ctx = cv.getContext('2d'), h = cv.height / NC, wd = cv.width, top = vals.indexOf(Math.max(...vals));
  const lab = h >= 14, x0 = lab ? 16 : 0, x1 = lab ? wd - 46 : wd;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, wd, cv.height);
  ctx.font = `${Math.min(12, h * 0.6)}px ${getComputedStyle(document.body).fontFamily}`; ctx.textBaseline = 'middle';
  for (let c = 0; c < NC; c++) {
    const y = c * h;
    ctx.fillStyle = cssVar('--line2'); ctx.fillRect(x0, y + 1, x1 - x0, h - 2);
    ctx.fillStyle = cssVar('--accent'); ctx.globalAlpha = c === top ? 1 : 0.6; ctx.fillRect(x0, y + 1, (x1 - x0) * vals[c], h - 2); ctx.globalAlpha = 1;
    if (glow && glow[c] > 0) { ctx.strokeStyle = cssVar('--hl2'); ctx.lineWidth = 2; ctx.strokeRect(x0 + 1, y + 1, x1 - x0 - 2, h - 2); }
    if (lab) { ctx.fillStyle = cssVar('--ink2'); ctx.textAlign = 'left'; ctx.fillText(String(c), 2, y + h / 2); ctx.textAlign = 'right'; ctx.fillText(pct(vals[c]), wd - 2, y + h / 2); }
  }
}
function barAt(v, ev) { const b = v.cv.getBoundingClientRect(), r = Math.floor((ev.clientY - b.top) / (b.height / NC)); return r >= 0 && r < NC ? [r, 0] : null; }

// ---------- render ----------
function render() {
  if (!img) return;
  const w = W();
  S = forward(w, img, { position, order });
  const green = vitTrace(S, probe), scale = {};
  for (const v of views) {
    const key = kOf(v), M = tensor(key, S, w), C = SHAPES[key][1];
    const G = v.own ? v.own() : green[key];
    const vals = new Float32Array(v.rows * v.cols), glow = G ? new Float32Array(v.rows * v.cols) : null;
    for (let r = 0; r < v.rows; r++) for (let c = 0; c < v.cols; c++) {
      const [mr, mc] = v.map(r, c); vals[r * v.cols + c] = M[mr][mc]; if (glow) glow[r * v.cols + c] = G[mr * C + mc];
    }
    if (v.bars) { drawBars(v, vals, glow); continue; }
    const s = key === 'digit' ? 1 : (scale[key] ??= maxAbs(M.flat()));
    const y = v.yl ? v.yl() : {};
    drawGrid(v.cv, vals, v.cols, v.rows, { mode: key === 'digit' ? null : 'div', scale: s, glow, lines: v.cell >= 7, blocks: v.blocks, win: y.win, cell: y.cell });
  }
  $('readout').innerHTML = describe(probe);
  $('m1').innerHTML = patchMath(); $('m2').textContent = shuffleText(); $('m3').innerHTML = attentionMath(); $('m5').innerHTML = headMath();
  wording();
  $('sl1').value = sweep; $('sl3').value = walk.step; $('qi').value = walk.i; $('qil').textContent = walk.i;
}

// ---------- readouts ----------
const NAME = { xp: 'patch token', z0: 'z₀', h1: 'LayerNorm 1 output', q: 'query', k: 'key', v: 'value', o: 'heads joined', z1: 'z₁', h2: 'LayerNorm 2 output', hid: 'MLP hidden', z2: 'z₂',
  Wp: 'patch kernels', bp: 'patch bias', pos: 'position table', ln1: 'LayerNorm 1 (row 0 scale, row 1 shift)', ln2: 'LayerNorm 2 (row 0 scale, row 1 shift)', lnf: 'final LayerNorm (row 0 scale, row 1 shift)',
  Wq: 'Wq', bq: 'bq', Wk: 'Wk', bk: 'bk', Wv: 'Wv', bv: 'bv', Wo: 'Wo', bo: 'bo', W1: 'W₁', b1: 'b₁', W2: 'W₂', b2: 'b₂', Wh: 'W head', bh: 'head bias' };
const meanAtt = () => S.att[0].map((r, a) => r.map((x, b) => (x + S.att[1][a][b]) / 2));
function describe(p) {
  if (!p) return 'Hover any cell: green shows what it is computed from. After attention, its strength is the attention weight.';
  const val = tensor(p.t, S, W())[p.r][p.c], at = `<b>${NAME[p.t] || p.t}</b>`;
  if (p.t === 'digit') { const s = (p.r >> 2) * 4 + (p.c >> 2); return `<b>pixel (${p.r}, ${p.c})</b> = ${fmt(val)}, in patch ${s}, which becomes token ${s}.`; }
  if (p.t === 'att0' || p.t === 'att1') {
    const h = +p.t[3];
    return `<b>head ${h + 1}</b>: token ${p.r} gives token ${p.c} the weight <b>${fmt(val)}</b> (score ${fmt(S.scores[h][p.r][p.c])}); each row sums to 1. Green: query ${p.r} and key ${p.c}, and every other key faintly, since the softmax divides by all of them.`;
  }
  if (['xp', 'z0', 'h1', 'q', 'k', 'v'].includes(p.t)) return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. Before attention a token has seen only its own patch.`;
  if (['o', 'z1', 'h2', 'hid', 'z2'].includes(p.t)) {
    const one = p.t === 'o', A = one ? S.att[Math.floor(p.c / DH)] : meanAtt();
    const top = A[p.r].map((a, j) => [a, j]).filter(([, j]) => j !== p.r).sort((x, y) => y[0] - x[0]).slice(0, 3).map(([a, j]) => `token ${j} (${fmt(a)})`).join(', ');
    return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. After attention: its own patch full green, every other patch by the attention weight it got${one ? ` in head ${Math.floor(p.c / DH) + 1}` : ', both heads averaged'}, scaled to the largest (attention, not importance). Most attended: ${top}.`;
  }
  if (p.t === 'avg') return `<b>average, column ${p.c}</b> = ${fmt(val)}: column ${p.c} of z₂ averaged over the 16 tokens, as the CNN averages each map.`;
  if (p.t === 'avgNorm') return `<b>normalised average, column ${p.c}</b> = ${fmt(val)}.`;
  if (p.t === 'prob') return `<b>class ${p.c}</b>: ${pct(val)}. It reads only the 8 normalised averages, through column ${p.c} of W head and its bias.`;
  return `${at} [${p.r}, ${p.c}] = ${fmt(val)}, learned. Green: what it reads and what it makes.`;
}
function patchMath() {
  const w = W(), px = S.patches[sweep], k = w.Wp[kSel], t = px.map((x, q) => `${fmt(x)}×${fmt(k[q])}`);
  return `token ${sweep}, column ${kSel} = Σ pixel × kernel ${kSel + 1} + bias   (patch ${sweep}: rows ${(sweep >> 2) * 4}–${(sweep >> 2) * 4 + 3}, cols ${(sweep & 3) * 4}–${(sweep & 3) * 4 + 3})\n`
    + `  = ${t.slice(0, 8).join(' + ')}\n  + ${t.slice(8).join(' + ')}\n  + ${fmt(w.bp[kSel])}\n  = <span class="sum">${fmt(S.xp[sweep][kSel])}</span>`;
}
function shuffleText() {
  if (!order) return 'Patches in reading order. Press "Shuffle patches" at the top, with position on and then off.';
  const base = forward(W(), img, { position }), d = Math.max(...base.prob.map((x, c) => Math.abs(x - S.prob[c])));
  const top = (pr) => { const c = pr.indexOf(Math.max(...pr)); return `${c} (${pct(pr[c])})`; };
  return `reading order: ${top(base.prob)}\nshuffled:      ${top(S.prob)}\n` + (position
    ? `different (largest change ${pct(d)}): each slot has its own row of the position table, so moving a patch changes what the model sees.`
    : `identical (largest difference ${d.toExponential(1)}): without position, attention treats the tokens as a set, and their average ignores order.`);
}
function attentionMath() {
  const h = head, off = h * DH, i = walk.i, s = walk.step, q = S.q[i].slice(off, off + DH);
  if (s < T) {
    const k = S.k[s].slice(off, off + DH);
    return `head ${h + 1}, score(${i}, ${s}) = query ${i} · key ${s} / 2\n  = (${q.map((x, c) => `${fmt(x)}×${fmt(k[c])}`).join(' + ')}) / 2\n  = <span class="sum">${fmt(S.scores[h][i][s])}</span>`;
  }
  if (s === T) return `head ${h + 1}, row ${i}: weights = softmax(scores) = exp(score) / Σ exp(score)\n  scores  ${S.scores[h][i].map(fmt).join('  ')}\n  weights ${S.att[h][i].map(fmt).join('  ')}   (sum 1)`;
  const A = S.att[h][i];
  return `head ${h + 1}, output of token ${i} = Σ_j weight(${i}, j) × value j, for each of its 4 numbers:\n` + [0, 1, 2, 3].map((c) => {
    const terms = A.map((a, j) => [a * S.v[j][off + c], a, j]).sort((x, y) => Math.abs(y[0]) - Math.abs(x[0]));
    return `  out[${c}] = ${terms.slice(0, 3).map(([, a, j]) => `${fmt(a)}×${fmt(S.v[j][off + c])}`).join(' + ')} + (13 smaller) = <span class="sum">${fmt(S.heads[h][i][c])}</span>`;
  }).join('\n');
}
function headMath() {
  const w = W();
  if (!probe || probe.t !== 'prob') return 'scores = normalised average · W head + bias, then softmax. Hover a class for its numbers.';
  const c = probe.c;
  return `score(${c}) = ${S.avgNorm.map((x, d) => `${fmt(x)}×${fmt(w.Wh[d][c])}`).join(' + ')} + ${fmt(w.bh[c])}\n  = <span class="sum">${fmt(S.logits[c])}</span>\nsoftmax: exp(score(${c})) / Σ exp(score) = <span class="sum">${pct(S.prob[c])}</span>`;
}
function wording() {
  const top = S.prob.indexOf(Math.max(...S.prob)), trained = useTrained && wTrained;
  const opt = meta && (meta.optimiser === 'adam' ? 'Adam' : 'SGD');
  $('wsrc').innerHTML = `prediction <b>${top}</b> (${pct(S.prob[top])}), ` + (trained ? `trained weights: ${pct(meta.testAcc)} right on held-out digits` : 'random weights: arbitrary');
  $('wnote').textContent = trained
    ? `These weights were trained offline by tools/train_vit.mjs on the 3,000 training digits (${meta.passes} passes, ${opt}); they get ${pct(meta.testAcc)} of the 1,000 held-out digits right, ${meta.testAcc >= 0.8 ? 'about what' : 'less than'} the CNN pages' toy reaches on the same digits (80 to 85%). On real photographs a ViT needs far more training data than a CNN to catch up: it has to learn from examples the locality a convolution has built in.`
    : 'These are random starting weights, before any training. Attention is learned: before training its maps are noise and the prediction is arbitrary.';
}

// ---------- controls ----------
function setDigit() { const list = byClass[pick.cls], i = list[pick.n % list.length]; img = image(data.test, i); $('dlab').textContent = `test digit ${i}, a ${pick.cls}`; }
function syncControls() {
  [...$('digits').children].forEach((b, c) => b.classList.toggle('sel', c === pick.cls));
  $('wTr').classList.toggle('sel', useTrained && !!wTrained); $('wRa').classList.toggle('sel', !(useTrained && wTrained)); $('wTr').disabled = !wTrained;
  $('pOn').classList.toggle('sel', position); $('pOff').classList.toggle('sel', !position); $('unshuf').disabled = !order;
  $('hd0').classList.toggle('sel', head === 0); $('hd1').classList.toggle('sel', head === 1);
}
for (let c = 0; c < NC; c++) { const b = document.createElement('button'); b.textContent = c; b.onclick = () => { pick = { cls: c, n: 0 }; setDigit(); syncControls(); render(); }; $('digits').append(b); }
$('another').onclick = () => { pick.n++; setDigit(); render(); };
const on = (id, f) => { $(id).onclick = () => { f(); syncControls(); render(); }; };
on('wTr', () => { useTrained = true; }); on('wRa', () => { useTrained = false; });
on('pOn', () => { position = true; }); on('pOff', () => { position = false; });
on('shuf', () => { order = shuffledOrder(T, ++shuffles); }); on('unshuf', () => { order = null; });
on('hd0', () => { head = 0; }); on('hd1', () => { head = 1; });
$('qi').oninput = (e) => { walk.i = +e.target.value; render(); };
/** Play, step and scrub one of the two walks: get/set its step, n steps in all. Reduced motion jumps to the end. */
function player(id, n, get, set) {
  let timer = null; const btn = $('play' + id);
  const stop = () => { clearInterval(timer); timer = null; btn.textContent = 'Play'; };
  btn.onclick = () => {
    if (timer) return stop();
    if (reduce) { set(n - 1); return render(); }
    if (get() >= n - 1) set(0);
    btn.textContent = 'Pause';
    timer = setInterval(() => { set(get() + 1); render(); if (get() >= n - 1) stop(); }, +$('sp' + id).value);
  };
  $('p' + id).onclick = () => { stop(); set(Math.max(0, get() - 1)); render(); };
  $('n' + id).onclick = () => { stop(); set((get() + 1) % n); render(); };
  $('sl' + id).oninput = (e) => { stop(); set(+e.target.value); render(); };
}
player(1, T, () => sweep, (s) => { sweep = s; });
player(3, T + 2, () => walk.step, (s) => { walk.step = s; });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
addEventListener('resize', layout);

// ---------- start ----------
function banner(text) { const d = el('banner'); d.textContent = text; document.querySelector('h1').after(d); }
async function start() {
  build();
  installTerms({ selectors: '.lede, .head p, .note', tip: $('tip'), gloss: $('gloss'), terms: VIT_TERMS });
  $('pc').textContent = paramCount(wRandom).toLocaleString('en-NZ');
  const [bin, js] = await Promise.all([
    fetch('../data/mnist16.bin').then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null),
    fetch('../models/vit16.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  if (!bin) { banner('Could not load the digits (data/mnist16.bin); this page needs to be served over http.'); return; }
  data = parseMnist16(bin);
  byClass = Array.from({ length: NC }, (_, c) => [...Array(data.test.n).keys()].filter((i) => data.test.labels[i] === c));
  if (js && js.version === 1 && isWeights(js.weights)) { wTrained = js.weights; meta = js.meta; }
  else { useTrained = false; banner('Could not load the trained weights (models/vit16.json), so this page shows random starting weights.'); }
  setDigit(); syncControls(); render(); layout();
}
const ready = start();
</script>
</body>
</html>
```

- [ ] **Step 4: Run the basic scenario**

Run: `node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_basic.js`
Expected JSON (the 1400 × 1000 window):
- `"views": 91`, `"trained": true`, and `"testAcc"` equal to `meta.testAcc` in `models/vit16.json`;
- `"overflow": 0`, `"ovSticky": "sticky"`, `"ovHeight"` about 218;
- `"glossary": 18`, `"dfn"` about 41, `"arcs": true`, `"z1views": 2`;
- `"sweepPositions": 16`, `"walkStep": 17`;
- `"offDiff"` below `1e-12` (position off: shuffling changes nothing), `"onDiff"` above `0.001` (position on: it does);
- `"randomNote"` starting "These are random starting weights", `"trainedNote"` starting "These weights were trained offline";
- `"banner": false`, `"consoleErrors": []`.

- [ ] **Step 5: Phone width without the weights file, and dark at 1000 px**

Run:
```bash
node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_none.js --no-weights --width 500 --height 1600 --shot .superpowers/check/phone.png
node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_none.js --dark --width 1000 --height 900 --shot .superpowers/check/dark1000.png
```
Expected:
- **First run:** `"overflow": 0`, `"ovSticky": "static"`, a `"banner"` reading "Could not load the trained weights (models/vit16.json), so this page shows random starting weights.", `"trainedDisabled": true`, and `"wsrc"` containing "random weights".
- **Second run:** `"overflow": 0`, `"ovSticky": "sticky"`, `"banner": null`, `"consoleErrors": []`.

- [ ] **Step 6: Screenshots, viewed and checked**

Run:
```bash
node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_hover.js --shot .superpowers/check/hover.png --height 700
node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_basic.js --shot .superpowers/check/tall.png --height 4200
```
Open each PNG (`hover.png`, `tall.png`, `phone.png`, `dark1000.png`) with the Read tool and check:
- **The overview** is one line: digit, kernels, tokens, position, z₀, LN₁, Wq Wk Wv (three side by side), Q1 with head 1 below K₁ᵀ and V₁ᵀ, A·V, then the same for head 2, Wo and the rest, scrolling sideways. Weight columns are tinted with an accent border. Dashed arcs run under the labels from z₀ and z₁ to their "+".
- **`hover.png`** (the probe is token 6, column 1 of the joined heads): the digit is graded green with patch 6 outlined; z₀ and the tokens are graded row by row; head 1's row 6 is green; K₁ᵀ and V₁ᵀ are green column by column; head 2 is not lit. The readout starts "heads joined, token 6, column 1".
- **Section 1:** the yellow window on a patch, the chosen kernel framed in yellow, the token row framed, and the sum written out under them.
- **Section 3:** Kᵀ and Vᵀ above the map; the walk's yellow; the digit at the right shaded by the chosen token's attention.
- **Section 5:** bars with the top class solid, and the weights note.
- **Dark:** colours readable, the patch grid's dashes visible. **Phone:** the controls wrap, the overview scrolls inside its box, the page does not overflow.

Fix anything that reads wrong, then rerun Steps 4 and 5.

- [ ] **Step 7: Commit**

```bash
git add vit/index.html
git commit -m "feat(vit): the ViT explorer page" -m "Pinned overview line and five sections built from one column builder; graded green from vitTrace; trained weights from models/vit16.json with a random switch and a fallback banner." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Landing page, documentation, final checks, push

**Files:**
- Create: `.superpowers/check/landing_edit.py` (git-ignored)
- Modify: `index.html`, `CLAUDE.md`, `HANDOFF.md`, `project_notes.md`

- [ ] **Step 1: Write and run the landing-page edit**

Create `.superpowers/check/landing_edit.py`. It applies each replacement once and fails if any original text is not found exactly once:

```python
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
reps = [
('.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;grid-column:1/-1}',
 '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;grid-column:1/-1}\n.cards.one{grid-template-columns:minmax(0,360px)}'),
('<p class="lede">The convolutional network is built. The Vision Transformer is next, then U-Net, CoAtNet and ConvNeXt.</p>',
 '<p class="lede">The convolutional network is built, and the Vision Transformer has its explorer. Next: the Vision Transformer\'s training page, then U-Net, CoAtNet and ConvNeXt.</p>'),
('<a href="#vit">Vision Transformer <span class="badge next">next</span></a>', '<a href="#vit">Vision Transformer <span class="badge live">1 page</span></a>'),
("what a hovered cell sees, or the two sources of a weight's gradient.</span>",
 "what a hovered cell sees, or the two sources of a weight's gradient. On the ViT page, after attention, its strength is the attention weight.</span>"),
('aria-label="An image cut into 16 patches becomes a row of tokens plus a class token; in the encoder every token attends to every other; the class token gives 10 class probabilities."',
 'aria-label="An image cut into 16 patches becomes a row of tokens; in the encoder every token attends to every other; the tokens are averaged and give 10 class probabilities."'),
("Patches → tokens → attention, every token to every other → class token → 10 classes. One token's attention in green.",
 "Patches → tokens → attention, every token to every other → average → 10 classes. One token's attention in green."),
('<h2>Vision Transformer (ViT)</h2><span class="badge next">next</span>', '<h2>Vision Transformer (ViT)</h2><span class="badge live">1 page</span>'),
('''    <div class="sub">What the page will show</div>
    <ul>
      <li><b>Patch embedding</b> as a convolution whose stride equals its size: the CNN pages' kernel, jumping instead of sliding.</li>
      <li><b>Position</b> added to each token, since attention by itself does not know where a patch came from.</li>
      <li><b>One attention head</b> written out: queries against keys, softmax, a weighted sum of values.</li>
      <li><b>The attention map</b> as a receptive field that is global in one step.</li>
    </ul>
    <p class="builds">Builds on the CNN pages: stride, kernels, softmax.</p>
  </div>
</section>''',
'''    <p class="builds">Builds on the CNN pages: stride, kernels, softmax.</p>
  </div>
    <div class="cards one">
      <a class="card" href="vit/"><b><svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><g class="k-act"><rect class="f" x="2" y="2" width="22" height="22" rx="2"/></g><rect x="13" y="7.5" width="5.5" height="5.5" fill="var(--hl2)"/><rect x="7.5" y="2" width="5.5" height="5.5" fill="var(--hl2)" opacity=".55"/><rect x="2" y="18.5" width="5.5" height="5.5" fill="var(--hl2)" opacity=".3"/><path class="grid" d="M7.5 2v22M13 2v22M18.5 2v22M2 7.5h22M2 13h22M2 18.5h22"/></svg>Explorer</b>
        <span>Patches become tokens, position is added, one attention head is written out score by score, then the MLP, the residuals and the average. Trained weights; hover any value to see what it reads.</span><em>Open →</em></a>
    </div>
</section>'''),
("""for (let i = 0; i < 9; i++) s += `<rect class="${i === 0 ? 'bar' : ''}" x="${98 + i * 10}" y="56" width="6.5" height="36" rx="1.5" ${i ? 'fill="var(--tok-f)" stroke="var(--edge)" stroke-width=".8"' : ''}/>`;""",
 """for (let i = 0; i < 9; i++) s += `<rect x="${98 + i * 10}" y="56" width="6.5" height="36" rx="1.5" fill="var(--tok-f)" stroke="var(--edge)" stroke-width=".8"/>`;"""),
("""`<rect class="bar" x="360" y="56" width="6.5" height="36" rx="1.5"/>` + label(363, 140, 'class')""",
 """`<rect x="360" y="56" width="6.5" height="36" rx="1.5" fill="var(--tok-f)" stroke="var(--edge)" stroke-width=".8"/>` + label(363, 140, 'average')"""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('landing page edited')
```

Run: `python .superpowers/check/landing_edit.py index.html`
Expected: `landing page edited`.

- [ ] **Step 2: Look at the landing page**

Run:
```bash
python -m http.server 8126 --bind 127.0.0.1 &
sleep 1
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --no-first-run --user-data-dir="$(mktemp -d)" --hide-scrollbars --window-size=1400,1500 --virtual-time-budget=3000 --screenshot="$(cygpath -w .superpowers/check/landing.png)" "http://127.0.0.1:8126/index.html#vit"
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --no-first-run --user-data-dir="$(mktemp -d)" --hide-scrollbars --window-size=500,1500 --virtual-time-budget=3000 --screenshot="$(cygpath -w .superpowers/check/landing500.png)" "http://127.0.0.1:8126/index.html#vit"
kill %1
```
Open both PNGs. Check:
- The ViT section shows its badge "1 page" and one Explorer card, about 360 px wide, under the text.
- The schematic has 9 plain token bars and an "average" bar before the 10 bars.
- The caption reads "… → average → 10 classes".
- The CNN cards are unchanged.

Then dark: the landing page's CSS has a `:root[data-theme="dark"]` rule, so run `sed 's/<html lang="en">/<html lang="en" data-theme="dark">/' index.html > landing_dark.html`, screenshot `http://127.0.0.1:8126/landing_dark.html#vit` the same way (start the server again) into `.superpowers/check/landing_dark.png`, open it, then `rm landing_dark.html`. The schematic's token bars and the card's green squares should read on the dark background.

- [ ] **Step 3: Update `CLAUDE.md`**

Make these replacements with the Edit tool.

```text
(CNN built; then ViT, U-Net, CoAtNet, ConvNeXt, in that order)
```
with
```text
(CNN built; ViT explorer built; then the ViT training page, U-Net, CoAtNet, ConvNeXt, in that order)
```

```text
Tests: `node core/cnn.test.mjs`.
```
with
```text
Tests: `node core/cnn.test.mjs`, `node core/data.test.mjs`, `node core/train.test.mjs`, `node core/vit.test.mjs`.
```

```text
`terms.js` (the `TERMS` array, tooltip, glossary) and `terms.css`; `grid.js` (canvas grids, hit-testing, colours);
```
with
```text
`terms.js` (term lists, one per page family: `TERMS` for the CNN pages, `VIT_TERMS` for the ViT page; tooltip, glossary) and `terms.css`; `grid.js` (canvas grids, hit-testing, colours, and the graded green `glow` and yellow `win` the ViT page uses);
```

```text
`tools/` — the standard-library Python script that makes it;
```
with
```text
`tools/` — the standard-library Python script that makes it, and `train_vit.mjs`, which trains the ViT and writes `models/vit16.json`;
```

```text
- `cnn-2d/`, `cnn-3d/`, `cnn-train/` — one self-contained page each, with a breadcrumb back to the landing page. New pages follow the same pattern (`vit/`, `unet/`, `coatnet/`, `convnext/`).
```
with
```text
- `cnn-2d/`, `cnn-3d/`, `cnn-train/`, `vit/` — one self-contained page each, with a breadcrumb back to the landing page. New pages follow the same pattern (`vit-train/`, `unet/`, `coatnet/`, `convnext/`). `vit/` builds its pinned overview line and its five sections with one column builder: a section is the overview's columns at a larger cell size, and every copy of a tensor lights together on hover.
```

```text
- `models/` — trained weights (JSON/ONNX) when added. Empty for now.
```
with
```text
- `models/` — trained weights: `vit16.json`, the ViT page's, made by `node tools/train_vit.mjs` (the command is in its `meta.made`).
```

```text
Keep the two colours distinct and consistent across every page.
```
with
```text
After attention (ViT page) green is graded: the hovered token's own patch full, every other token by its attention weight, scaled so the most-attended reaches 0.9. Keep the two colours distinct and consistent across every page.
```

```text
its weights can be loaded into the explorer pages, which then say so in a banner and in their wording.
```
with
```text
its weights can be loaded into the explorer pages, which then say so in a banner and in their wording. The ViT page runs on weights trained offline (`models/vit16.json`), can switch to random ones, and says which, with the held-out accuracy, wherever a prediction appears.
```

- [ ] **Step 4: Update `HANDOFF.md`**

```text
   - ViT: patch embedding as a strided conv,
```
with
```text
   - ViT: _built 2026-10-28 as `vit/`, the explorer, with the tokens averaged instead of a class token (see `docs/superpowers/specs/2026-10-28-vit-explorer-design.md`); its training page `vit-train/` is next._ Patch embedding as a strided conv,
```

```text
    node core/cnn.test.mjs
```
with
```text
    node core/cnn.test.mjs
    node core/vit.test.mjs
```

- [ ] **Step 5: Update `project_notes.md` with the `project-notes` skill**

Invoke the `project-notes` skill and update "As built" only. Leave Intent untouched.
- **Header line:** last updated 2026-10-28, the ViT explorer built.
- **Overview:** four pages now. The ViT explorer shows patches → tokens → position → one encoder block with two heads → average → 10 classes, on the same digits, on weights trained offline.
- **Method**, a new bullet for the ViT:
  - the model as in the spec;
  - `core/vit.js` (forward, backward, SGD and Adam, `tensor`/`SHAPES`, `vitTrace` and its rules);
  - `tools/train_vit.mjs` and `models/vit16.json` (the chosen optimiser and learning rate);
  - the page: a pinned overview line plus five sections from one column builder, the matrix layout of attention (Kᵀ and Vᵀ above the map), the yellow patch sweep and attention walk, the shuffle demo, trained | random and the fallback banner, `VIT_TERMS`;
  - the new `ui/grid.js` options.
- **Results**, dated 2026-10-28, laptop, Node 24.19.0, headless Edge:
  - the test lines from Task 2 Step 4;
  - the six training figures and the chosen setting, and the shipped file's held-out and training accuracy;
  - the page checks from Task 5 Steps 4 and 5;
  - the CNN pages' identical DOM (Task 4);
  - the landing page check.
- **Decisions**, each with its why, from the spec's "Decisions taken in the brainstorm":
  - the sequence;
  - trained weights by default;
  - graded green;
  - two heads;
  - the pinned overview with sections (Alexy chose C);
  - attention as a matrix product;
  - the average instead of a class token (Alexy chose A, after the planning finding that with one block only the class token's row is trained);
  - the optimiser, by the 3-point rule.
- **Map:** `vit/index.html`, `core/vit.js`, `core/vit.test.mjs`, `tools/train_vit.mjs`, `models/vit16.json`, the spec and this plan.
- **Status:** ViT explorer built; next, the ViT training page (`vit-train/`), then U-Net.
- **Open questions:**
  - Do the two heads specialise? Look at both maps on a few digits and say what they show.
  - On this toy, ViT and CNN held-out accuracy are close (about 82 to 85% against 80 to 85%), so the data-hunger lesson will need the training page, or more data, to show.

- [ ] **Step 6: Final checks**

Run:
```bash
node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs && node core/vit.test.mjs
node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_basic.js
node .superpowers/check/domcheck.mjs . .superpowers/check/dom-final && diff -r .superpowers/check/dom-before .superpowers/check/dom-final && echo IDENTICAL
```
Expected:
- all four tests pass;
- the basic scenario gives the values of Task 5 Step 4;
- `IDENTICAL`, since `index.html` is not one of the CNN pages.

- [ ] **Step 7: Commit**

```bash
git add index.html
git commit -m "feat: landing page card for the ViT explorer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add CLAUDE.md HANDOFF.md project_notes.md
git commit -m "docs: record the ViT explorer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git status --short
```
Expected: `git status --short` prints nothing (`.superpowers/` is ignored).

- [ ] **Step 8: Push and check the live site**

Run:
```bash
git push origin main
gh run watch "$(gh run list --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
curl -s https://smooth-cactus0.github.io/Visualising-AI/vit/ | grep -c "ViT explorer"
curl -s -o /dev/null -w "%{http_code}\n" https://smooth-cactus0.github.io/Visualising-AI/models/vit16.json
curl -s https://smooth-cactus0.github.io/Visualising-AI/ | grep -c 'href="vit/"'
```
Expected:
- the Pages workflow succeeds;
- the first `grep -c` prints at least 1;
- the weights file returns `200`;
- the landing page links `vit/`.

Report the live URL, https://smooth-cactus0.github.io/Visualising-AI/vit/, and mention the browser's 10-minute cache (a hard refresh).

---

## Plan self-review

- **Spec coverage**, section by section:
  - **Model:** Task 1.
  - **`vitTrace` rules:** Task 2.
  - **Training script, optimiser rule, `models/vit16.json`:** Task 3.
  - **`ui/terms.js` and `ui/grid.js`:** Task 4.
  - **Page:**
    - top, pinned overview with its controls, readout, arcs and click-to-section: Task 5;
    - sections 1 to 6 and hover: Task 5;
    - wording, the random switch and the fallback banner: Task 5;
    - theme, responsive layout and reduced motion: Task 5 (CSS, `reduce`, Steps 4 to 6).
  - **Landing page:** Task 6, Steps 1 and 2.
  - **Checks:** Tasks 1 and 2 (Node), 4 (DOM), 5 and 6 (page and landing).
  - **Left for later:** not built, by design.
- **Placeholders:** the only free text is the commit body in Task 3, Step 7, which records measurements that do not exist until that step runs.
- **Names are consistent across tasks:**
  - `SHAPES` keys;
  - `vitTrace(S, { t, r, c })`;
  - `tensor(key, S, w)`;
  - `isWeights`;
  - `meta.testAcc`, `meta.passes`, `meta.optimiser`;
  - `installTerms({ …, terms })`;
  - `drawGrid` options `glow`, `win`, `blocks`, `lines`.
- **The page code depends only on these, and on `shuffledOrder`, `parseMnist16` and `image`, which already exist.**

