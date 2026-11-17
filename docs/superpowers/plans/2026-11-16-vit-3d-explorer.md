# ViT 3D Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A page, `vit-3d/`, that draws the toy ViT of `vit/` as blocks of cubes, one cube per number:
- each token sits at its patch's place, with its 8 numbers running into depth;
- each head's attention map is a slab of 16 little 4×4 maps, token i's map at token i's place;
- the whole chain is drawn, weights included, in accent frames.

Hovering a cube traces `vitTrace`'s green as graded boxes, plus a fan of links from the token's little map to the tokens it read. A click pins the probe while the camera moves. There is a patch sweep, and the controls of `vit/`.

**Architecture:**
- **`core/vit3d.js`** (pure, tested in Node):
  - which tensor each block draws, as rows × columns × slices;
  - the two-way mapping between a cube and a tensor cell;
  - the green as merged boxes;
  - the attention fan.
- **`ui/vitwords.js`:** the readout, the wording about the weights, and the loading of digits and weights. They move out of `vit/`, which is checked unchanged, so both explorers share them.
- **`ui/blocks3d.js`** gains opt-in calls that the CNN pages never make: `strengthBox`, `strengthLine`, `fit`, `snap`, `onClick`. Both CNN 3D scenes are checked unchanged.
- **`vit-3d/index.html`:** the page. A cube becomes the same probe `vit/` uses, so the green and the readout are the 2D explorer's.

Spec: `docs/superpowers/specs/2026-11-16-vit-3d-explorer-design.md`.

**Tech Stack:** Plain HTML and ES modules (no build step), three.js r128 UMD from cdnjs (as on `cnn-3d/`), Node 24 for the tests, Python 3 for edit scripts and `http.server`, headless Edge with SwiftShader for the page checks.

## Global Constraints

- **Build and scripts:**
  - No build step.
  - External scripts only three.js r128 from cdnjs, pinned; the Google Fonts stylesheet is the only external stylesheet.
  - `core/` stays DOM-free; `ui/` holds shared DOM helpers.
- **Unchanged behaviour:**
  - The model, `vitTrace` and every existing test stay as they are.
  - `vit/` keeps its scenario values and screenshots through the move to `ui/vitwords.js` (Task 2).
  - `cnn-3d/` and the 3D view of `cnn-train/` keep their scene digests through the `ui/blocks3d.js` additions (Task 3).
- **Colour grammar:**
  - Yellow is the patch sweep's window.
  - Green is what a value is computed from. Its opacity is the strength `vitTrace` gives, and a box at strength 1 gets an outline.
  - Learned weights sit in accent frames.
  - Cubes are blue to orange for sign, with size for magnitude.
- **Copy:**
  - No tables on any page (project rule).
  - NZ English.
  - Wherever a prediction appears, the page says which weights it uses and what their accuracy was measured on.
- **Files and commits:**
  - Files are LF; ignore Git's CRLF warnings.
  - **Never write files with shell heredocs**: this environment drops backslashes, and the page and the edit scripts contain `\n`. Use the Write tool, then run the scripts.
  - Check scripts, edit scripts and their outputs go in `.superpowers/check/` (git-ignored).
  - Work on the branch `feat/vit-3d` **in the main checkout, not a separate worktree**. `.superpowers/check/` holds the harness, `three.min.js` and the `vit/` scenarios, and it lives only there.
  - Conventional Commits, local, each ending with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Do not push. The controller merges into `main` and pushes after the final review.
- **Headless checks run one at a time:** they share port 8123. If one is interrupted, kill only the `python -m http.server 8123` it started and that run's Edge processes; never `taskkill /IM msedge.exe`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `core/vit3d.js`, `core/vit3d.test.mjs` | `VIT_BLOCKS`, `blockOf`, `cubeToCell`, `cellToCube`, `sliceValues`, `traceBoxes`, `attentionFan`, and their tests | 1 |
| `ui/vitwords.js` | `pct`, `NAME`, `describe`, `weightWords`, `loadDigitsAndWeights` | 2 |
| `vit/index.html` | imports `ui/vitwords.js` (unchanged behaviour); later a link to `vit-3d/` | 2, 5 |
| `ui/blocks3d.js` | `strengthBox`, `strengthLine`, `fit`, `snap`, `onClick` | 3 |
| `vit-3d/index.html` | the page | 4 |
| `index.html`, `CLAUDE.md`, `HANDOFF.md`, `project_notes.md` | card, badges, records | 5 |
| `.superpowers/check/*` | the harness (`--three`, `--seeded`), `unchanged.mjs`, `vit3d_checks.mjs`, scenarios, edit scripts (not committed) | 2–5 |

**Context for the executor.** Every file in this plan was prototyped and run on 2026-11-16, in a scratch worktree of the repo at `ffad883`, before the plan was written. The code and the expected outputs below come from that run. The edit scripts were checked to reproduce the prototype exactly from the repo's files.

---

### Task 1: The ViT as blocks (`core/vit3d.js`)

**Files:**
- Create: `core/vit3d.test.mjs`, `core/vit3d.js`

**Interfaces:**
- Consumes `SHAPES`, `T`, `D`, `HID`, `NC` from `core/vit.js`.
- Produces:
  - `VIT_BLOCKS`: 36 entries `{ key, kind, rows, cols, chans, weight?, group }`, in flow order. Kinds: image, kernels, depth, tokens, norm, matrix, row, attention, classes. Groups: embedding, attention, MLP, head.
  - `blockOf(key)`.
  - `cubeToCell(key, r, c, ch)` → `{ t, r, c }`, a probe as `vit/` uses them.
  - `cellToCube(t, row, col)` → `{ key, r, c, ch }`.
  - `sliceValues(key, ch, value)` → `Float32Array`, row-major.
  - `traceBoxes(trace)` → `[{ key, r0, r1, c0, c1, ch0, ch1, s }]`.
  - `attentionFan(trace, probe)` → `[{ i, j, h, s }]`.
- **The mapping:**
  - Token t sits at row t ÷ 4, column t mod 4, and its column is the slice.
  - Attention weight (i, j) of head h sits at row (i ÷ 4)·4 + (j ÷ 4), column (i mod 4)·4 + (j mod 4), slice h.
  - Patch kernel d is slice d, its 16 weights a 4×4 face.
  - Matrices and biases are one slice.
  - LayerNorm parameters are a 2×1 strip, 8 deep.
  - The classes are a column of 10.

- [ ] **Step 1: Write the test**

Create `core/vit3d.test.mjs`:

```js
import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { SHAPES, T, init, forward, tensor, vitTrace } from './vit.js';
import { VIT_BLOCKS, blockOf, cubeToCell, cellToCube, sliceValues, traceBoxes, attentionFan } from './vit3d.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

// every drawn tensor has a block, with as many cubes as the tensor has cells
const drawn = new Set(VIT_BLOCKS.map((b) => b.key));
for (const key of Object.keys(SHAPES)) {
  const bk = key === 'att0' || key === 'att1' ? 'att' : key;
  assert(drawn.has(bk), `${key} has a block`);
}
for (const b of VIT_BLOCKS) {
  const cells = b.key === 'att' ? 2 * T * T : SHAPES[b.key][0] * SHAPES[b.key][1];
  assert(b.rows * b.cols * b.chans === cells, `${b.key}: ${b.rows}×${b.cols}×${b.chans} cubes for ${cells} cells`);
}

// cube → cell → cube round-trips for every cube of every block, and every cell is drawn exactly once
let n = 0; const seen = new Set();
for (const b of VIT_BLOCKS) for (let ch = 0; ch < b.chans; ch++) for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
  const p = cubeToCell(b.key, r, c, ch), q = cellToCube(p.t, p.r, p.c);
  assert(q.key === b.key && q.r === r && q.c === c && q.ch === ch, `${b.key} (${r}, ${c}, ${ch}) → ${JSON.stringify(p)} → ${JSON.stringify(q)}`);
  const id = `${p.t}:${p.r}:${p.c}`; assert(!seen.has(id), `${id} drawn twice`); seen.add(id); n++;
}
console.log(`vit3d: ${VIT_BLOCKS.length} blocks, ${n} cubes, every cube ↔ cell round-trips`);

// the layout: token t at its patch's place; attention weight (i, j) of head h in token i's group at token j's place
assert(JSON.stringify(cellToCube('z0', 6, 3)) === JSON.stringify({ key: 'z0', r: 1, c: 2, ch: 3 }), 'token 6 sits at row 1, column 2');
assert(JSON.stringify(cellToCube('att1', 6, 9)) === JSON.stringify({ key: 'att', r: 6, c: 9, ch: 1 }), 'att1 (6, 9): group of token 6 (rows 4–7, cols 8–11), place of token 9');
assert(JSON.stringify(cellToCube('att0', 5, 0)) === JSON.stringify({ key: 'att', r: 4, c: 4, ch: 0 }), 'att0 (5, 0)');
assert(JSON.stringify(cellToCube('Wp', 3, 6)) === JSON.stringify({ key: 'Wp', r: 1, c: 2, ch: 3 }), 'kernel 3, pixel 6');

// sliceValues reads the tensor through the mapping
const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
const w = init(3), S = forward(w, image(data.test, 0)), value = (t, r, c) => tensor(t, S, w)[r][c];
const z = sliceValues('z0', 5, value);
assert(z.length === 16 && z[1 * 4 + 2] === Math.fround(S.z0[6][5]), 'z0 slice 5 at (1, 2) is token 6, number 5');
const a = sliceValues('att', 1, value);
assert(a[6 * 16 + 9] === Math.fround(S.att[1][6][9]), 'attention slice 1 at (6, 9) is head 2, weight (6, 9)');

// traceBoxes: a token before attention is one box through its depth; a lit matrix column is one box
const tq = vitTrace(S, { t: 'q', r: 5, c: 2 }), bq = traceBoxes(tq);
const box = (key) => bq.filter((x) => x.key === key);
assert(box('xp').length === 1 && JSON.stringify(box('xp')[0]) === JSON.stringify({ key: 'xp', r0: 1, r1: 1, c0: 1, c1: 1, ch0: 0, ch1: 7, s: 1 }), 'token 5 of xp: one box through its 8 slices');
assert(box('digit').length === 1 && box('digit')[0].r0 === 4 && box('digit')[0].r1 === 7 && box('digit')[0].c0 === 4 && box('digit')[0].c1 === 7, 'patch 5: one 4×4 box on the digit');
assert(box('Wq').length === 1 && box('Wq')[0].c0 === 2 && box('Wq')[0].c1 === 2 && box('Wq')[0].r0 === 0 && box('Wq')[0].r1 === 7, 'column 2 of Wq: one box');
// after attention: graded boxes, one per token and block, with the own token at 1
const tz = vitTrace(S, { t: 'z1', r: 6, c: 0 }), bz = traceBoxes(tz);
const z0boxes = bz.filter((x) => x.key === 'z0');
assert(z0boxes.length >= 2 && z0boxes.every((x) => x.ch0 === 0 && x.ch1 === 7), 'z0 after attention: whole-depth boxes');
assert(z0boxes.some((x) => x.r0 === 1 && x.c0 === 2 && x.s === 1), 'token 6 at full strength');
// covering: every lit cell lies in exactly one box, at its strength
for (const [t, arr] of Object.entries(tz)) {
  const C = SHAPES[t][1];
  arr.forEach((s, q) => {
    const cube = cellToCube(t, Math.floor(q / C), q % C);
    const hits = bz.filter((x) => x.key === cube.key && cube.r >= x.r0 && cube.r <= x.r1 && cube.c >= x.c0 && cube.c <= x.c1 && cube.ch >= x.ch0 && cube.ch <= x.ch1);
    if (s > 0) assert(hits.length === 1 && Math.abs(hits[0].s - s) < 1e-6, `${t}[${q}] covered once at ${s}`);
    else assert(hits.length === 0 || cube.key === 'att', `${t}[${q}] unlit and uncovered`); // att0 and att1 share a block, so each other's cells may be lit
  });
}
console.log(`traceBoxes: before attention ${bq.length} boxes, after attention ${bz.length}; every lit cell covered once`);

// the fan: after attention, links from token 6's group to the tokens it read, graded; nothing before attention
const fan = attentionFan(tz, { t: 'z1', r: 6, c: 0 });
assert(fan.length > 0 && fan.every((f) => f.i === 6 && f.j !== 6 && f.s >= 0.05 && f.s <= 0.9 + 1e-6), 'fan of token 6');
assert(Math.max(...fan.map((f) => f.s)) > 0.9 - 1e-6 && new Set(fan.map((f) => f.h)).size === 2, 'the most attended at 0.9, both heads');
const fanO = attentionFan(vitTrace(S, { t: 'o', r: 6, c: 5 }), { t: 'o', r: 6, c: 5 });
assert(fanO.length > 0 && fanO.every((f) => f.h === 1), "a head's own output fans from that head's slice");
assert(attentionFan(tq, { t: 'q', r: 5, c: 2 }).length === 0, 'no fan before attention');
console.log(`attentionFan: ${fan.length} links for z₁ token 6 (both heads), ${fanO.length} for head 2's output`);
console.log('core/vit3d.test.mjs: all checks passed');
```

- [ ] **Step 2: Run it and see it fail**

Run: `node core/vit3d.test.mjs`
Expected: `ERR_MODULE_NOT_FOUND` for `core/vit3d.js`.

- [ ] **Step 3: Write the module**

Create `core/vit3d.js`:

```js
// core/vit3d.js — the toy ViT as blocks of cubes, for vit-3d/: which tensors are drawn, each as rows × columns × slices
// (slices run into depth), and the two-way mapping between a block's cube and a cell of the tensor it draws.
// Token t sits at its patch's place (row t ÷ 4, column t mod 4) and its numbers run into depth. Attention weight (i, j)
// of head h sits in token i's 4 × 4 group, at token j's place inside it, on slice h. Pure, no DOM.
import { SHAPES, T, D, HID, NC } from './vit.js';

const G = 4; // the patch grid is 4 × 4
/** Every drawn block, in flow order. kind decides the mapping (see cubeToCell); group, the camera preset it belongs to. */
export const VIT_BLOCKS = [
  { key: 'digit', kind: 'image', rows: 16, cols: 16, chans: 1, group: 'embedding' },
  { key: 'Wp', kind: 'kernels', rows: G, cols: G, chans: D, weight: true, group: 'embedding' },
  { key: 'bp', kind: 'depth', rows: 1, cols: 1, chans: D, weight: true, group: 'embedding' },
  { key: 'xp', kind: 'tokens', rows: G, cols: G, chans: D, group: 'embedding' },
  { key: 'pos', kind: 'tokens', rows: G, cols: G, chans: D, weight: true, group: 'embedding' },
  { key: 'z0', kind: 'tokens', rows: G, cols: G, chans: D, group: 'embedding' },
  { key: 'ln1', kind: 'norm', rows: 2, cols: 1, chans: D, weight: true, group: 'attention' },
  { key: 'h1', kind: 'tokens', rows: G, cols: G, chans: D, group: 'attention' },
  { key: 'Wq', kind: 'matrix', rows: D, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'bq', kind: 'row', rows: 1, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'Wk', kind: 'matrix', rows: D, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'bk', kind: 'row', rows: 1, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'Wv', kind: 'matrix', rows: D, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'bv', kind: 'row', rows: 1, cols: D, chans: 1, weight: true, group: 'attention' },
  { key: 'q', kind: 'tokens', rows: G, cols: G, chans: D, group: 'attention' },
  { key: 'k', kind: 'tokens', rows: G, cols: G, chans: D, group: 'attention' },
  { key: 'v', kind: 'tokens', rows: G, cols: G, chans: D, group: 'attention' },
  { key: 'att', kind: 'attention', rows: T, cols: T, chans: 2, group: 'attention' },
  { key: 'o', kind: 'tokens', rows: G, cols: G, chans: D, group: 'attention' },
  { key: 'Wo', kind: 'matrix', rows: D, cols: D, chans: 1, weight: true, group: 'MLP' },
  { key: 'bo', kind: 'row', rows: 1, cols: D, chans: 1, weight: true, group: 'MLP' },
  { key: 'z1', kind: 'tokens', rows: G, cols: G, chans: D, group: 'MLP' },
  { key: 'ln2', kind: 'norm', rows: 2, cols: 1, chans: D, weight: true, group: 'MLP' },
  { key: 'h2', kind: 'tokens', rows: G, cols: G, chans: D, group: 'MLP' },
  { key: 'W1', kind: 'matrix', rows: D, cols: HID, chans: 1, weight: true, group: 'MLP' },
  { key: 'b1', kind: 'row', rows: 1, cols: HID, chans: 1, weight: true, group: 'MLP' },
  { key: 'hid', kind: 'tokens', rows: G, cols: G, chans: HID, group: 'MLP' },
  { key: 'W2', kind: 'matrix', rows: HID, cols: D, chans: 1, weight: true, group: 'MLP' },
  { key: 'b2', kind: 'row', rows: 1, cols: D, chans: 1, weight: true, group: 'MLP' },
  { key: 'z2', kind: 'tokens', rows: G, cols: G, chans: D, group: 'head' },
  { key: 'avg', kind: 'depth', rows: 1, cols: 1, chans: D, group: 'head' },
  { key: 'lnf', kind: 'norm', rows: 2, cols: 1, chans: D, weight: true, group: 'head' },
  { key: 'avgNorm', kind: 'depth', rows: 1, cols: 1, chans: D, group: 'head' },
  { key: 'Wh', kind: 'matrix', rows: D, cols: NC, chans: 1, weight: true, group: 'head' },
  { key: 'bh', kind: 'row', rows: 1, cols: NC, chans: 1, weight: true, group: 'head' },
  { key: 'prob', kind: 'classes', rows: NC, cols: 1, chans: 1, group: 'head' },
];
const BY_KEY = Object.fromEntries(VIT_BLOCKS.map((b) => [b.key, b]));
export const blockOf = (key) => BY_KEY[key];

/** The tensor cell { t, r, c } (t a SHAPES key) that cube (r, c, ch) of block `key` draws. */
export function cubeToCell(key, r, c, ch) {
  switch (BY_KEY[key].kind) {
    case 'tokens': return { t: key, r: r * G + c, c: ch };
    case 'kernels': return { t: key, r: ch, c: r * G + c };
    case 'attention': return { t: 'att' + ch, r: (r >> 2) * G + (c >> 2), c: (r & 3) * G + (c & 3) };
    case 'depth': return { t: key, r: 0, c: ch };
    case 'norm': return { t: key, r, c: ch };
    case 'classes': return { t: key, r: 0, c: r };
    default: return { t: key, r, c }; // image, matrix, row
  }
}
/** The cube { key, r, c, ch } that draws cell (row, col) of tensor t; the inverse of cubeToCell. */
export function cellToCube(t, row, col) {
  if (t === 'att0' || t === 'att1') return { key: 'att', r: (row >> 2) * G + (col >> 2), c: (row & 3) * G + (col & 3), ch: +t[3] };
  switch (BY_KEY[t].kind) {
    case 'tokens': return { key: t, r: row >> 2, c: row & 3, ch: col };
    case 'kernels': return { key: t, r: col >> 2, c: col & 3, ch: row };
    case 'depth': return { key: t, r: 0, c: 0, ch: col };
    case 'norm': return { key: t, r: row, c: 0, ch: col };
    case 'classes': return { key: t, r: col, c: 0, ch: 0 };
    default: return { key: t, r: row, c: col, ch: 0 };
  }
}
/** The values of one slice of a block, row-major, as the renderer draws them: value(t, row, col) reads a tensor cell. */
export function sliceValues(key, ch, value) {
  const b = BY_KEY[key], out = new Float32Array(b.rows * b.cols);
  for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) { const p = cubeToCell(key, r, c, ch); out[r * b.cols + c] = value(p.t, p.r, p.c); }
  return out;
}

/**
 * The green of a trace ({ tensor key: Float32Array of strengths }, as vitTrace returns it) as boxes over blocks:
 * [{ key, r0, r1, c0, c1, ch0, ch1, s }]. Cells of one strength merge: down a column, then across columns with the same
 * rows, then across consecutive slices with the same rectangle, so a lit token is one box through its depth and a lit
 * row of tokens or of a matrix is one box.
 */
export function traceBoxes(trace) {
  const cells = {}; // block key → Map('r,c,ch' → s)
  for (const [t, a] of Object.entries(trace)) {
    const C = SHAPES[t][1];
    for (let q = 0; q < a.length; q++) if (a[q] > 0) {
      const cube = cellToCube(t, Math.floor(q / C), q % C), m = cells[cube.key] || (cells[cube.key] = new Map()), id = `${cube.r},${cube.c},${cube.ch}`;
      m.set(id, Math.max(m.get(id) || 0, a[q]));
    }
  }
  const boxes = [];
  for (const [key, m] of Object.entries(cells)) {
    const b = BY_KEY[key], rects = []; // per slice: rectangles of one strength
    for (let ch = 0; ch < b.chans; ch++) {
      const s = (r, c) => m.get(`${r},${c},${ch}`) || 0, runs = [];
      for (let c = 0; c < b.cols; c++) for (let r = 0; r < b.rows;) {
        const v = s(r, c); if (!v) { r++; continue; }
        let r1 = r; while (r1 + 1 < b.rows && s(r1 + 1, c) === v) r1++;
        runs.push({ r0: r, r1, c0: c, c1: c, s: v }); r = r1 + 1;
      }
      for (const run of runs) { // join a run to the rectangle ending in the column just before it, with the same rows
        const prev = rects.find((x) => x.ch === ch && x.c1 === run.c0 - 1 && x.r0 === run.r0 && x.r1 === run.r1 && x.s === run.s);
        if (prev) prev.c1 = run.c1; else rects.push({ ...run, ch });
      }
    }
    for (const x of rects) { // join a rectangle to the same one on the slice just before it
      const prev = boxes.find((y) => y.key === key && y.ch1 === x.ch - 1 && y.r0 === x.r0 && y.r1 === x.r1 && y.c0 === x.c0 && y.c1 === x.c1 && y.s === x.s);
      if (prev) prev.ch1 = x.ch; else boxes.push({ key, r0: x.r0, r1: x.r1, c0: x.c0, c1: x.c1, ch0: x.ch, ch1: x.ch, s: x.s });
    }
  }
  return boxes;
}

const AFTER = ['o', 'z1', 'h2', 'hid', 'z2'];
/**
 * The fan of an after-attention probe { t, r, c }: the tokens j that token i = r read, each with the strength vitTrace
 * gives it (0.9 for the most attended, scaled by attention weight), as [{ i, j, h, s }] with h the head whose slice of
 * the attention block the link starts from (the probed head for a head's own output, else both heads). Token i itself
 * and strengths below 0.05 are left out. Empty for any other probe.
 */
export function attentionFan(trace, probe) {
  if (!probe || !AFTER.includes(probe.t) || !trace.z0) return [];
  const i = probe.r, heads = probe.t === 'o' ? [Math.floor(probe.c / (D / 2))] : [0, 1], fan = [];
  for (let j = 0; j < T; j++) {
    const s = trace.z0[j * D]; // a token's z₀ row is lit at its strength
    if (j !== i && s >= 0.05) for (const h of heads) fan.push({ i, j, h, s });
  }
  return fan;
}
```

- [ ] **Step 4: Run the tests**

Run: `node core/vit3d.test.mjs`
Expected, exactly:
```
vit3d: 36 blocks, 3828 cubes, every cube ↔ cell round-trips
traceBoxes: before attention 8 boxes, after attention 118; every lit cell covered once
attentionFan: 30 links for z₁ token 6 (both heads), 6 for head 2's output
core/vit3d.test.mjs: all checks passed
```
Then: `node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs && node core/vit.test.mjs && node core/race.test.mjs`. Expected: all pass, as before.

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/vit-3d
git add core/vit3d.js core/vit3d.test.mjs
git commit -m "feat(core): lay the ViT out as blocks for a 3D view" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `ui/vitwords.js`, and `vit/` moved onto it unchanged

**Files:**
- Create: `ui/vitwords.js`
- Modify: `vit/index.html` (edit script)
- Create: `.superpowers/check/unchanged.mjs`, `.superpowers/check/edit_vitwords.py`

**Interfaces:**
- Produces, from `ui/vitwords.js`:
  - `pct(x)` and `NAME`;
  - `describe(p, S, w)`: the readout for a probe `{ t, r, c }`, moved verbatim from `vit/`;
  - `weightWords({ S, trained, meta, position, order, data })` → `{ src, note }`, where `src` is HTML for beside the prediction and `note` is a sentence;
  - `loadDigitsAndWeights(anchor, digitsUrl, weightsUrl)` → `null` when the digits cannot load, else `{ data, byClass, wTrained, meta }`. It shows the hand-off banner, or the "Could not load the trained weights" banner, after `anchor`.
- `vit/index.html` keeps all its behaviour.

- [ ] **Step 1: The check files that already exist**

Run: `ls .superpowers/check/vitcheck.mjs .superpowers/check/scen_basic.js .superpowers/check/scen_hover.js .superpowers/check/scen_stored.js .superpowers/check/three.min.js`
Expected: all five listed. They come from the earlier ViT plans; `three.min.js` is three.js r128 (603,445 bytes). If it is missing, download `https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js` into `.superpowers/check/` with Python's `urllib`, since `curl` is blocked here.

- [ ] **Step 2: A baseline of `vit/` before the move**

Create `.superpowers/check/unchanged.mjs`:

```js
// unchanged.mjs — prove that a refactor left pages as they were.
//   node .superpowers/check/unchanged.mjs save <set> [--root <dir>]      before the change
//   node .superpowers/check/unchanged.mjs compare <set> [--root <dir>]   after it
// Sets:
//   vit    vit/: scenario values with the shipped weights, with none and with a hand-off; screenshots at rest, hovered
//          and dark, three captures each. Edge sometimes renders text edges differently from one capture to the next, so
//          a screenshot passes when the before and after sets of hashes share one: the same pixels were drawn by both.
//   cnn3d  cnn-3d/ and the 3D view of cnn-train/: scene digests (every cube's matrix and colour, the highlight overlays,
//          the camera goal) with a seeded Math.random and local three.js.
// Baselines go to .superpowers/check/base/<set>.json. One run at a time: the harness serves on port 8123.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const argv = process.argv.slice(2), [mode, set] = argv, ri = argv.indexOf('--root');
const chk = dirname(fileURLToPath(import.meta.url)), root = ri < 0 ? join(chk, '..', '..') : resolve(argv[ri + 1]);
const SETS = {
  vit: {
    runs: [['basic', 'scen_basic.js', ['--page', 'vit']], ['no-weights', 'scen_basic.js', ['--page', 'vit', '--no-weights']], ['stored', 'scen_stored.js', ['--page', 'vit']]],
    shots: [['rest', 'scen_basic.js', ['--page', 'vit']], ['hover', 'scen_hover.js', ['--page', 'vit']], ['dark', 'scen_basic.js', ['--page', 'vit', '--dark']]],
  },
  cnn3d: {
    runs: [['cnn-3d', 'scen_cnn3d.js', ['--page', 'cnn-3d', '--three', '--seeded']], ['cnn-train 3D', 'scen_cnntrain3d.js', ['--page', 'cnn-train', '--three', '--seeded']]],
    shots: [],
  },
};
if (!['save', 'compare'].includes(mode) || !SETS[set]) { console.log('usage: node unchanged.mjs save|compare vit|cnn3d [--root <dir>]'); process.exit(2); }
const harness = (scen, flags, shot) => {
  const out = execFileSync('node', [join(chk, 'vitcheck.mjs'), root, join(chk, scen), ...flags, ...(shot ? ['--shot', shot] : [])], { encoding: 'utf8', timeout: 240000 });
  return JSON.parse(out.slice(0, out.lastIndexOf('}') + 1));
};
const tmp = mkdtempSync(join(tmpdir(), 'unchanged-')), now = { runs: {}, shots: {} };
for (const [name, scen, flags] of SETS[set].runs) now.runs[name] = harness(scen, flags);
for (const [name, scen, flags] of SETS[set].shots) {
  now.shots[name] = [];
  for (let k = 0; k < 3; k++) { const png = join(tmp, `${name}${k}.png`); harness(scen, flags, png); now.shots[name].push(createHash('sha256').update(readFileSync(png)).digest('hex').slice(0, 12)); }
}
rmSync(tmp, { recursive: true, force: true });
const file = join(chk, 'base', `${set}.json`);
if (mode === 'save') {
  mkdirSync(join(chk, 'base'), { recursive: true }); writeFileSync(file, JSON.stringify(now, null, 1));
  for (const [name, r] of Object.entries(now.runs)) console.log(`saved ${name}${r.error ? ` (scenario error: ${r.error})` : ''}${r.consoleErrors.length ? ` (console errors: ${r.consoleErrors.length})` : ''}`);
  for (const [name, h] of Object.entries(now.shots)) console.log(`saved ${name} screenshots: ${[...new Set(h)].join(' ')}`);
} else {
  const was = JSON.parse(readFileSync(file, 'utf8')); let bad = 0;
  for (const [name, r] of Object.entries(now.runs)) {
    const same = JSON.stringify(r) === JSON.stringify(was.runs[name]); if (!same) bad++;
    console.log(`${same ? 'IDENTICAL' : 'DIFFERS  '} ${name}`);
    if (!same) for (const k of new Set([...Object.keys(r), ...Object.keys(was.runs[name])])) if (JSON.stringify(r[k]) !== JSON.stringify(was.runs[name][k])) console.log(`  ${k}: was ${JSON.stringify(was.runs[name][k]).slice(0, 200)}\n  ${k}: now ${JSON.stringify(r[k]).slice(0, 200)}`);
  }
  for (const [name, h] of Object.entries(now.shots)) {
    const shared = h.some((x) => was.shots[name].includes(x)); if (!shared) bad++;
    console.log(`${shared ? 'IDENTICAL' : 'DIFFERS  '} ${name} screenshot (before ${[...new Set(was.shots[name])].join(' ')}, after ${[...new Set(h)].join(' ')})`);
  }
  console.log(bad ? `${set}: ${bad} difference(s)` : `${set}: unchanged`); process.exit(bad ? 1 : 0);
}
```

Run: `node .superpowers/check/unchanged.mjs save vit` (about two minutes).
Expected: `saved basic`, `saved no-weights` and `saved stored`, with no errors noted, then one `saved … screenshots:` line each for rest, hover and dark. On 2026-11-16 the hashes were `17218c44b1cc`, `06cb08981c43` and `a78c5be3d4d6`.

- [ ] **Step 3: Write the shared module**

Create `ui/vitwords.js`:

```js
// ui/vitwords.js — what the ViT explorer pages say and load: the readout for a probed value, the sentences about where
// the weights came from and what their accuracy was measured on, and loading the digits and the weights (a hand-off from
// vit-train/, the shipped file, or random with a banner). Shared by vit/ and vit-3d/.
import { DH, tensor, isWeights } from '../core/vit.js';
import { parseMnist16 } from '../core/data.js';
import { fmt } from './grid.js';
import { loadTrainedViT, showTrainedViTBanner } from './trained.js';

export const pct = (x) => (100 * x).toFixed(1) + '%';
export const NAME = { xp: 'patch token', z0: 'z₀', h1: 'LayerNorm 1 output', q: 'query', k: 'key', v: 'value', o: 'heads joined', z1: 'z₁', h2: 'LayerNorm 2 output', hid: 'MLP hidden', z2: 'z₂',
  Wp: 'patch kernels', bp: 'patch bias', pos: 'position table', ln1: 'LayerNorm 1 (row 0 scale, row 1 shift)', ln2: 'LayerNorm 2 (row 0 scale, row 1 shift)', lnf: 'final LayerNorm (row 0 scale, row 1 shift)',
  Wq: 'Wq', bq: 'bq', Wk: 'Wk', bk: 'bk', Wv: 'Wv', bv: 'bv', Wo: 'Wo', bo: 'bo', W1: 'W₁', b1: 'b₁', W2: 'W₂', b2: 'b₂', Wh: 'W head', bh: 'head bias' };
const meanAtt = (S) => S.att[0].map((r, a) => r.map((x, b) => (x + S.att[1][a][b]) / 2));

/** The readout for a probe { t, r, c } of the forward pass S under the weights w. */
export function describe(p, S, w) {
  if (!p) return 'Hover any cell: green shows what it is computed from. After attention, its strength is the attention weight.';
  const val = tensor(p.t, S, w)[p.r][p.c], at = `<b>${NAME[p.t] || p.t}</b>`;
  if (p.t === 'digit') { const s = (p.r >> 2) * 4 + (p.c >> 2); return `<b>pixel (${p.r}, ${p.c})</b> = ${fmt(val)}, in patch ${s}, which becomes token ${s}.`; }
  if (p.t === 'att0' || p.t === 'att1') {
    const h = +p.t[3];
    return `<b>head ${h + 1}</b>: token ${p.r} gives token ${p.c} the weight <b>${fmt(val)}</b> (score ${fmt(S.scores[h][p.r][p.c])}); each row sums to 1. Green: query ${p.r} and key ${p.c}, and every other key faintly, since the softmax divides by all of them.`;
  }
  if (['xp', 'z0', 'h1', 'q', 'k', 'v'].includes(p.t)) return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. Before attention a token has seen only its own patch.`;
  if (['o', 'z1', 'h2', 'hid', 'z2'].includes(p.t)) {
    const one = p.t === 'o', A = one ? S.att[Math.floor(p.c / DH)] : meanAtt(S);
    const top = A[p.r].map((a, j) => [a, j]).filter(([, j]) => j !== p.r).sort((x, y) => y[0] - x[0]).slice(0, 3).map(([a, j]) => `token ${j} (${fmt(a)})`).join(', ');
    return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. After attention: its own patch full green, every other patch by the attention weight it got${one ? ` in head ${Math.floor(p.c / DH) + 1}` : ', both heads averaged'}, scaled to the largest (attention, not importance). Most attended: ${top}.`;
  }
  if (p.t === 'avg') return `<b>average, column ${p.c}</b> = ${fmt(val)}: column ${p.c} of z₂ averaged over the 16 tokens, as the CNN averages each map.`;
  if (p.t === 'avgNorm') return `<b>normalised average, column ${p.c}</b> = ${fmt(val)}.`;
  if (p.t === 'prob') return `<b>class ${p.c}</b>: ${pct(val)}. It reads only the 8 normalised averages, through column ${p.c} of W head and its bias.`;
  return `${at} [${p.r}, ${p.c}] = ${fmt(val)}, learned. Green: what it reads and what it makes.`;
}

/** The page's two statements about the weights: `src` (HTML, beside the prediction) and `note` (a sentence). The
 *  accuracy was measured with position on and the patches in reading order; they say so when the page is set otherwise. */
export function weightWords({ S, trained, meta, position, order, data }) {
  const top = S.prob.indexOf(Math.max(...S.prob));
  const opt = meta && (meta.optimiser === 'adam' ? 'Adam' : 'SGD');
  const noPos = meta && Number.isFinite(meta.testAccNoPos) ? pct(meta.testAccNoPos) : null;
  let short = `trained weights: ${trained ? pct(meta.testAcc) : ''} right on held-out digits`, full = '';
  if (trained && !position) { short = `trained weights, trained with position${noPos ? `: ${noPos} right on held-out digits without it` : ''}`; full = ` The weights were trained with position, so with position off${noPos ? ` they get only ${noPos} right` : ' that figure does not apply'}.`; }
  if (trained && order) { short += ' (accuracy measured on patches in reading order, not shuffled)'; full += ' The accuracy was measured on patches in reading order, not on shuffled ones.'; }
  const n = (set) => set.n.toLocaleString('en-NZ');
  return {
    src: `prediction <b>${top}</b> (${pct(S.prob[top])}), ` + (trained ? short : 'random weights: arbitrary'),
    note: trained
      ? `These weights were trained ${meta.source === 'vit-train' ? `on the ViT training page (${meta.digits.toLocaleString('en-NZ')} digits seen, ${opt}, learning rate ${meta.lr})` : `offline by tools/train_vit.mjs on the ${n(data.train)} training digits (${meta.passes} passes, ${opt})`}; with position on and the patches in reading order they get ${pct(meta.testAcc)} of the ${n(data.test)} held-out digits right; the CNN pages' toy, trained on the same 3,000 digits for 30,000 steps, reaches 71 to 85% depending on the seed (5 runs, shown on the ViT training page).${full} On real photographs a ViT needs far more training data than a CNN to catch up: it has to learn from examples the locality a convolution has built in.`
      : 'These are random starting weights, before any training. Attention is learned: before training its maps are noise and the prediction is arbitrary.',
  };
}

function banner(anchor, text) { const d = document.createElement('div'); d.className = 'banner'; d.textContent = text; anchor.after(d); }
/**
 * Load the digits and the weights, with a banner after `anchor` when something is missing or the weights are a hand-off.
 * The page passes both URLs, so its checks can point them elsewhere. Returns null when the digits cannot be loaded,
 * else { data, byClass (held-out digit indices per class), wTrained (null when none), meta }.
 */
export async function loadDigitsAndWeights(anchor, digitsUrl, weightsUrl) {
  const [bin, js] = await Promise.all([
    fetch(digitsUrl).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null),
    fetch(weightsUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  if (!bin) { banner(anchor, 'Could not load the digits (data/mnist16.bin); this page needs to be served over http.'); return null; }
  const data = parseMnist16(bin);
  const byClass = Array.from({ length: 10 }, (_, c) => [...Array(data.test.n).keys()].filter((i) => data.test.labels[i] === c));
  const stored = loadTrainedViT(isWeights);
  let wTrained = null, meta = null;
  if (stored) { wTrained = stored.weights; meta = stored.meta; showTrainedViTBanner(anchor, stored); }
  else if (js && js.version === 1 && isWeights(js.weights)) { wTrained = js.weights; meta = js.meta; }
  else banner(anchor, 'Could not load the trained weights (models/vit16.json), so this page shows random starting weights.');
  return { data, byClass, wTrained, meta };
}
```

- [ ] **Step 4: Move `vit/` onto it**

Create `.superpowers/check/edit_vitwords.py`:

```python
# vit/index.html: the readout, the weights' wording and the loading move to ui/vitwords.js. Each original occurs once.
import sys, re
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
def cut(start, end):
    """Replace from `start` up to (not including) `end` with nothing; both occur once."""
    global s
    assert s.count(start) == 1 and s.count(end) == 1, start[:60]
    a = s.index(start); b = s.index(end, a)
    s = s[:a] + s[b:]

rep("import { SHAPES, P, T, D, DH, NC, init, forward, tensor, vitTrace, isWeights, paramCount } from '../core/vit.js';\n",
    "import { SHAPES, P, T, D, DH, NC, init, forward, tensor, vitTrace, paramCount } from '../core/vit.js';\n")
rep("import { parseMnist16, image } from '../core/data.js';\n", "import { image } from '../core/data.js';\n")
rep("import { loadTrainedViT, showTrainedViTBanner } from '../ui/trained.js';\n",
    "import { describe as describeProbe, weightWords, loadDigitsAndWeights, pct } from '../ui/vitwords.js';\n")
rep("const pct = (x) => (100 * x).toFixed(1) + '%';\n", "")
cut("const NAME = { xp: 'patch token',", "function patchMath() {")
rep("function patchMath() {", "const describe = (p) => describeProbe(p, S, W());\nfunction patchMath() {")
cut("  const top = S.prob.indexOf(Math.max(...S.prob)), trained = useTrained && wTrained;\n", "}\n\n// ---------- controls ----------")
rep("function wording() {\n}\n\n// ---------- controls ----------",
    "function wording() {\n  const words = weightWords({ S, trained: useTrained && !!wTrained, meta, position, order, data });\n  $('wsrc').innerHTML = words.src; $('wnote').textContent = words.note;\n}\n\n// ---------- controls ----------")
rep("function banner(text) { const d = el('banner'); d.textContent = text; document.querySelector('h1').after(d); }\n", "")
a = s.index("  const [bin, js] = await Promise.all([")
b = s.index("  setDigit(); syncControls(); render(); layout();")
s = s[:a] + ("  const got = await loadDigitsAndWeights(document.querySelector('h1'), '../data/mnist16.bin', '../models/vit16.json');\n"
             "  if (!got) return;\n"
             "  ({ data, byClass, wTrained, meta } = got); useTrained = !!wTrained;\n") + s[b:]
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('vit/index.html: wording and loading from ui/vitwords.js')
```

Run: `python .superpowers/check/edit_vitwords.py vit/index.html`
Expected: `vit/index.html: wording and loading from ui/vitwords.js`.

- [ ] **Step 5: `vit/` is unchanged**

Run: `node .superpowers/check/unchanged.mjs compare vit`
Expected:
```
IDENTICAL basic
IDENTICAL no-weights
IDENTICAL stored
IDENTICAL rest screenshot (before 17218c44b1cc, after 17218c44b1cc)
IDENTICAL hover screenshot (before 06cb08981c43, after 06cb08981c43)
IDENTICAL dark screenshot (before a78c5be3d4d6, after a78c5be3d4d6)
vit: unchanged
```
If a screenshot line says `DIFFERS`, run the compare once more: Edge sometimes renders a text edge differently. If it still differs, compare one before and one after capture with `python .superpowers/check/pngdiff.py a.png b.png` and fix the cause before going on. Do not re-save the baseline.

- [ ] **Step 6: Commit**

```bash
git add ui/vitwords.js vit/index.html
git commit -m "refactor(vit): move the readout, wording and loading to ui/vitwords.js" -m "The ViT 3D explorer needs the same three, so they move to a shared module. vit/ is unchanged: the same scenario values with shipped, missing and handed-off weights, and the same screenshots at rest, hovered and dark." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The renderer's opt-in additions, with the CNN scenes unchanged

**Files:**
- Modify: `.superpowers/check/vitcheck.mjs` (edit script: `--three`, `--seeded`)
- Create: `.superpowers/check/edit_harness.py`, `.superpowers/check/scen_cnn3d.js`, `.superpowers/check/scen_cnntrain3d.js`, `.superpowers/check/edit_blocks3d.py`
- Modify: `ui/blocks3d.js` (edit script)

**Interfaces:**
- Produces, on the object `createBlockView` returns:
  - `strengthBox(b, r0, r1, c0, c1, ch0, ch1, s, group)`: a green box over cells r0..r1 × c0..c1 of slices ch0..ch1, with fill opacity 0.05 + 0.3·s and an outline when s is 1. Materials are shared per twentieth of strength.
  - `strengthLine(p, q, s, group)`: a green line between two `THREE.Vector3`, with opacity 0.15 + 0.85·s.
  - `fit(list, theta, phi, pad = 1.04)`: sets the camera goal that frames the blocks (ids or blocks) seen from theta, phi, with room above for two-line labels.
  - `snap()`: jumps the camera to its goal and draws that frame now, labels included. The checks read the canvas straight after.
  - `onClick(cb)`: calls `cb(hit or null)` for a press and release that travelled at most 4 px. With `onClick` set, the press picks first, so a tap knows its cube.
- The render loop's body becomes `draw()`, with the same order of calls. The flag `drag.moved`, which was set but read nowhere, now means "travelled more than 4 px".
- The harness gains:
  - `--three`: three.js from `.superpowers/check/three.min.js`, SwiftShader WebGL, and at most 40 animation frames;
  - `--seeded`: a seeded `Math.random`;
  - a console filter for SwiftShader's "GL Driver Message" lines.

- [ ] **Step 1: The harness learns `--three` and `--seeded`**

Create `.superpowers/check/edit_harness.py`:

```python
# .superpowers/check/vitcheck.mjs: a --three flag for WebGL pages (local three.js, SwiftShader, capped animation frames).
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep("//   node vitcheck.mjs <root> <scenario.js> [--page vit|vit-train] [--shot out.png] [--width 1400] [--height 1000] [--dark] [--no-weights]\n",
    "//   node vitcheck.mjs <root> <scenario.js> [--page vit|vit-train|vit-3d|cnn-3d|cnn-train] [--shot out.png] [--width 1400] [--height 1000] [--dark] [--no-weights] [--three] [--sub 'a=>b']\n"
    "// --three, for WebGL pages: three.js from .superpowers/check/three.min.js instead of cdnjs, SwiftShader WebGL, and at\n"
    "// most 40 animation frames (an uncapped render loop under software WebGL keeps headless Edge busy for minutes).\n")
rep("const sub = opt('sub', null); if (sub) { const [a, b] = sub.split('=>'); html = html.replace(a, b); }\n",
    "const sub = opt('sub', null); if (sub) { const [a, b] = sub.split('=>'); html = html.replace(a, b); }\n"
    "if (flag('three')) {\n"
    "  html = html.replace('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js', '../.superpowers/check/three.min.js');\n"
    "  html = html.replace('<head>', '<head><script>(() => { let n = 0; const raf = window.requestAnimationFrame.bind(window); window.requestAnimationFrame = (cb) => (n++ < 40 ? raf(cb) : 0); })();</script>');\n"
    "}\n")
rep("'--headless=new', '--disable-gpu', '--no-first-run',", "'--headless=new', '--disable-gpu', ...(flag('three') ? ['--enable-unsafe-swiftshader'] : []), '--no-first-run',")
# --seeded: a seeded Math.random (the CNN pages reroll weights with it), so scene digests repeat
rep("if (flag('three')) {\n",
    "if (flag('seeded')) html = html.replace('<head>', '<head><script>(() => { let s = 12345; Math.random = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();</script>');\n"
    "if (flag('three')) {\n")
# software WebGL's performance notices are not page errors
a = "!/Tracking Prevention/.test(l)"
assert s.count(a) == 1, 'console filter'
s = s.replace(a, "!/Tracking Prevention|GL Driver Message/.test(l)")
s = s.replace("[--no-weights] [--three] [--sub 'a=>b']\n", "[--no-weights] [--three] [--seeded] [--sub 'a=>b']\n")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('vitcheck.mjs: --three')
```

Run: `python .superpowers/check/edit_harness.py .superpowers/check/vitcheck.mjs`
Expected: `vitcheck.mjs: --three`.

- [ ] **Step 2: Scene digests of the two CNN 3D views, before the change**

Create `.superpowers/check/scen_cnn3d.js`:

```js
// The CNN 3D explorer's scene, as numbers: every cube's matrix and colour, the highlight overlays for a probe, the
// camera goal. Equal digests before and after a change to ui/blocks3d.js mean the scene did not change.
await new Promise((r) => setTimeout(r, 300));
const sum = (arr) => { let a = 0, b = 0; for (let i = 0; i < arr.length; i++) { a += arr[i]; b += arr[i] * ((i % 97) + 1); } return [+a.toFixed(4), +b.toFixed(4)]; };
const digest = () => ({
  meshes: view.net.children.map((m) => [m.count, sum(m.instanceMatrix.array), m.instanceColor ? sum(m.instanceColor.array) : null]),
  hl: view.hl.children.map((o) => [o.type, +o.position.x.toFixed(3), +o.position.y.toFixed(3), +o.position.z.toFixed(3)]),
  goal: [+view.goal.theta.toFixed(4), +view.goal.phi.toFixed(4), +view.goal.dist.toFixed(3), ...view.goal.target.toArray().map((v) => +v.toFixed(3))],
});
R.plain = digest();
probe = { stage: 'conv2', r: 3, c: 4, ch: 2 }; updateHighlights(); R.probed = digest();
setLayout('depth'); probe = { stage: 'pool1', r: 2, c: 5, ch: 1 }; updateHighlights(); R.depth = digest();
setSweep('l2'); swStep = 7; updateSweep(); R.sweep = view.aux.children.length;
R.webgl = !!view.renderer.getContext() && !view.renderer.getContext().isContextLost();
```

Create `.superpowers/check/scen_cnntrain3d.js`:

```js
// The CNN training page's 3D view, as numbers (see scen_cnn3d.js): side by side after a full backward step with a probed
// weight, then in depth.
for (let k = 0; k < 100 && !(data && cur); k++) await new Promise((r) => setTimeout(r, 50));
const sum = (arr) => { let a = 0, b = 0; for (let i = 0; i < arr.length; i++) { a += arr[i]; b += arr[i] * ((i % 97) + 1); } return [+a.toFixed(4), +b.toFixed(4)]; };
const digest = () => ({
  meshes: view3d.net.children.map((m) => [m.count, sum(m.instanceMatrix.array), m.instanceColor ? sum(m.instanceColor.array) : null]),
  hl: view3d.hl.children.map((o) => [o.type, +o.position.x.toFixed(3), +o.position.y.toFixed(3), +o.position.z.toFixed(3)]),
  goal: [+view3d.goal.theta.toFixed(4), +view3d.goal.phi.toFixed(4), +view3d.goal.dist.toFixed(3), ...view3d.goal.target.toArray().map((v) => +v.toFixed(3))],
});
setRep('3d');
for (let k = 0; k < 5; k++) $('stepBtn').click();
probe = { kind: 'w', layer: 'W', f: 2, c: 3 }; renderAll(); R.side = digest();
$('layDepth3d').click(); R.depth = digest();
R.webgl = !!view3d.renderer.getContext() && !view3d.renderer.getContext().isContextLost();
```

Run: `node .superpowers/check/unchanged.mjs save cnn3d`
Expected: `saved cnn-3d` and `saved cnn-train 3D`, with no errors noted.

- [ ] **Step 3: The additions**

Create `.superpowers/check/edit_blocks3d.py`:

```python
# ui/blocks3d.js: opt-in additions for vit-3d/ (graded boxes and lines, a camera fit, a camera snap, a click callback).
# The CNN pages call none of them, so their scenes do not change. Each original occurs once.
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)

rep("// a `ghost` block holds its place in the layout without being built. `place(defs)` lays blocks out without building them.\n",
    "// a `ghost` block holds its place in the layout without being built. `place(defs)` lays blocks out without building them.\n"
    "// Used only by vit-3d/: `strengthBox` and `strengthLine` (graded green), `fit` (frame a set of blocks), `snap` (jump the\n"
    "// camera to its goal) and `onClick` (a click without a drag).\n")
rep("  // ---- labels ----\n",
    "  // ---- graded highlights: a box or a line whose opacity is a strength from 0 to 1 ----\n"
    "  const strengthMats = new Map();\n"
    "  function strengthMat(kind, s) { // materials shared per twentieth of strength\n"
    "    const q = Math.round(clamp(s, 0, 1) * 20), id = kind + q; let m = strengthMats.get(id);\n"
    "    if (!m) { m = kind === 'fill' ? new THREE.MeshBasicMaterial({ color: 0x3fae4b, transparent: true, opacity: 0.05 + 0.3 * q / 20, depthWrite: false })\n"
    "      : new THREE.LineBasicMaterial({ color: 0x2e8f3a, transparent: true, opacity: 0.15 + 0.85 * q / 20 }); strengthMats.set(id, m); }\n"
    "    return m; }\n"
    "  /** A green box over cells r0..r1 × c0..c1 of slices ch0..ch1 of block b, filled as strongly as s; outlined when s is 1. */\n"
    "  function strengthBox(b, r0, r1, c0, c1, ch0, ch1, s, group) {\n"
    "    const p0 = voxelPos(b, r0, c0, ch0), p1 = voxelPos(b, r1, c1, ch1);\n"
    "    const g = new THREE.BoxGeometry(Math.abs(p1.x - p0.x) + 1.16, Math.abs(p1.y - p0.y) + 1.16, Math.abs(p1.z - p0.z) + 1.16);\n"
    "    const m = new THREE.Mesh(g, strengthMat('fill', s)); m.position.set((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, (p0.z + p1.z) / 2); group.add(m);\n"
    "    if (s >= 0.999) { const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), mats.hlEdge); e.position.copy(m.position); group.add(e); } }\n"
    "  /** A green line from p to q (THREE.Vector3), as strong as s. */\n"
    "  function strengthLine(p, q, s, group) { group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([p, q]), strengthMat('line', s))); }\n"
    "\n"
    "  // ---- labels ----\n")
rep("  function applyCam() {",
    "  /** Frame blocks (ids or blocks) seen from theta, phi: the distance that brings every corner inside the view with a\n"
    "   *  margin `pad` and room above the top blocks for their two-line labels, then the target that centres them. */\n"
    "  function fit(list, theta, phi, pad = 1.04) {\n"
    "    const bs = list.map((x) => (typeof x === 'string' ? blockById(x) : x)).filter(Boolean); if (!bs.length) return;\n"
    "    const corners = [], box = new THREE.Box3();\n"
    "    for (const b of bs) { const bx = bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1); box.union(bx);\n"
    "      for (const x of [bx.min.x, bx.max.x]) for (const y of [bx.min.y, bx.max.y]) for (const z of [bx.min.z, bx.max.z]) corners.push(new THREE.Vector3(x, y, z)); }\n"
    "    const c2 = new THREE.PerspectiveCamera(camera.fov, camera.aspect, 0.1, 5000), q = new THREE.Vector3(), head = 2 * 44 / Math.max(200, viewEl.clientHeight);\n"
    "    const dir = new THREE.Vector3(Math.sin(phi) * Math.sin(theta), Math.cos(phi), Math.sin(phi) * Math.cos(theta)), target = box.getCenter(new THREE.Vector3());\n"
    "    const extent = (dist) => { c2.position.copy(target).addScaledVector(dir, dist); c2.lookAt(target); c2.updateMatrixWorld(); let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, front = true;\n"
    "      for (const p of corners) { q.copy(p).project(c2); if (q.z >= 1) front = false; x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }\n"
    "      return { x0, x1, y0, y1: y1 + head, front }; };\n"
    "    let dist = 100;\n"
    "    for (let pass = 0; pass < 4; pass++) { // fit the distance, slide the target to centre the projection, again\n"
    "      let lo = 1, hi = 3000; for (let k = 0; k < 40; k++) { const mid = (lo + hi) / 2, e = extent(mid); if (e.front && Math.max(-e.x0, e.x1, -e.y0, e.y1) * pad <= 1) hi = mid; else lo = mid; } dist = hi;\n"
    "      const e = extent(dist), halfH = Math.tan((camera.fov / 2) * Math.PI / 180) * dist, halfW = halfH * c2.aspect;\n"
    "      target.addScaledVector(new THREE.Vector3().setFromMatrixColumn(c2.matrixWorld, 0), (e.x0 + e.x1) / 2 * halfW).addScaledVector(new THREE.Vector3().setFromMatrixColumn(c2.matrixWorld, 1), (e.y0 + e.y1) / 2 * halfH);\n"
    "    }\n"
    "    setGoal({ theta, phi, dist, target });\n"
    "  }\n"
    "  /** Jump the camera to its goal and draw that frame now, labels included (checks read the canvas straight after). */\n"
    "  function snap() { cam.theta = camGoal.theta; cam.phi = camGoal.phi; cam.dist = camGoal.dist; cam.target.copy(camGoal.target); draw(); }\n"
    "  function applyCam() {")
rep("    applyCam(); renderer.setClearColor(new THREE.Color(cssVar('--scene'))); renderer.render(scene, camera); syncLabels(); syncLateral(); requestAnimationFrame(loop); }\n",
    "    draw(); requestAnimationFrame(loop); }\n"
    "  function draw() { applyCam(); renderer.setClearColor(new THREE.Color(cssVar('--scene'))); renderer.render(scene, camera); syncLabels(); syncLateral(); }\n")
rep("  cv3.addEventListener('pointerup', () => { drag = null; });",
    "  cv3.addEventListener('pointerup', () => { if (drag && !drag.moved && clickCb) clickCb(lastHit); drag = null; });")
# a click is a press and release that travelled at most 4 px (a tap jitters); with onClick set, the press picks first, so
# a tap with no hover before it still knows its cube. `moved` was set but read nowhere, so the CNN pages are unaffected.
rep("shift: e.shiftKey, moved: false }; cv3.setPointerCapture(e.pointerId); });",
    "shift: e.shiftKey, moved: false, travel: 0 }; cv3.setPointerCapture(e.pointerId); if (clickCb) pick(e); });")
rep("if (Math.abs(dx) + Math.abs(dy) > 0) drag.moved = true;",
    "drag.travel += Math.abs(dx) + Math.abs(dy); if (drag.travel > 4) drag.moved = true;")
rep("  let pickCb = null, lastKey = null;", "  let pickCb = null, lastKey = null, clickCb = null, lastHit = null;")
rep("    if (key !== lastKey) { lastKey = key; if (pickCb) pickCb(hit); } }",
    "    lastHit = hit; if (key !== lastKey) { lastKey = key; if (pickCb) pickCb(hit); } }")
rep("  cv3.addEventListener('pointerleave', () => { drag = null; if (lastKey !== null) {",
    "  cv3.addEventListener('pointerleave', () => { drag = null; lastHit = null; if (lastKey !== null) {")
rep("  Object.assign(view, { setBlocks, place, refresh, blockById, voxelPos, bounds, flowRange, flowAxis, centerOf, boxAt, boxSpanZ, connect, setGoal, cam, goal: camGoal, resize,\n"
    "    onPick: (cb) => { pickCb = cb; },",
    "  Object.assign(view, { setBlocks, place, refresh, blockById, voxelPos, bounds, flowRange, flowAxis, centerOf, boxAt, boxSpanZ, connect, setGoal, cam, goal: camGoal, resize,\n"
    "    strengthBox, strengthLine, fit, snap, onClick: (cb) => { clickCb = cb; },\n"
    "    onPick: (cb) => { pickCb = cb; },")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ui/blocks3d.js: opt-in additions for vit-3d/')
```

Run: `python .superpowers/check/edit_blocks3d.py ui/blocks3d.js`
Expected: `ui/blocks3d.js: opt-in additions for vit-3d/`.

- [ ] **Step 4: The CNN scenes are unchanged**

Run: `node .superpowers/check/unchanged.mjs compare cnn3d`
Expected:
```
IDENTICAL cnn-3d
IDENTICAL cnn-train 3D
cnn3d: unchanged
```

- [ ] **Step 5: Commit**

```bash
git add ui/blocks3d.js
git commit -m "feat(ui): graded highlights, a camera fit and a click in blocks3d" -m "Opt-in calls for the ViT 3D explorer: strengthBox and strengthLine draw green as strong as a trace says, fit frames a set of blocks, snap draws the goal frame at once, onClick reports a click that is not a drag. The CNN 3D explorer and the training page's 3D view keep identical scene digests." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The page

**Files:**
- Create: `vit-3d/index.html`
- Create: `.superpowers/check/shot3d.js`, `scen_vit3d.js`, `scen_readouts.js`, `scen_vit3d_noweights.js`, `scen_frames.js`, `vit3d_checks.mjs`

**Interfaces:**
- Consumes everything above, plus:
  - `image` (`core/data.js`);
  - `shuffledOrder` (`core/train.js`);
  - `fmt` (`ui/grid.js`);
  - `installTerms` and `VIT_TERMS` (`ui/terms.js`).
- The page's module exposes, for the checks: `ready`, `view`, `frames`, `hover`, `pinned`, `sweep`, `S`, `probeOf`, `updateHighlights`, `updateSweep`, `camPreset`, `setLayout`, `GROUPS`, `ORDER`, `LAYOUT_KEY`. The checks run inside the module, so they see these variables.
- **Layout:**
  - One row of blocks.
  - A bias sits under its matrix.
  - Wq, Wk and Wv stand in one column, with Q, K and V beside them.
  - Every learned weight, with its bias, sits in one accent frame (12 frames).
  - Side by side or depth, remembered under `dl-explorer.vit3d.layout`.
- **Camera presets** (whole, embedding, attention, MLP, head) frame their blocks with `view.fit`, from theta, phi:
  - side: whole 0.3, 1.2; a group 0.45, 1.12;
  - depth: whole 1.25, 1.15, from the side, where the chain of 36 blocks runs across the screen; a group 0.9, 1.1.

- [ ] **Step 1: The page**

Create `vit-3d/index.html` with the Write tool (it contains `\n` in its labels):

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ViT explorer 3D — tokens where their patches are</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&display=swap">
<link rel="stylesheet" href="../ui/terms.css">
<style>
:root{
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
  --bg:#f7f8f6; --panel:#ffffff; --ink:#1d2430; --ink2:#5b6474; --line:#d9ddd6; --line2:#eef0ec;
  --accent:#2457a8; --accent-soft:#dde7f7; --scene:#eef0ea; --wtint:rgba(36,87,168,.07);
  --hl:#ffd94a; --hl2:#3fae4b;
  --mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --scene:#1b1f26; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
}}
:root[data-theme="dark"]{
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --scene:#1b1f26; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
}
html{scroll-padding-top:env(safe-area-inset-top,0px);height:100%}
*{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:14.5px;line-height:1.45;-webkit-font-smoothing:antialiased;height:100%}
main{max-width:1280px;margin:0 auto;padding:22px 18px 60px}
main a{color:var(--accent)}
h1{font-size:26px;font-weight:600;margin:0 0 4px}
h2{font-size:15px;font-weight:600;margin:14px 0 4px}
.lede{color:var(--ink2);margin:0 0 14px;max-width:none}
.wrap{display:grid;grid-template-columns:1fr 300px;gap:16px;align-items:start}
@media (max-width:860px){ .wrap{grid-template-columns:1fr} }
#view{position:relative;background:var(--scene);border:1px solid var(--line);border-radius:10px;height:min(600px,72vh);min-height:320px;overflow:hidden;touch-action:none}
#view canvas{display:block;width:100%!important;height:100%!important}
.lbl{position:absolute;transform:translate(-50%,-100%);font-size:12.5px;color:var(--ink2);text-align:center;pointer-events:none;white-space:nowrap;line-height:1.3}
.lbl b{display:block;color:var(--ink);font-weight:600}
#readout{position:absolute;left:10px;bottom:10px;right:10px;font-size:13px;background:color-mix(in srgb,var(--panel) 88%,transparent);border:1px solid var(--line);border-radius:7px;padding:8px 12px;pointer-events:none;min-height:2.2em}
#readout b{font-weight:600}
.cams{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--panel)}
.seg button{border:0;border-radius:0}
.seg button+button{border-left:1px solid var(--line)}
.lat{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:12.5px;color:var(--ink2)}
.lat input{flex:1}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
button,select{font:inherit;font-size:13px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 9px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.sel{background:var(--accent-soft);border-color:var(--accent)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.ctrls{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:6px 0}
.digits button{min-width:26px;padding:4px 0}
.ctl{display:inline-flex;align-items:center;gap:5px;font-size:13px;color:var(--ink2)}
input[type=range]{width:100%;accent-color:var(--accent)}
.hint{font-size:12.5px;color:var(--ink2)}
.hint b{color:var(--ink)}
.banner{font-size:13.5px;background:var(--accent-soft);border:1px solid var(--accent);border-radius:8px;padding:8px 12px;margin:0 0 14px;max-width:78ch}
.banner a{color:var(--accent)}
.keys{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:8px 0 0}
.keys span{display:inline-flex;align-items:center;gap:5px}
.keys i{display:inline-block;flex:none;width:12px;height:12px;border-radius:3px;border:3px solid var(--hl)}
.keys i.g{border-color:var(--hl2)}
.keys i.wt{border:1px solid var(--accent);background:var(--wtint)}
.crumb{font-size:13px;color:var(--ink2);margin:0 0 10px}.crumb a{color:var(--accent);text-decoration:none}.crumb a:hover{text-decoration:underline}
@media (prefers-reduced-motion: reduce){ *{transition:none!important} }
</style>
</head>
<body>
<main>
<nav class="crumb"><a href="../">Visualising AI</a> › Vision Transformer, 3D explorer</nav>
<h1>ViT explorer, in 3D</h1>
<p class="lede">The toy Vision Transformer of the <a href="../vit/">explorer</a>, drawn as blocks of cubes, one cube per number. Each token sits where its patch was: the 16 tokens make a 4×4 block whose depth is the token's 8 numbers, the shape of a CNN feature map. Each head's attention map is a slab of 16 little 4×4 maps, one at each token's place, showing where in the image that token looks. Cube size is magnitude and colour is sign (blue negative, orange positive). Drag to orbit, scroll to zoom, right-drag to pan. Hover a cube to see what it is computed from; click to pin it while you turn the view.</p>
<div class="wrap">
  <div>
    <div id="view">
      <div class="cams"><button data-cam="whole" class="sel">whole</button><button data-cam="embedding">embedding</button><button data-cam="attention">attention</button><button data-cam="MLP">MLP</button><button data-cam="head">head</button></div>
      <div class="cams" style="left:auto;right:10px"><span class="seg"><button id="laySide" class="sel">side by side</button><button id="layDepth">depth</button></span></div>
      <div id="readout">Hover a cube.</div>
    </div>
    <div class="lat"><span>start</span><input type="range" id="lateral" min="0" max="1000" value="500" title="scroll along the network"><span>end</span></div>
  </div>
  <div class="panel">
    <h2 style="margin-top:0">Digit</h2>
    <div class="ctrls"><span class="digits" id="digits"></span><button id="another">another</button></div>
    <div class="hint" id="dlab"></div>
    <h2>Weights and position</h2>
    <div class="ctrls"><span class="seg"><button id="wTr" class="sel">trained</button><button id="wRa">random</button></span>
      <span class="seg"><button id="pOn" class="sel">position on</button><button id="pOff">off</button></span></div>
    <div class="ctrls"><button id="shuf">Shuffle patches</button><button id="unshuf" disabled>Restore order</button></div>
    <div class="hint" id="wsrc"></div>
    <h2>Patch sweep</h2>
    <div class="ctrls"><button id="swPrev">←</button><button id="swPlay" class="primary">Play</button><button id="swNext">→</button><button id="swOff">off</button></div>
    <input type="range" id="swStep" min="0" max="15" value="0">
    <div class="hint" id="swHint">The patch embedding is a convolution whose kernel is as big as its stride: 8 kernels of 4×4 jump 4 pixels at a time, and each patch becomes one token of 8 numbers.</div>
    <div class="keys"><span><i></i> the window being computed</span><span><i class="g"></i> what a value is computed from, after attention by attention weight</span><span><i class="wt"></i> learned weights</span></div>
    <div class="hint" style="margin-top:10px" id="wnote"></div>
  </div>
</div>
<div class="panel" style="margin-top:16px">
  <h2 style="margin-top:0">Terms</h2>
  <div class="hint" style="margin-bottom:8px">Hover or tap a dotted term above for the same definition in place.</div>
  <div class="gloss" id="gloss"></div>
</div>
</main>
<div id="tip" role="tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script type="module">
import { P, T, D, NC, init, forward, tensor, vitTrace } from '../core/vit.js';
import { image } from '../core/data.js';
import { shuffledOrder } from '../core/train.js';
import { VIT_BLOCKS, blockOf, cubeToCell, sliceValues, traceBoxes, attentionFan } from '../core/vit3d.js';
import { fmt } from '../ui/grid.js';
import { installTerms, VIT_TERMS } from '../ui/terms.js';
import { describe, weightWords, loadDigitsAndWeights } from '../ui/vitwords.js';
import { createBlockView } from '../ui/blocks3d.js';
const $ = (id) => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const LAYOUT_KEY = 'dl-explorer.vit3d.layout';

// ---------- state ----------
const wRandom = init(1);
let wTrained = null, meta = null, useTrained = true;
let data = null, byClass = [], pick = { cls: 7, n: 0 }, img = null;
let position = true, order = null, shuffles = 0;
let hover = null, pinned = null, sweep = null, swTimer = null;
let S = null;
const W = () => (useTrained && wTrained ? wTrained : wRandom);
const value = (t, r, c) => tensor(t, S, W())[r][c];

// ---------- the scene (blocks drawn by ../ui/blocks3d.js; which tensor each block draws, in ../core/vit3d.js) ----------
let layout = 'side'; try { layout = localStorage.getItem(LAYOUT_KEY) || 'side'; } catch { /* storage blocked */ }
const view = createBlockView($('view'), { explode: 0.5, layout, gapX: 5, gapZ: 9, lateral: $('lateral'), ease: reduce ? 1 : 0.12, declutter: true });
const frames = new THREE.Group(); view.scene.add(frames);
const frameMat = new THREE.LineBasicMaterial({ color: 0x2457a8, transparent: true, opacity: 0.6 });
// A label is a name and a line under it (blocks3d splits at \n). Biases, K and V share the label of the block above them.
const LABEL = { digit: 'digit\n16×16', Wp: 'patch kernels\n8 of 4×4', xp: 'tokens\n4×4×8', pos: 'position\n4×4×8', z0: 'z₀\ntokens + position', ln1: 'LN₁\nscale, shift', h1: 'LN₁(z₀)\n4×4×8',
  Wq: 'Wq Wk Wv\n8×8 each', q: 'Q K V\nhead 1 front, 2 back', att: 'attention\na 4×4 map per token', o: 'A·V\nheads joined', Wo: 'Wo\n8×8', z1: 'z₁\nz₀ + mix', ln2: 'LN₂\nscale, shift', h2: 'LN₂(z₁)\n4×4×8',
  W1: 'W₁\n8×32', hid: 'hidden\nGELU, 4×4×32', W2: 'W₂\n32×8', z2: 'z₂\nz₁ + MLP', avg: 'average\n8', lnf: 'LN\nscale, shift', avgNorm: 'normalised\n8', Wh: 'W head\n8×10', prob: 'classes\n10' };
// Stacks: a bias sits under its matrix, Wq Wk Wv in one column, Q K V beside them. The smaller block comes first and the
// larger takes its place (`at`), so the next block starts after the larger one.
const PLACE = {
  bp: { dy: -3.4 }, Wp: { at: 'bp' },
  bq: { dy: 5.8 }, bk: { at: 'bq', dy: -5.2 }, bv: { at: 'bq', dy: -16.2 }, Wq: { at: 'bq', dy: 11 }, Wk: { at: 'bq' }, Wv: { at: 'bq', dy: -11 },
  q: { dy: 11 }, k: { at: 'q' }, v: { at: 'q', dy: -11 },
  bo: { dy: -5.2 }, Wo: { at: 'bo' }, b1: { dy: -5.2 }, W1: { at: 'b1' }, b2: { dy: -17.2 }, W2: { at: 'b2' }, bh: { dy: -5.2 }, Wh: { at: 'bh' },
};
const ORDER = ['digit', 'bp', 'Wp', 'xp', 'pos', 'z0', 'ln1', 'h1', 'bq', 'bk', 'bv', 'Wq', 'Wk', 'Wv', 'q', 'k', 'v', 'att', 'o', 'bo', 'Wo', 'z1', 'ln2', 'h2', 'b1', 'W1', 'hid', 'b2', 'W2', 'z2', 'avg', 'lnf', 'avgNorm', 'bh', 'Wh', 'prob'];
let scales = {};
const tensorsOf = (key) => (key === 'att' ? ['att0', 'att1'] : [key]);
function scaleOf(key) {
  if (key === 'digit') return 1;
  return scales[key] ??= Math.max(0.01, ...tensorsOf(key).flatMap((t) => tensor(t, S, W()).flat().map(Math.abs)));
}
function defs() {
  return ORDER.map((key) => { const b = blockOf(key);
    return { id: key, rows: b.rows, cols: b.cols, chans: b.chans, mode: key === 'digit' ? 'gray' : 'div', scale: () => scaleOf(key), data: (ch) => sliceValues(key, ch, value), label: LABEL[key] || '', ...(PLACE[key] || {}) }; });
}
// The learned weights in accent frames, as on the CNN training page's 3D view: a matrix and its bias share one frame.
const FRAMED = [['Wp', 'bp'], ['pos'], ['ln1'], ['Wq', 'bq'], ['Wk', 'bk'], ['Wv', 'bv'], ['Wo', 'bo'], ['ln2'], ['W1', 'b1'], ['W2', 'b2'], ['lnf'], ['Wh', 'bh']];
function drawFrames() {
  frames.clear(); frameMat.color.set(getComputedStyle(document.documentElement).getPropertyValue('--accent').trim());
  for (const ids of FRAMED) {
    const bx = new THREE.Box3(); for (const id of ids) { const b = view.blockById(id); bx.union(view.bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1)); }
    bx.expandByScalar(0.4); const size = bx.getSize(new THREE.Vector3());
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)), frameMat); e.position.copy(bx.getCenter(new THREE.Vector3())); frames.add(e);
  }
}

// ---------- the green: hover or pin (core vitTrace and its boxes) ----------
const probeOf = (hit) => (hit ? cubeToCell(hit.block.id, hit.r, hit.c, hit.ch) : null);
const shown = () => hover || pinned;
function tokenCentre(key, j) { const b = view.blockById(key); return view.voxelPos(b, j >> 2, j & 3, 0).add(view.voxelPos(b, j >> 2, j & 3, b.chans - 1)).multiplyScalar(0.5); }
function updateHighlights() {
  view.hl.clear(); const p = shown();
  if (!p || !S) { $('readout').textContent = 'Hover a cube.'; return; }
  const trace = vitTrace(S, p);
  for (const x of traceBoxes(trace)) view.strengthBox(view.blockById(x.key), x.r0, x.r1, x.c0, x.c1, x.ch0, x.ch1, x.s, view.hl);
  const att = view.blockById('att');
  for (const f of attentionFan(trace, p)) { // from token i's little map, at token j's place, to the value token j handed over
    const from = view.voxelPos(att, (f.i >> 2) * 4 + (f.j >> 2), (f.i & 3) * 4 + (f.j & 3), f.h);
    view.strengthLine(from, tokenCentre('v', f.j), f.s, view.hl);
  }
  $('readout').innerHTML = describe(p, S, W()) + (pinned && !hover ? ' <i>(pinned: click empty space to release)</i>' : '');
}

// ---------- the patch sweep (yellow) ----------
function updateSweep() {
  view.aux.clear(); const { swFill, swEdge, swOut, swLine } = view.mats;
  $('swStep').value = sweep ?? 0;
  if (sweep == null || !S) { $('swHint').textContent = 'The patch embedding is a convolution whose kernel is as big as its stride: 8 kernels of 4×4 jump 4 pixels at a time, and each patch becomes one token of 8 numbers.'; return; }
  const s = sweep, pr = s >> 2, pc = s & 3, dg = view.blockById('digit'), kb = view.blockById('Wp'), xb = view.blockById('xp');
  view.boxAt(dg, pr * P, pr * P + P - 1, pc * P, pc * P + P - 1, 0, view.aux, swFill, swEdge);
  view.boxSpanZ(kb, 0, P - 1, 0, P - 1, 0, D - 1, view.aux, swFill, swEdge);
  view.boxSpanZ(xb, pr, pr, pc, pc, 0, D - 1, view.aux, swOut, swEdge);
  view.connect(view.bounds(dg, pr * P, pr * P + P - 1, pc * P, pc * P + P - 1, 0, 0), view.bounds(xb, pr, pr, pc, pc, 0, D - 1), swLine, view.aux);
  $('swHint').innerHTML = `Patch ${s} (rows ${pr * P}–${pr * P + P - 1}, columns ${pc * P}–${pc * P + P - 1}): each of the 8 kernels multiplies its 16 pixels and adds its bias, giving token ${s} its 8 numbers: <b>${S.xp[s].map(fmt).join(', ')}</b>.`;
}
function stopSweep() { clearInterval(swTimer); swTimer = null; $('swPlay').textContent = 'Play'; }
$('swPlay').onclick = () => {
  if (swTimer) return stopSweep();
  if (reduce) { sweep = T - 1; return updateSweep(); }
  if (sweep == null || sweep >= T - 1) sweep = -1;
  $('swPlay').textContent = 'Pause';
  swTimer = setInterval(() => { sweep++; updateSweep(); if (sweep >= T - 1) stopSweep(); }, 450);
};
$('swPrev').onclick = () => { stopSweep(); sweep = Math.max(0, (sweep ?? 0) - 1); updateSweep(); };
$('swNext').onclick = () => { stopSweep(); sweep = ((sweep ?? -1) + 1) % T; updateSweep(); };
$('swStep').oninput = (e) => { stopSweep(); sweep = +e.target.value; updateSweep(); };
$('swOff').onclick = () => { stopSweep(); sweep = null; updateSweep(); };

// ---------- camera presets and layout ----------
const GROUPS = ['embedding', 'attention', 'MLP', 'head'];
// theta, phi, padding. Depth sends 36 blocks away from you, so its whole view looks from the side, where the chain runs
// across the screen; a group is seen from three-quarters, as on cnn-3d.
const CAM = { side: { whole: [0.3, 1.2, 1.03], group: [0.45, 1.12, 1.06] }, depth: { whole: [1.25, 1.15, 1.04], group: [0.9, 1.1, 1.06] } };
let preset = 'whole';
function camPreset(name) {
  preset = name; const ids = name === 'whole' ? ORDER : VIT_BLOCKS.filter((b) => b.group === name).map((b) => b.key);
  const [theta, phi, pad] = CAM[view.layout][name === 'whole' ? 'whole' : 'group'];
  view.fit(ids, theta, phi, pad);
  document.querySelectorAll('.cams button[data-cam]').forEach((x) => x.classList.toggle('sel', x.dataset.cam === name));
}
document.querySelectorAll('.cams button[data-cam]').forEach((x) => { x.onclick = () => camPreset(x.dataset.cam); });
function setLayout(l) {
  view.layout = l; try { localStorage.setItem(LAYOUT_KEY, l); } catch { /* storage blocked */ }
  view.blockById('bp').dx = l === 'side' ? 1.5 : 0; // side by side a block starts at its left edge, in depth it is centred
  $('laySide').classList.toggle('sel', l === 'side'); $('layDepth').classList.toggle('sel', l === 'depth');
  view.refresh(); drawFrames(); updateHighlights(); updateSweep(); camPreset(preset);
}
$('laySide').onclick = () => setLayout('side'); $('layDepth').onclick = () => setLayout('depth');
view.onPick((hit) => { const p = probeOf(hit); if (JSON.stringify(p) !== JSON.stringify(hover)) { hover = p; updateHighlights(); } });
view.onClick((hit) => { pinned = probeOf(hit); hover = null; updateHighlights(); });

// ---------- compute ----------
function compute() {
  if (!img) return;
  S = forward(W(), img, { position, order }); scales = {};
  view.refresh(); updateHighlights(); updateSweep();
  const words = weightWords({ S, trained: useTrained && !!wTrained, meta, position, order, data });
  $('wsrc').innerHTML = words.src; $('wnote').textContent = words.note;
}

// ---------- controls ----------
function setDigit() { if (!data) return; const list = byClass[pick.cls], i = list[pick.n % list.length]; img = image(data.test, i); $('dlab').textContent = `held-out digit ${i}, a ${pick.cls}`; }
function syncControls() {
  [...$('digits').children].forEach((b, c) => b.classList.toggle('sel', c === pick.cls));
  $('wTr').classList.toggle('sel', useTrained && !!wTrained); $('wRa').classList.toggle('sel', !(useTrained && wTrained)); $('wTr').disabled = !wTrained;
  $('pOn').classList.toggle('sel', position); $('pOff').classList.toggle('sel', !position); $('unshuf').disabled = !order;
}
for (let c = 0; c < NC; c++) { const b = document.createElement('button'); b.textContent = c; b.onclick = () => { pick = { cls: c, n: 0 }; setDigit(); syncControls(); compute(); }; $('digits').append(b); }
$('another').onclick = () => { pick.n++; setDigit(); compute(); };
const on = (id, f) => { $(id).onclick = () => { f(); syncControls(); compute(); }; };
on('wTr', () => { useTrained = true; }); on('wRa', () => { useTrained = false; });
on('pOn', () => { position = true; }); on('pOff', () => { position = false; });
on('shuf', () => { order = shuffledOrder(T, ++shuffles); }); on('unshuf', () => { order = null; });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { drawFrames(); compute(); });

// ---------- start ----------
async function start() {
  installTerms({ selectors: '.lede, .hint:not(#swHint):not(#wnote):not(#wsrc):not(#dlab)', tip: $('tip'), gloss: $('gloss'), terms: VIT_TERMS });
  const got = await loadDigitsAndWeights(document.querySelector('h1'), '../data/mnist16.bin', '../models/vit16.json');
  if (!got) { document.querySelectorAll('.panel button, .panel input').forEach((x) => { x.disabled = true; }); return; }
  ({ data, byClass, wTrained, meta } = got); useTrained = !!wTrained;
  setDigit(); syncControls();
  S = forward(W(), img, { position, order });
  view.resize(); view.setBlocks(defs()); setLayout(layout); compute(); view.start();
}
const ready = start();
</script>
</body>
</html>
```

- [ ] **Step 2: The checks**

Create `.superpowers/check/shot3d.js`, which reads rendered frames back from the page:

```js
// Rendered frames of a WebGL page, read back in the page (headless --screenshot comes out black with a capped loop):
// `await shot(name)` jumps the camera to its goal, draws, copies the canvas with its HTML labels and the readout onto a
// 2D canvas and stores the PNG in R.shots[name]. Prepend to a scenario; tools/../shoot.mjs writes the PNGs out.
await ready;
R.shots = {};
const shot = async (name) => {
  view.snap();
  const el = $('view'), w = el.clientWidth, h = el.clientHeight, url = view.renderer.domElement.toDataURL('image/png');
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h; const x = cv.getContext('2d');
  const im = new Image(); im.src = url; await im.decode(); x.drawImage(im, 0, 0, w, h);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  x.textAlign = 'center';
  for (const l of el.querySelectorAll('.lbl')) {
    if (l.style.display === 'none') continue;
    const lx = parseFloat(l.style.left), ly = parseFloat(l.style.top), [t, sh] = l.innerHTML.split('</b>');
    x.font = '600 12.5px system-ui'; x.fillStyle = css('--ink'); x.fillText(t.replace('<b>', ''), lx, ly - 18);
    x.font = '12.5px system-ui'; x.fillStyle = css('--ink2'); x.fillText(sh, lx, ly - 3);
  }
  x.textAlign = 'left'; x.font = '12px system-ui'; x.fillStyle = css('--ink');
  const words = $('readout').textContent.split(' '); let line = '', y = h - 40; // the readout, wrapped
  for (const wd of words) { if (x.measureText(line + wd).width > w - 30) { x.fillText(line, 14, y); line = ''; y += 15; } line += wd + ' '; }
  x.fillText(line, 14, y);
  R.shots[name] = cv.toDataURL('image/png');
};
```

Create `.superpowers/check/scen_vit3d.js`:

```js
// vit-3d/: the scene, the cube → probe mapping and readouts, highlight counts, the sweep, the pin, presets in view,
// the controls. Run by vit3d_checks.mjs, which compares the readouts with vit/'s (scen_readouts.js).
await ready;
const gl = view.renderer.getContext(); R.webgl = !!gl && !gl.isContextLost();
R.blocks = view.blocks.length; R.cubes = view.blocks.reduce((t, b) => t + b.mesh.count, 0);
R.trained = !!wTrained; R.wsrc = $('wsrc').textContent; R.dlab = $('dlab').textContent;
R.glossary = $('gloss').children.length; R.dfn = document.querySelectorAll('dfn').length;
R.overflow = document.documentElement.scrollWidth - innerWidth;
R.frames = frames.children.length;

// a cube → the probe of the cell it draws → the readout
const CUBES = [['xp', 1, 2, 3], ['z1', 1, 2, 0], ['att', 5, 9, 1], ['Wp', 2, 1, 6], ['hid', 3, 0, 17], ['prob', 7, 0, 0], ['W1', 4, 20, 0], ['ln2', 1, 0, 5], ['digit', 9, 6, 0], ['o', 1, 2, 5]];
R.readouts = CUBES.map(([id, r, c, ch]) => { hover = probeOf({ block: view.blockById(id), r, c, ch }); updateHighlights(); return [hover, $('readout').textContent]; });

// green boxes and fan links for known probes: what the page drew against what traceBoxes and attentionFan say
const count = (p) => { hover = p; updateHighlights(); const tr = vitTrace(S, p), bx = traceBoxes(tr), fan = attentionFan(tr, p);
  return { drawn: view.hl.children.length, expected: bx.length + bx.filter((b) => b.s >= 0.999).length + fan.length, boxes: bx.length, links: fan.length }; };
R.counts = { xp: count({ t: 'xp', r: 6, c: 3 }), o: count({ t: 'o', r: 6, c: 5 }), z1: count({ t: 'z1', r: 6, c: 2 }), att: count({ t: 'att1', r: 6, c: 5 }) };
hover = null; updateHighlights();

// the sweep: 16 positions, each a patch box, a kernel box, a token box (fill and edges) and the connector
$('swOff').click(); const seen = [];
for (let k = 0; k < 16; k++) { $('swNext').click(); seen.push([sweep, view.aux.children.length]); }
R.sweep = { positions: new Set(seen.map((x) => x[0])).size, aux: [...new Set(seen.map((x) => x[1]))], hint: $('swHint').textContent.slice(0, 60) };
$('swOff').click(); R.sweepOff = view.aux.children.length;

// the pin, with synthetic pointer events on the canvas
camPreset('attention'); view.snap();
const cv = view.renderer.domElement, rect = cv.getBoundingClientRect();
const screenOf = (id, r, c, ch) => { const p = view.voxelPos(view.blockById(id), r, c, ch).project(view.camera); return [rect.left + (p.x + 1) / 2 * rect.width, rect.top + (1 - p.y) / 2 * rect.height]; };
const ev = (type, [x, y]) => cv.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, pointerId: 1, pointerType: 'mouse', isPrimary: true, bubbles: true, button: 0, buttons: type === 'pointerup' ? 0 : 1 }));
const at = screenOf('z1', 1, 2, 0), empty = [rect.left + 4, rect.top + rect.height - 4];
ev('pointermove', at); R.pin = { hovered: JSON.stringify(hover) };
ev('pointerdown', at); ev('pointerup', at); R.pin.pinned = JSON.stringify(pinned); const lit = view.hl.children.length; R.pin.lit = lit;
const th = view.cam.theta; ev('pointerdown', at); ev('pointermove', [at[0] + 60, at[1] + 10]); ev('pointerup', [at[0] + 60, at[1] + 10]);
R.pin.orbited = +(view.cam.theta - th).toFixed(3); R.pin.keptAfterDrag = JSON.stringify(pinned) === R.pin.pinned && view.hl.children.length === lit;
ev('pointermove', empty); R.pin.keptOffCubes = hover === null && view.hl.children.length === lit;
ev('pointerdown', empty); ev('pointerup', empty); R.pin.released = pinned === null && view.hl.children.length === 0;

// every preset keeps its blocks inside the view: the largest projected corner coordinate (1 is the edge)
R.presets = {};
for (const l of ['side', 'depth']) { setLayout(l);
  for (const name of ['whole', ...GROUPS]) { camPreset(name); view.snap(); view.camera.updateMatrixWorld();
    const ids = name === 'whole' ? ORDER : VIT_BLOCKS.filter((b) => b.group === name).map((b) => b.key); let m = 0;
    for (const id of ids) { const b = view.blockById(id), bx = view.bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1);
      for (const x of [bx.min.x, bx.max.x]) for (const y of [bx.min.y, bx.max.y]) for (const z of [bx.min.z, bx.max.z]) { const p = new THREE.Vector3(x, y, z).project(view.camera); m = Math.max(m, Math.abs(p.x), Math.abs(p.y)); } }
    R.presets[`${l} ${name}`] = +m.toFixed(3); } }
R.stored = localStorage.getItem(LAYOUT_KEY); setLayout('side');

// controls: with position off, shuffling the patches leaves the prediction as it was; with position on, it changes it
$('pOff').click(); const p0 = S.prob.slice(); $('shuf').click(); R.offDiff = +Math.max(...p0.map((x, c) => Math.abs(x - S.prob[c]))).toFixed(6);
$('pOn').click(); const p2 = S.prob.slice(); $('unshuf').click(); R.onDiff = +Math.max(...p2.map((x, c) => Math.abs(x - S.prob[c]))).toFixed(6);
$('wRa').click(); R.randomNote = $('wnote').textContent.slice(0, 40); R.randomSrc = $('wsrc').textContent.slice(0, 40);
$('wTr').click(); R.trainedNote = $('wnote').textContent.slice(0, 40);
$('digits').children[3].click(); R.digit3 = $('dlab').textContent; $('another').click(); R.another = $('dlab').textContent;
R.banner = (document.querySelector('.banner') || {}).textContent || null;
```

Create `.superpowers/check/scen_readouts.js`, the same cells on `vit/`:

```js
// vit/: the readout for the cells that scen_vit3d.js reaches through cubes, for vit3d_checks.mjs to compare.
await ready;
const P = [{ t: 'xp', r: 6, c: 3 }, { t: 'z1', r: 6, c: 0 }, { t: 'att1', r: 6, c: 5 }, { t: 'Wp', r: 6, c: 9 }, { t: 'hid', r: 12, c: 17 }, { t: 'prob', r: 0, c: 7 },
  { t: 'W1', r: 4, c: 20 }, { t: 'ln2', r: 1, c: 5 }, { t: 'digit', r: 9, c: 6 }, { t: 'o', r: 6, c: 5 }];
R.readouts = P.map((p) => { setProbe(p); return [p, $('readout').textContent]; });
```

Create `.superpowers/check/scen_vit3d_noweights.js`:

```js
// vit-3d/ with models/vit16.json missing (run with --no-weights): random weights, the banner, "trained" disabled.
await ready;
R.banner = (document.querySelector('.banner') || {}).textContent || null;
R.trDisabled = $('wTr').disabled; R.wsrc = $('wsrc').textContent;
R.webgl = !!view.renderer.getContext();
```

Create `.superpowers/check/scen_frames.js`:

```js
// vit-3d/: rendered frames for every preset in both layouts, a hover with its fan, a pin seen after an orbit, and the
// sweep (needs shot3d.js before it; vit3d_checks.mjs prepends it and writes the PNGs to .superpowers/check/shots/).
for (const l of ['side', 'depth']) { setLayout(l); for (const name of ['whole', ...GROUPS]) { camPreset(name); await shot(`${l}-${name}`); } }
setLayout('side'); camPreset('attention'); hover = { t: 'z1', r: 6, c: 2 }; updateHighlights(); await shot('hover-z1');
hover = { t: 'att0', r: 5, c: 9 }; updateHighlights(); await shot('hover-att');
hover = null; pinned = { t: 'o', r: 6, c: 1 }; updateHighlights(); view.setGoal({ theta: view.goal.theta + 0.5 }); await shot('pinned-orbited');
pinned = null; updateHighlights(); camPreset('embedding'); sweep = 9; updateSweep(); await shot('sweep');
```

Create `.superpowers/check/vit3d_checks.mjs`, which runs them all:

```js
// vit3d_checks.mjs — every check of vit-3d/ in one run: node .superpowers/check/vit3d_checks.mjs [--no-frames]
// Runs the page's scenarios in headless Edge (vitcheck.mjs --three, one at a time: they share port 8123), compares the
// readouts with vit/'s, prints PASS or FAIL per check, and writes rendered frames, light and dark, to
// .superpowers/check/shots/ for viewing. Exits 1 if any check fails.
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const chk = dirname(fileURLToPath(import.meta.url)), root = join(chk, '..', '..');
const run = (scen, ...flags) => JSON.parse(execFileSync('node', [join(chk, 'vitcheck.mjs'), root, scen, ...flags], { encoding: 'utf8', maxBuffer: 1 << 30, timeout: 240000 }));
let failed = 0;
const ok = (cond, what, got) => { if (!cond) failed++; console.log(`${cond ? 'PASS' : 'FAIL'}  ${what}${cond ? '' : `  (got ${JSON.stringify(got)})`}`); };

const A = run(join(chk, 'scen_vit3d.js'), '--page', 'vit-3d', '--three');
if (A.error) console.log(A.error);
ok(!A.error && A.consoleErrors.length === 0, 'no scenario or console errors', [A.error, A.consoleErrors]);
ok(A.webgl, 'a live WebGL context', A.webgl);
ok(A.blocks === 36 && A.cubes === 3828, '36 blocks, 3,828 cubes', [A.blocks, A.cubes]);
ok(A.frames === 12, '12 accent frames on the learned weights', A.frames);
ok(A.trained && /prediction 7 \(\d+\.\d%\), trained weights: \d+\.\d% right on held-out digits/.test(A.wsrc), 'trained weights and their accuracy beside the prediction', A.wsrc);
ok(A.overflow <= 0, 'no horizontal page scroll', A.overflow);
ok(A.glossary > 0 && A.dfn > 0, 'terms linked and listed', [A.dfn, A.glossary]);
const B = run(join(chk, 'scen_readouts.js'), '--page', 'vit');
ok(JSON.stringify(A.readouts) === JSON.stringify(B.readouts), 'cubes give the probes and readouts vit/ gives for the same cells', A.readouts.map((x, i) => x[1] === (B.readouts[i] || [])[1]));
for (const [k, c] of Object.entries(A.counts)) ok(c.drawn === c.expected, `${k} probe: ${c.boxes} green boxes and ${c.links} attention links drawn`, c);
ok(A.counts.xp.links === 0 && A.counts.z1.links > 0 && A.counts.o.links > 0, 'a fan after attention only', A.counts);
ok(A.sweep.positions === 16 && A.sweep.aux.length === 1 && A.sweep.aux[0] === 7 && A.sweepOff === 0, 'the sweep visits 16 patches, each with its patch, kernels, token and connector', A.sweep);
ok(A.pin.hovered === A.pin.pinned && A.pin.lit > 0, 'a click on a hovered cube pins its probe', A.pin);
ok(Math.abs(A.pin.orbited) > 0.1 && A.pin.keptAfterDrag, 'the pin survives an orbit (a drag is not a click)', A.pin);
ok(A.pin.keptOffCubes, 'leaving the cubes returns to the pinned probe', A.pin);
ok(A.pin.released, 'a click on empty space releases it', A.pin);
ok(Object.values(A.presets).every((m) => m > 0.85 && m <= 1), 'every preset, in both layouts, frames its blocks inside the view', A.presets);
ok(A.stored === 'depth', 'the layout is remembered', A.stored);
ok(A.offDiff < 1e-9 && A.onDiff > 1e-3, 'shuffling changes the prediction only with position on', [A.offDiff, A.onDiff]);
ok(/^These are random/.test(A.randomNote) && /random weights: ar/.test(A.randomSrc) && /^These weights were trained/.test(A.trainedNote), 'the weights switch and its wording', [A.randomNote, A.randomSrc, A.trainedNote]);
ok(/a 3$/.test(A.digit3) && A.another !== A.digit3 && /a 3$/.test(A.another), 'the digit picker and "another"', [A.digit3, A.another]);
ok(A.banner === null, 'no banner with the shipped weights', A.banner);

const N = run(join(chk, 'scen_vit3d_noweights.js'), '--page', 'vit-3d', '--three', '--no-weights');
ok(/Could not load the trained weights/.test(N.banner) && N.trDisabled && /random weights/.test(N.wsrc) && N.consoleErrors.length === 0, 'without models/vit16.json: random weights, a banner, trained disabled', N);
const key = 'dl-explorer.vit-weights.v1';
const H = run(join(chk, 'scen_stored.js'), '--page', 'vit-3d', '--three');
ok(H.trained && /training page/i.test(H.banner || '') && /on the ViT training page/.test(H.wnote), `weights handed off by vit-train/ (${key}) load with the banner`, H);

if (!process.argv.includes('--no-frames')) {
  const out = join(chk, 'shots'); rmSync(out, { recursive: true, force: true }); mkdirSync(out);
  const scen = join(chk, '_frames.js'); writeFileSync(scen, readFileSync(join(chk, 'shot3d.js'), 'utf8') + '\n' + readFileSync(join(chk, 'scen_frames.js'), 'utf8'));
  for (const theme of ['light', 'dark']) {
    const F = run(scen, '--page', 'vit-3d', '--three', ...(theme === 'dark' ? ['--dark'] : []));
    ok(!F.error && F.consoleErrors.length === 0, `${theme} frames rendered`, [F.error, F.consoleErrors]);
    for (const [k, v] of Object.entries(F.shots || {})) writeFileSync(join(out, `${theme}-${k}.png`), Buffer.from(v.split(',')[1], 'base64'));
  }
  rmSync(scen); console.log(`frames: ${out} (view them)`);
}
console.log(failed ? `${failed} check(s) FAILED` : 'vit-3d: all checks passed');
process.exit(failed ? 1 : 0);
```

- [ ] **Step 3: Run the checks**

Run: `node .superpowers/check/vit3d_checks.mjs` (about three minutes).
Expected, exactly:
```
PASS  no scenario or console errors
PASS  a live WebGL context
PASS  36 blocks, 3,828 cubes
PASS  12 accent frames on the learned weights
PASS  trained weights and their accuracy beside the prediction
PASS  no horizontal page scroll
PASS  terms linked and listed
PASS  cubes give the probes and readouts vit/ gives for the same cells
PASS  xp probe: 4 green boxes and 0 attention links drawn
PASS  o probe: 115 green boxes and 12 attention links drawn
PASS  z1 probe: 118 green boxes and 12 attention links drawn
PASS  att probe: 32 green boxes and 0 attention links drawn
PASS  a fan after attention only
PASS  the sweep visits 16 patches, each with its patch, kernels, token and connector
PASS  a click on a hovered cube pins its probe
PASS  the pin survives an orbit (a drag is not a click)
PASS  leaving the cubes returns to the pinned probe
PASS  a click on empty space releases it
PASS  every preset, in both layouts, frames its blocks inside the view
PASS  the layout is remembered
PASS  shuffling changes the prediction only with position on
PASS  the weights switch and its wording
PASS  the digit picker and "another"
PASS  no banner with the shipped weights
PASS  without models/vit16.json: random weights, a banner, trained disabled
PASS  weights handed off by vit-train/ (dl-explorer.vit-weights.v1) load with the banner
PASS  light frames rendered
PASS  dark frames rendered
frames: <repo>\.superpowers\check\shots (view them)
vit-3d: all checks passed
```
The counts (115 and 118 boxes, 12 links) belong to the shipped weights and the default digit, a 7 (held-out digit 7).

- [ ] **Step 4: Look at the frames**

View the PNGs in `.superpowers/check/shots/` with the Read tool: 14 per theme, `light-*` and `dark-*`. Check:
- **`side-whole`:** the chain runs left to right: digit, kernels, tokens, position, z₀, LN₁, the Wq·Wk·Wv column with Q·K·V beside it, attention, A·V, Wo, z₁, LN₂, W₁, hidden, W₂, z₂, then the head. Overlapping labels are hidden.
- **`side-attention`:**
  - the attention slab shows a grid of 16 little 4×4 maps;
  - the large orange cubes are each token's strongest weights;
  - the accent frames sit around Wq, Wk and Wv, each with its bias.
- **`hover-z1`:**
  - token 6's column is fully green, and the other tokens are faintly green by attention weight;
  - one green box covers token 6's little map in the attention slab, with lines from it to the V block;
  - the readout names token 9 first (0.61).
- **`hover-att`:** query 5 and key 9 are fully green, every other key is faint, and the readout says "head 1: token 5 gives token 9 the weight 0.8".
- **`pinned-orbited`:** the view has turned, the green of A·V token 6 is still lit, and the readout ends "(pinned: click empty space to release)".
- **`sweep`:** a yellow window over patch 9 of the digit, yellow over the kernels and over token 9's column, and yellow lines from patch to token.
- **The depth frames:** each preset's blocks are in view; the whole chain runs across the screen.
- **Dark:** the same pictures on the dark scene; the frames use the dark accent (`#7fa7ea`).

- [ ] **Step 5: Commit**

```bash
git add vit-3d/index.html
git commit -m "feat(vit-3d): the ViT 3D explorer" -m "The toy ViT as blocks of cubes: tokens on the patch grid with their numbers in depth, each head's attention as 16 little 4x4 maps at their tokens' places, the whole chain with its weights in accent frames. Hover traces vitTrace's green as graded boxes and a fan of links, a click pins it while the camera moves, a patch sweep, presets, side or depth, and the explorer's controls and wording." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Landing page, link from `vit/`, documentation, final checks

**Files:**
- Modify: `index.html`, `vit/index.html` (edit script), `CLAUDE.md`, `HANDOFF.md` (edit script), `project_notes.md` (skill)
- Create: `.superpowers/check/edit_landing.py`, `.superpowers/check/edit_docs.py`, `.superpowers/check/scen_landing.js`

- [ ] **Step 1: The landing page and the link from `vit/`**

Create `.superpowers/check/edit_landing.py`. It adds the "3D explorer" card between the explorer and the training page, as in the CNN section, and renames the first card "2D explorer" to match the CNN's. The chip and the badge read "3 pages". The second lede names the three kinds of page, and the colour key's green line says "ViT pages". `vit/`'s lede gains one sentence linking to `vit-3d/`.

```python
# index.html: the ViT section gets its 3D explorer card (between the explorer and the training page, as for the CNN),
# "3 pages" on its chip and badge, and the ledes say so. vit/index.html: one sentence pointing to the 3D explorer.
import sys, os
root = sys.argv[1]
def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (rel, a[:80])
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)

edit('index.html', [
    ('<a href="#vit">Vision Transformer <span class="badge live">2 pages</span></a>',
     '<a href="#vit">Vision Transformer <span class="badge live">3 pages</span></a>'),
    ('<h2>Vision Transformer (ViT)</h2><span class="badge live">2 pages</span>',
     '<h2>Vision Transformer (ViT)</h2><span class="badge live">3 pages</span>'),
    ('The convolutional network and the Vision Transformer are built, each with an explorer and a training page.',
     'The convolutional network and the Vision Transformer are built, each with a 2D explorer, a 3D explorer and a training page.'),
    ('On the ViT page, after attention, its strength is the attention weight.',
     'On the ViT pages, after attention, its strength is the attention weight.'),
    ('<path class="grid" d="M7.5 2v22M13 2v22M18.5 2v22M2 7.5h22M2 13h22M2 18.5h22"/></svg>Explorer</b>',
     '<path class="grid" d="M7.5 2v22M13 2v22M18.5 2v22M2 7.5h22M2 13h22M2 18.5h22"/></svg>2D explorer</b>'),
    ('the residuals and the average. Trained weights; hover any value to see what it reads.</span><em>Open →</em></a>\n',
     'the residuals and the average. Trained weights; hover any value to see what it reads.</span><em>Open →</em></a>\n'
     '      <a class="card" href="vit-3d/"><b><svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><g class="k-act"><path class="t" d="M3 9L9 4H24L18 9Z"/><path class="s" d="M18 9L24 4V18L18 23Z"/><rect class="f" x="3" y="9" width="15" height="14"/></g><path class="grid" d="M6.75 9v14M10.5 9v14M14.25 9v14M3 12.5h15M3 16h15M3 19.5h15"/><rect x="10.5" y="12.5" width="3.75" height="3.5" fill="var(--hl2)"/></svg>3D explorer</b>\n'
     '        <span>The same network as blocks of cubes: each token sits where its patch was, and each token\'s attention is a 4×4 map at its place. Hover to trace, click to pin while you turn the view.</span><em>Open →</em></a>\n'),
])
edit('vit/index.html', [
    ('the sections zoom into one part at a time.</p>',
     'the sections zoom into one part at a time. The <a href="../vit-3d/">3D explorer</a> shows the same network as blocks, each token where its patch was.</p>'),
])
print('index.html: ViT 3D explorer card, 3 pages; vit/: link to vit-3d/')
```

Run: `python .superpowers/check/edit_landing.py .`
Expected: `index.html: ViT 3D explorer card, 3 pages; vit/: link to vit-3d/`.

Create `.superpowers/check/scen_landing.js`:

```js
// The landing page: the ViT section's cards and badges, and no horizontal scroll.
R.cards = [...document.querySelectorAll('#vit .card b')].map((b) => b.textContent);
R.badges = [...document.querySelectorAll('a[href="#vit"] .badge, #vit .badge')].map((b) => b.textContent);
R.lede = document.querySelectorAll('.lede')[1].textContent;
R.overflow = document.documentElement.scrollWidth - innerWidth;
```

Run: `node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_landing.js --page . --width 1300 --height 1500 --shot .superpowers/check/landing.png`
Expected:
- `cards` is `["2D explorer", "3D explorer", "Training"]`;
- `badges` is `["3 pages", "3 pages"]`;
- `lede` is "The convolutional network and the Vision Transformer are built, each with a 2D explorer, a 3D explorer and a training page. Next: U-Net, CoAtNet and ConvNeXt.";
- `overflow` is 0, with no console errors.

View `landing.png`: the ViT section shows three cards in a row, and the 3D card's icon is a cube whose front face is a 4×4 grid with one green cell.

- [ ] **Step 2: `CLAUDE.md` and `HANDOFF.md`**

Create `.superpowers/check/edit_docs.py`:

```python
# CLAUDE.md and HANDOFF.md record the ViT 3D explorer. Each original occurs exactly once.
import sys, os
root = sys.argv[1]
def edit(rel, pairs):
    p = os.path.join(root, rel); s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (rel, a[:80])
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)

edit('CLAUDE.md', [
    ('(CNN and ViT built, each with an explorer and a training page;',
     '(CNN and ViT built, each with a 2D explorer, a 3D explorer and a training page;'),
    ('`node core/race.test.mjs`. `core/race.js` steps the CNN and the ViT on the same digits, for both the offline comparison and the live race.',
     '`node core/race.test.mjs`, `node core/vit3d.test.mjs`. `core/race.js` steps the CNN and the ViT on the same digits, for both the offline comparison and the live race. `core/vit3d.js` lays the ViT out as blocks for `vit-3d/`: which tensor each block draws (rows × columns × slices), and the two-way mapping between a cube and a tensor cell.'),
    ('(the ViT pages\' tensor drawing: views, lines of one or several rows, arcs, paint); `blocks3d.js` (the 3D block renderer: scene, instanced cubes, highlights, labels, orbit camera, picking).',
     '(the ViT pages\' tensor drawing: views, lines of one or several rows, arcs, paint); `vitwords.js` (what the two ViT explorers say and load: the probe readout, the wording about the weights, loading the digits and the weights); `blocks3d.js` (the 3D block renderer: scene, instanced cubes, highlights, labels, orbit camera, picking; for `vit-3d/` also graded green boxes and lines, a camera fit to a set of blocks, and a click).'),
    ('- `cnn-2d/`, `cnn-3d/`, `cnn-train/`, `vit/`, `vit-train/` — one self-contained page each',
     '- `cnn-2d/`, `cnn-3d/`, `cnn-train/`, `vit/`, `vit-3d/`, `vit-train/` — one self-contained page each'),
    ('and every copy of a tensor lights together on hover.',
     'and every copy of a tensor lights together on hover. `vit-3d/` draws the same ViT as blocks: each token at its patch\'s place with its numbers in depth, each head\'s attention as a slab of 16 little 4×4 maps (token i\'s map at token i\'s place). Its green is `vitTrace`\'s, as graded boxes plus a fan of links to the tokens read, and a click pins it while the camera moves.'),
    ('After attention (ViT page) green is graded:', 'After attention (ViT explorers) green is graded:'),
    ('The ViT page runs on weights trained offline (`models/vit16.json`), can switch to random ones, and says which,',
     'The ViT explorers (`vit/`, `vit-3d/`) run on weights trained offline (`models/vit16.json`), can switch to random ones, and say which,'),
    ('used by the 3D explorer and the training page\'s 3D view.', 'used by the two 3D explorers and the CNN training page\'s 3D view.'),
])
edit('HANDOFF.md', [
    ('many steps, and the CNN-against-ViT comparison._',
     'many steps, and the CNN-against-ViT comparison; a 3D explorer, `vit-3d/`, followed on 2026-11-16, with each token at its patch\'s place and each token\'s attention as a 4×4 map at its place._'),
    ('    node core/race.test.mjs\n', '    node core/race.test.mjs\n    node core/vit3d.test.mjs\n'),
])
print('CLAUDE.md, HANDOFF.md: the ViT 3D explorer recorded')
```

Run: `python .superpowers/check/edit_docs.py .`
Expected: `CLAUDE.md, HANDOFF.md: the ViT 3D explorer recorded`.

- [ ] **Step 3: `project_notes.md`, with the `project-notes` skill**

Invoke the `project-notes` skill and update "As built" only. Never touch Intent.
- **Header:** last updated 2026-11-16, the ViT 3D explorer built.
- **Overview:** six pages now; the ViT has a 2D explorer, a 3D explorer and a training page.
- **Method:** a `vit-3d/` bullet:
  - the geometry: tokens on the patch grid, attention as little maps at their tokens' places, weights in frames;
  - hover, the fan, the pin, the sweep, the presets and the layouts;
  - `core/vit3d.js`, `ui/vitwords.js`, and the `ui/blocks3d.js` additions.
- **Results**, dated 2026-11-16, laptop, Node 24.19.0, headless Edge with SwiftShader:
  - Task 1's test lines;
  - `vit/` unchanged through the move to `ui/vitwords.js`;
  - both CNN 3D scenes unchanged through the renderer additions;
  - the 28 page checks of Task 4;
  - the frames viewed in light and dark.
- **Decisions**, with their why, from the spec:
  - a separate page first, with the training page's 3D switch later;
  - geometry C;
  - the whole chain;
  - hover, pin and sweep.

  Add the deviations listed in this plan's self-review.
- **Map:** the new files, the spec and this plan.
- **Status:** the ViT 3D explorer is built; next, the 2D | 3D switch on `vit-train/`, with its own spec, then U-Net.

- [ ] **Step 4: Final checks**

Run:
```bash
node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs && node core/vit.test.mjs && node core/race.test.mjs && node core/vit3d.test.mjs
node .superpowers/check/vit3d_checks.mjs --no-frames
node .superpowers/check/unchanged.mjs compare cnn3d
```
Expected:
- all six test files pass;
- `vit-3d: all checks passed` (26 PASS lines: those of Task 4 without the two frame lines);
- `cnn3d: unchanged`.

Do not run `compare vit` now: `vit/` has changed on purpose, by its link to `vit-3d/`.

- [ ] **Step 5: Commit**

```bash
git add index.html vit/index.html
git commit -m "feat: landing card and a link from vit/ for the ViT 3D explorer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add CLAUDE.md HANDOFF.md project_notes.md
git commit -m "docs: record the ViT 3D explorer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git status --short
```
Expected: a clean working tree. Do not push. After the final review the controller fast-forwards `main` to `feat/vit-3d` and pushes `main` (the spec commit `ffad883` goes with it).

---

## Plan self-review

- **Spec coverage:**
  - Part 1 (the scene, the frames, the layouts, the presets): Task 1 for the blocks, Task 4 for the page.
  - Part 2 (hover, the fan, the pin, the sweep, the controls, the weights and their wording): Tasks 2 and 4.
  - Part 3 (`core/vit3d.js`, `ui/vitwords.js`, the `ui/blocks3d.js` additions, the page, the landing card): Tasks 1 to 5.
  - Part 4 (the checks): the Node test in Task 1; `vit/` unchanged in Task 2; the CNN scenes unchanged in Task 3; Task 4's 28 checks with frames in light and dark.
- **Deviations from the spec, by design:**
  - `ui/blocks3d.js` gains five calls, not two. `strengthLine` draws the fan, `snap` draws a frame on demand for the checks, and `onClick` makes the pin. All are opt-in, and the CNN digests prove it.
  - The depth layout's whole view looks from the side (theta 1.25), not from behind. Viewed from behind, the chain of 36 blocks recedes into a thin diagonal, which the prototype's frames showed.
  - The landing page's first ViT card is renamed "2D explorer", as in the CNN section, and `vit/`'s lede gains one sentence linking to `vit-3d/`. Neither is in the spec; both make the new page findable.
- **Placeholders:** none. Every file is given in full, and every command has its expected output.
- **Names are consistent across tasks:**
  - `VIT_BLOCKS`, `cubeToCell`, `cellToCube`, `sliceValues`, `traceBoxes`, `attentionFan`;
  - `describe`, `weightWords`, `loadDigitsAndWeights`;
  - `strengthBox`, `strengthLine`, `fit`, `snap`, `onClick`;
  - the scenario files and `unchanged.mjs`'s sets `vit` and `cnn3d`.
