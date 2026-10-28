# ViT Training Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A page, `vit-train/`, that shows the toy ViT learning:
- one step in full, with the backward pass through attention revealed a stage at a time;
- many steps, with attention and position snapshots over training;
- a CNN-against-ViT comparison: a chart measured offline over 5 seeds, plus a live race.

**Architecture:**
- **`core/vit.js` gains:**
  - the gradient of every drawn activation (`back`);
  - a `nudge` hook, so the tests can check each of them by finite differences;
  - the backward green (`gradTrace`), and the stage map (`BACK_STAGES`).
- **`core/race.js` holds one racer for both models.** The offline script and the live race step the same racer, so a live run at a measured seed reproduces its dot exactly.
- **The explorer's tensor drawing moves to `ui/vitline.js` and `ui/vitline.css`.** The move is checked pixel-identical on `vit/`. The new page draws its three rows with the same module.

Spec: `docs/superpowers/specs/2026-10-28-vit-training-page-design.md`.

**Tech Stack:** Plain HTML + ES modules (no build step), Node 24 for the tests and the measurement, Python 3 for edit scripts and `http.server`, headless Edge for the page checks.

## Global Constraints

- No build step; no external scripts (Google Fonts stylesheet only). `core/` stays DOM-free; `ui/` holds shared DOM helpers.
- **Model:** the toy ViT of `core/vit.js`, unchanged in behaviour. Every existing test still passes, and `vit/` keeps its scenario values and screenshots through the refactor (Task 3).
- **Training on the page:** SGD only, with the update row w′ = w − lr·∂L/∂w. Learning rate on a log slider from 0.0003 to 0.3, default 0.003. Batch 1, 8 or 32. Speed 2, 10, 50, 200 or 500 digits a second.
- **The comparison:** sizes 100, 300, 1,000 and 3,000 digits (balanced: the first N/10 of each class); 30,000 SGD steps per run; CNN lr 0.1, ViT lr 0.003; seeds 1 to 5 offline; the live race's seeds from 6 up.
- **Colour grammar:**
  - green = computed from: forward by a row of attention, backward by a column;
  - a green frame = what the last backward step computed;
  - faint = not computed yet;
  - weights tinted where they act;
  - chart series: CNN `#e5641e`; ViT `#2457a8` in light and `#5f8fe0` in dark (validated for colour vision deficiency and contrast).
- **No tables on any page** (project rule). Numbers go in hover tooltips and in generated text.
- Wherever a prediction or accuracy appears, the page says what it was measured on. NZ English in all copy.
- **Files and commits:**
  - Files are LF; ignore Git's CRLF warnings.
  - **Never write files with shell heredocs**: this environment drops backslashes. Use the Write tool, then run the scripts.
  - Check scripts, edit scripts and their outputs go in `.superpowers/check/` (git-ignored).
  - Conventional Commits, one or two local commits per task, on the branch `feat/vit-train`. The merge into `main` and the push happen after the final review.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `core/vit.js` | `nudge` in `forward`; `back` from `lossAndGrad`; `ACTIVATIONS`, `WEIGHT_KEYS`, `BACK_STAGES`, `gradTrace` | 1 |
| `core/vit.test.mjs` | each `back` tensor by finite differences; `gradTrace` rules | 1 |
| `core/race.js`, `core/race.test.mjs` | balanced subsets; the racer shared by the measurement and the live race | 2 |
| `tools/vit_vs_cnn.mjs`, `models/vit_vs_cnn.json` | the 40 offline runs and their results | 2 |
| `ui/vitline.js`, `ui/vitline.css` | views, lines (one or several rows), arcs, bars, paint | 3 |
| `vit/index.html` | uses `ui/vitline.*`; loads weights handed off from `vit-train/`; links to it | 3, 4 |
| `ui/trained.js` | the ViT hand-off (`dl-explorer.vit-weights.v1`) | 4 |
| `ui/terms.js` | `VIT_TRAIN_TERMS`; "classifier head" is not an attention head | 4 |
| `vit-train/index.html` | the page | 5 |
| `index.html`, `CLAUDE.md`, `HANDOFF.md`, `project_notes.md` | card, badge, records | 6 |
| `.superpowers/check/*` | edit scripts, the page harness, scenarios (not committed) | 1–6 |

**Context for the executor.** Every file in this plan was prototyped and run on 2026-10-28, in a scratch worktree of the repo, before the plan was written. The code and the expected outputs below are from that run. The edit scripts were checked to reproduce the prototype exactly from the repo's files.

---

### Task 1: Activation gradients, the nudge hook, the backward green

**Files:**
- Modify: `core/vit.js` (an edit script, plus a block appended at the end)
- Modify: `core/vit.test.mjs` (an edit script inserts a block)

**Interfaces:**
- Produces, in `core/vit.js`:
  - `forward(w, img, { position, order, nudge })`: `nudge` is `{ t, r, c, by }`, and adds `by` to one activation cell as soon as it is computed. `t` is a SHAPES activation key or `'logits'`.
  - `lossAndGrad(...)` returns `{ S, loss, grads, back }`:
    - `back` is keyed as SHAPES activations: `prob` (= ∂L/∂scores, p − y), `avgNorm`, `avg`, `z2`, `hid`, `h2`, `z1`, `o`, `att0`, `att1`, `q`, `k`, `v`, `h1`, `z0`, `xp`;
    - each has the activation's shape; `avg`, `avgNorm` and `prob` are vectors.
  - `ACTIVATIONS` (17 keys, in flow order) and `WEIGHT_KEYS` (the other SHAPES keys).
  - `BACK_STAGES`: 4 stages `{ name, act, w }`, named `head`, `MLP`, `attention`, `embedding`.
  - `gradTrace(S, probe)` returns `{ key: Float32Array }`. Keys are namespaced: `'d'+key` for a gradient, a plain key for forward, `'n'+key` for w′.
    - A probe `'d'+activation` returns its downstream sources, graded through attention by a column.
    - A weight probe (plain, or with `'d'` or `'n'`) returns the weight in all three rows, its input column (forward) and its output-gradient column.

- [ ] **Step 1: Write the test additions and the script that inserts them**

Create `.superpowers/check/back_tests.js`:

```js
// ---- back: the gradient of every drawn activation, against central finite differences (nudging that activation) ----
{
  const { back } = lossAndGrad(w, img, label);
  const at = (M, r, c) => (Array.isArray(M[0]) ? M[r][c] : M[c]);
  let worst = 0, n = 0;
  for (const key of ACTIVATIONS.filter((key) => key !== 'digit')) {
    const [R, C] = SHAPES[key], M = back[key];
    assert(Array.isArray(M) && (R === 1 ? M.length === C : M.length === R && M[0].length === C), `back.${key} is ${R}×${C}`);
    const t = key === 'prob' ? 'logits' : key; // ∂L/∂(scores) is drawn under the probabilities
    for (const [r, c] of [[0, 0], [R - 1, C - 1], [R >> 1, C >> 1]]) {
      const h = 1e-5, lossAt = (by) => crossEntropy(forward(w, img, { nudge: { t, r, c, by } }).prob, label);
      const num = (lossAt(h) - lossAt(-h)) / (2 * h), ana = at(M, r, c), err = Math.abs(num - ana) / (1 + Math.abs(ana));
      assert(err < 1e-5, `back.${key}[${r}][${c}]: numeric ${num} vs analytic ${ana}`);
      worst = Math.max(worst, err); n++;
    }
  }
  const unions = (pick) => BACK_STAGES.flatMap(pick).sort().join();
  assert(unions((s) => s.act) === ACTIVATIONS.filter((key) => key !== 'digit').sort().join(), 'BACK_STAGES covers every activation but the digit');
  assert(unions((s) => s.w) === [...WEIGHT_KEYS].sort().join(), 'BACK_STAGES covers every weight');
  console.log(`back: ${n} cells of ${ACTIVATIONS.length - 1} activation gradients against finite differences, max normalised error ${worst.toExponential(2)}`);
}

// ---- gradTrace: the backward green ----
{
  const at = (tr, key, r, c) => (tr[key] ? tr[key][r * (SHAPES[key] || SHAPES[key.slice(1)])[1] + c] : 0);
  // a value's gradient collects from the queries that read it: column 4 of head 1's map, graded
  const tv = gradTrace(S, { t: 'dv', r: 4, c: 1 }), A = S.att[0];
  let m = 0; A.forEach((row, i) => { if (i !== 4 && row[4] > m) m = row[4]; });
  for (let i = 0; i < T; i++) {
    const want = i === 4 ? 1 : 0.9 * A[i][4] / m;
    close(at(tv, 'do', i, 1), want, 1e-6, `dv probe: query ${i} strength`);
    close(at(tv, 'att0', i, 4), want, 1e-6, `dv probe: map column 4, row ${i}`);
    assert(at(tv, 'do', i, 5) === 0, 'dv probe: the other head is not lit');
  }
  close(Math.max(...[...Array(T).keys()].filter((i) => i !== 4).map((i) => at(tv, 'do', i, 1))), Math.fround(0.9), 1e-7, 'dv probe: most-reading query at 0.9');
  assert(at(tv, 'dprob', 0, 3) > 0 && !tv.dq && !tv.dh1, 'dv probe: up to the loss, nothing upstream');
  // a query's gradient comes from its own row of ∂L/∂A
  const tq = gradTrace(S, { t: 'dq', r: 3, c: 6 });
  assert([...Array(T).keys()].every((j) => at(tq, 'datt1', 3, j) === 1 && at(tq, 'datt1', 2, j) === 0) && !tq.datt0, 'dq probe: its row of ∂L/∂A, its head');
  // before attention's backward step, per token: z₂ reads the average's same column, and the head
  const tz = gradTrace(S, { t: 'dz2', r: 2, c: 5 });
  assert(at(tz, 'davg', 0, 5) === 1 && at(tz, 'davg', 0, 4) === 0 && at(tz, 'dprob', 0, 9) === 1 && !tz.dz1 && !tz.dhid, 'dz2 probe');
  const th = gradTrace(S, { t: 'dhid', r: 7, c: 30 });
  assert(at(th, 'dz2', 7, 0) === 1 && at(th, 'dz2', 6, 0) === 0 && at(th, 'W2', 30, 3) === 1 && at(th, 'W2', 29, 3) === 0, 'dhid probe');
  // a weight: all three rows, the input column it reads and the gradient column it feeds
  const tw = gradTrace(S, { t: 'dWo', r: 2, c: 5 });
  assert(['Wo', 'dWo', 'nWo'].every((key) => at(tw, key, 2, 5) === 1 && at(tw, key, 2, 4) === 0), 'weight probe: three rows');
  assert([...Array(T).keys()].every((i) => at(tw, 'o', i, 2) === 1 && at(tw, 'o', i, 3) === 0 && at(tw, 'dz1', i, 5) === 1 && at(tw, 'dz1', i, 4) === 0), 'weight probe: input and gradient columns');
  const tp = gradTrace(S, { t: 'nWp', r: 6, c: 5 });
  assert(at(tp, 'digit', 1, 1) === 1 && at(tp, 'digit', 5, 13) === 1 && at(tp, 'digit', 0, 0) === 0 && at(tp, 'dxp', 9, 6) === 1, 'patch-kernel probe');
  assert(Object.keys(gradTrace(S, { t: 'z1', r: 0, c: 0 })).length === 0 && Object.keys(gradTrace(S, null)).length === 0, 'forward activations and no probe: nothing');
  console.log('gradTrace: value, query, z2, hidden and weight probes ok');
}
```

Create `.superpowers/check/edit_test.py`:

```python
# core/vit.test.mjs: import the new exports and add the back and gradTrace checks (from back_tests.js) before the last line.
import sys
p, blk = sys.argv[1], open(sys.argv[2], encoding='utf-8').read()
s = open(p, encoding='utf-8').read()
imp = s.splitlines()[3]
assert imp.startswith('import {') and imp.endswith("vitTrace } from './vit.js';"), imp
s = s.replace(imp, imp.replace('vitTrace }', 'vitTrace, gradTrace, ACTIVATIONS, WEIGHT_KEYS, BACK_STAGES }'))
last = "console.log('core/vit.test.mjs: all checks passed');"
assert s.count(last) == 1
s = s.replace(last, blk.rstrip('\n') + '\n' + last)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('core/vit.test.mjs edited')
```

Run: `python .superpowers/check/edit_test.py core/vit.test.mjs .superpowers/check/back_tests.js`
Expected: `core/vit.test.mjs edited`.

- [ ] **Step 2: Run the tests to see them fail**

Run: `node core/vit.test.mjs`
Expected: FAIL, `SyntaxError: The requested module './vit.js' does not provide an export named 'gradTrace'` (or `ACTIVATIONS`).

- [ ] **Step 3: Edit the forward and backward passes**

Create `.superpowers/check/edit_core.py`:

```python
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
reps = [
# forward: the nudge hook (used by the tests to check each activation's gradient by finite differences)
("""/** Every intermediate of one forward pass. position false leaves the position table out; order shuffles the patches. */
export function forward(w, img, { position = true, order = null } = {}) {
  const patches = patchify(img, order);
  const xp = patches.map((px) => w.Wp.map((k, d) => dot(px, k) + w.bp[d]));
  const z0 = xp.map((t, i) => t.map((v, d) => v + (position ? w.pos[i][d] : 0)));
  const n1 = z0.map((t) => layerNorm(t, w.ln1)), h1 = n1.map((n) => n.y);
  const q = linear(h1, w.Wq, w.bq), k = linear(h1, w.Wk, w.bk), v = linear(h1, w.Wv, w.bv);""",
"""/** Every intermediate of one forward pass. position false leaves the position table out; order shuffles the patches.
 *  nudge { t, r, c, by } adds `by` to one cell of an activation as soon as it is computed (t a SHAPES activation key,
 *  or 'logits'); the tests use it to check each activation's gradient against finite differences. */
export function forward(w, img, { position = true, order = null, nudge = null } = {}) {
  const bump = (key, M) => { if (nudge && nudge.t === key) { if (Array.isArray(M[0])) M[nudge.r][nudge.c] += nudge.by; else M[nudge.c] += nudge.by; } return M; };
  const patches = patchify(img, order);
  const xp = bump('xp', patches.map((px) => w.Wp.map((k, d) => dot(px, k) + w.bp[d])));
  const z0 = bump('z0', xp.map((t, i) => t.map((v, d) => v + (position ? w.pos[i][d] : 0))));
  const n1 = z0.map((t) => layerNorm(t, w.ln1)), h1 = bump('h1', n1.map((n) => n.y));
  const q = bump('q', linear(h1, w.Wq, w.bq)), k = bump('k', linear(h1, w.Wk, w.bk)), v = bump('v', linear(h1, w.Wv, w.bv));"""),
("    const a = s.map(softmax);\n", "    const a = bump('att' + h, s.map(softmax));\n"),
("""  const o = q.map((_, i) => heads.flatMap((hh) => hh[i]));
  const mixed = linear(o, w.Wo, w.bo);
  const z1 = z0.map((t, i) => t.map((x, d) => x + mixed[i][d]));
  const n2 = z1.map((t) => layerNorm(t, w.ln2)), h2 = n2.map((n) => n.y);
  const pre = linear(h2, w.W1, w.b1), hid = pre.map((r) => r.map(gelu));
  const mlp = linear(hid, w.W2, w.b2);
  const z2 = z1.map((t, i) => t.map((x, d) => x + mlp[i][d]));
  const avg = Array.from({ length: D }, (_, d) => z2.reduce((t, r) => t + r[d], 0) / T);
  const nf = layerNorm(avg, w.lnf), avgNorm = nf.y;
  const logits = w.bh.map((b, c) => avgNorm.reduce((t, x, d) => t + x * w.Wh[d][c], b));""",
"""  const o = bump('o', q.map((_, i) => heads.flatMap((hh) => hh[i])));
  const mixed = linear(o, w.Wo, w.bo);
  const z1 = bump('z1', z0.map((t, i) => t.map((x, d) => x + mixed[i][d])));
  const n2 = z1.map((t) => layerNorm(t, w.ln2)), h2 = bump('h2', n2.map((n) => n.y));
  const pre = linear(h2, w.W1, w.b1), hid = bump('hid', pre.map((r) => r.map(gelu)));
  const mlp = linear(hid, w.W2, w.b2);
  const z2 = bump('z2', z1.map((t, i) => t.map((x, d) => x + mlp[i][d])));
  const avg = bump('avg', Array.from({ length: D }, (_, d) => z2.reduce((t, r) => t + r[d], 0) / T));
  const nf = layerNorm(avg, w.lnf), avgNorm = bump('avgNorm', nf.y);
  const logits = bump('logits', w.bh.map((b, c) => avgNorm.reduce((t, x, d) => t + x * w.Wh[d][c], b)));"""),
# lossAndGrad: keep the activation gradients for drawing
("/** Loss and the gradient of every weight (same structure as the weights) for one digit. */",
 "/** Loss and the gradient of every weight (same structure as the weights) for one digit, and `back`: the gradient of\n *  every drawn activation, keyed as SHAPES (`prob` holds ∂L/∂(scores) = p − y, drawn under the probabilities). */"),
("""  const dQ = zeros(T, D), dK = zeros(T, D), dV = zeros(T, D), sc = 1 / Math.sqrt(DH);
  for (let h = 0; h < H; h++) {
    const A = S.att[h], off = h * DH;
    for (let i = 0; i < T; i++) {
      const dA = new Array(T);""",
"""  const dQ = zeros(T, D), dK = zeros(T, D), dV = zeros(T, D), sc = 1 / Math.sqrt(DH), dAtt = [];
  for (let h = 0; h < H; h++) {
    const A = S.att[h], off = h * DH;
    dAtt.push([]);
    for (let i = 0; i < T; i++) {
      const dA = new Array(T); dAtt[h].push(dA);"""),
("""  const dHq = linearBack(S.h1, w.Wq, dQ, g.Wq, g.bq), dHk = linearBack(S.h1, w.Wk, dK, g.Wk, g.bk), dHv = linearBack(S.h1, w.Wv, dV, g.Wv, g.bv);
  const dZ0 = dZ1.map((r, i) => {
    const dx = lnBack(dHq[i].map((x, d) => x + dHk[i][d] + dHv[i][d]), S.n1[i], w.ln1.g, g.ln1.g, g.ln1.b);
    return r.map((x, d) => x + dx[d]);
  });""",
"""  const dHq = linearBack(S.h1, w.Wq, dQ, g.Wq, g.bq), dHk = linearBack(S.h1, w.Wk, dK, g.Wk, g.bk), dHv = linearBack(S.h1, w.Wv, dV, g.Wv, g.bv);
  const dH1 = dHq.map((r, i) => r.map((x, d) => x + dHk[i][d] + dHv[i][d]));
  const dZ0 = dZ1.map((r, i) => { const dx = lnBack(dH1[i], S.n1[i], w.ln1.g, g.ln1.g, g.ln1.b); return r.map((x, d) => x + dx[d]); });"""),
("""  return { S, loss, grads: g };
}""",
"""  const back = {
    prob: dLogits, avgNorm: dAvgNorm, avg: dAvg, z2: dZ2, hid: dHid, h2: dH2, z1: dZ1, o: dO, att0: dAtt[0], att1: dAtt[1],
    q: dQ, k: dK, v: dV, h1: dH1, z0: dZ0, xp: dZ0, // z₀ = patch tokens + position, so both get the same gradient
  };
  return { S, loss, grads: g, back };
}"""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('core edited')
```

Run: `python .superpowers/check/edit_core.py core/vit.js`
Expected: `core edited`.

- [ ] **Step 4: Append the stages and the backward green**

Create `.superpowers/check/gradtrace.js` with exactly this content (it starts with two blank lines):

```js


/** The drawn activations, in the order the data flows; every other SHAPES key is a weight. */
export const ACTIVATIONS = ['digit', 'xp', 'z0', 'h1', 'q', 'k', 'v', 'att0', 'att1', 'o', 'z1', 'h2', 'hid', 'z2', 'avg', 'avgNorm', 'prob'];
export const WEIGHT_KEYS = Object.keys(SHAPES).filter((key) => !ACTIVATIONS.includes(key));

/** The backward pass in the order the training page reveals it, from the loss towards the input: for each stage, the
 *  activations whose gradient it computes and the weights whose gradient (and so w′) it computes. */
export const BACK_STAGES = [
  { name: 'head', act: ['prob', 'avgNorm', 'avg', 'z2'], w: ['Wh', 'bh', 'lnf'] },
  { name: 'MLP', act: ['hid', 'h2', 'z1'], w: ['W2', 'b2', 'W1', 'b1', 'ln2'] },
  { name: 'attention', act: ['o', 'att0', 'att1', 'q', 'k', 'v', 'h1'], w: ['Wo', 'bo', 'Wq', 'bq', 'Wk', 'bk', 'Wv', 'bv'] },
  { name: 'embedding', act: ['z0', 'xp'], w: ['ln1', 'pos', 'Wp', 'bp'] },
];

const AFTER = ['o', 'z1', 'h2', 'hid', 'z2']; // per-token activations after attention, in the order the data flows
const WEIGHT_IO = { Wq: ['h1', 'q'], Wk: ['h1', 'k'], Wv: ['h1', 'v'], Wo: ['o', 'z1'], W1: ['h2', 'hid'], W2: ['hid', 'z2'], Wh: ['avgNorm', 'prob'] };
const BIAS_OUT = { bp: 'xp', bq: 'q', bk: 'k', bv: 'v', bo: 'z1', b1: 'hid', b2: 'z2', bh: 'prob' };
const LN_OUT = { ln1: 'h1', ln2: 'h2', lnf: 'avgNorm' };

/**
 * The backward green. For a hovered gradient probe { t: 'd' + activation key, r, c }: the cells it is computed from,
 * downstream up to the loss, with a strength from 0 to 1, as { key: Float32Array(rows × cols) }. Keys are namespaced:
 * 'd' + key for a gradient, a plain key for a forward activation or weight, 'n' + key for an updated weight w′.
 * Through attention the grading turns over: the gradient of token j's key or value collects from every query i that
 * read it, so query i gets 1 for i = j and otherwise 0.9 · A[i][j] / (the largest A[k][j], k ≠ j), a column of the map
 * where the forward green uses a row; values after the heads are joined use the mean of the two maps. For a weight,
 * plain or with 'd' or 'n' { t, r, c }: the weight in all three rows, the input column it reads in the forward row and
 * the gradient column it feeds in the backward row, whose products summed over the tokens are its gradient.
 */
export function gradTrace(S, probe) {
  const out = {};
  if (!probe) return out;
  const shape = (key) => SHAPES[key] || SHAPES[key.slice(1)];
  const put = (key, r, c, s) => {
    const [R, C] = shape(key), a = out[key] || (out[key] = new Float32Array(R * C));
    if (s > a[r * C + c]) a[r * C + c] = s;
  };
  const row = (key, r, s, c0 = 0, c1 = shape(key)[1] - 1) => { for (let c = c0; c <= c1; c++) put(key, r, c, s); };
  const col = (key, c, s) => { for (let r = 0; r < shape(key)[0]; r++) put(key, r, c, s); };
  const head = (s) => { row('davg', 0, s); row('davgNorm', 0, s); row('dprob', 0, s); };
  const tail = (i, from, s) => { for (const key of AFTER.slice(AFTER.indexOf(from) + 1)) row('d' + key, i, s); head(s); };
  const readers = (j, A) => {
    let m = 0; A.forEach((r, i) => { if (i !== j && r[j] > m) m = r[j]; });
    return A.map((r, i) => (i === j ? 1 : m > 0 ? 0.9 * r[j] / m : 0));
  };
  const meanA = S.att[0].map((r, a) => r.map((x, b) => S.att.reduce((t, M) => t + M[a][b], 0) / H));
  // what the gradient of token i's LayerNorm-1 output comes from: its own query, key and value gradients, the queries
  // that read it as a key or value (graded by the mean map's column i), and their tails up to the loss
  const fromH1 = (i) => {
    row('dq', i, 1); row('dk', i, 1); row('dv', i, 1);
    readers(i, meanA).forEach((s, i2) => {
      for (let h = 0; h < H; h++) put('datt' + h, i2, i, s);
      row('do', i2, s); tail(i2, 'o', s);
    });
  };
  let { t, r, c } = probe;
  const base = SHAPES[t] ? t : t.slice(1);
  if (WEIGHT_KEYS.includes(base)) {
    for (const key of [base, 'd' + base, 'n' + base]) put(key, r, c, 1);
    if (base === 'Wp') { for (let j = 0; j < T; j++) put('digit', (j >> 2) * P + (c >> 2), (j & 3) * P + (c & 3), 1); col('dxp', r, 1); }
    else if (base === 'pos') put('dz0', r, c, 1);
    else if (WEIGHT_IO[base]) { col(WEIGHT_IO[base][0], r, 1); col('d' + WEIGHT_IO[base][1], c, 1); }
    else if (BIAS_OUT[base]) col('d' + BIAS_OUT[base], c, 1);
    else if (LN_OUT[base]) { col(LN_OUT[base], c, 1); col('d' + LN_OUT[base], c, 1); }
    return out;
  }
  if (t[0] !== 'd' || !ACTIVATIONS.includes(base)) return out;
  put(t, r, c, 1);
  const hd = Math.floor(c / DH), c0 = hd * DH, c1 = c0 + DH - 1;
  switch (base) {
    case 'prob': put('prob', 0, c, 1); break; // p − y: this class's probability, and the label
    case 'avgNorm': row('dprob', 0, 1); row('Wh', c, 1); break;
    case 'avg': row('davgNorm', 0, 1); row('dprob', 0, 1); row('lnf', 0, 1); break;
    case 'z2': put('davg', 0, c, 1); row('davgNorm', 0, 1); row('dprob', 0, 1); break;
    case 'hid': row('dz2', r, 1); row('W2', c, 1); head(1); break;
    case 'h2': row('dhid', r, 1); row('W1', c, 1); tail(r, 'hid', 1); break;
    case 'z1': put('dz2', r, c, 1); row('dh2', r, 1); tail(r, 'z1', 1); break;
    case 'o': row('dz1', r, 1); row('Wo', c, 1); tail(r, 'z1', 1); break;
    case 'att0': case 'att1': {
      const h = +base[3];
      row('do', r, 1, h * DH, h * DH + DH - 1); row('v', c, 1, h * DH, h * DH + DH - 1); tail(r, 'o', 1); break;
    }
    case 'v': readers(r, S.att[hd]).forEach((s, i) => { put('do', i, c, s); put('att' + hd, i, r, s); tail(i, 'o', s); }); break;
    case 'k': readers(r, S.att[hd]).forEach((s, i) => { put('datt' + hd, i, r, s); row('q', i, s, c0, c1); row('do', i, s, c0, c1); tail(i, 'o', s); }); break;
    case 'q': {
      row('datt' + hd, r, 1); row('do', r, 1, c0, c1); tail(r, 'o', 1);
      const A = S.att[hd]; let m = 0; A[r].forEach((a, j) => { if (j !== r && a > m) m = a; });
      A[r].forEach((a, j) => row('k', j, j === r ? 1 : m > 0 ? 0.9 * a / m : 0, c0, c1)); break;
    }
    case 'h1': fromH1(r); row('Wq', c, 1); row('Wk', c, 1); row('Wv', c, 1); break;
    case 'z0': put('dz1', r, c, 1); row('dh1', r, 1); fromH1(r); tail(r, 'z1', 1); break;
    case 'xp': put('dz0', r, c, 1); row('dh1', r, 1); fromH1(r); tail(r, 'z1', 1); put('dz1', r, c, 1); break;
    default: break;
  }
  return out;
}
```

Run: `node -e "const fs=require('fs');fs.appendFileSync('core/vit.js',fs.readFileSync('.superpowers/check/gradtrace.js','utf8'))"`

- [ ] **Step 5: Run the tests**

Run: `node core/vit.test.mjs`
Expected (about 3 s), ending:
```text
vitTrace: before attention, one head, both heads, attention weight, head, weights ok
back: 48 cells of 16 activation gradients against finite differences, max normalised error 8.22e-10
gradTrace: value, query, z2, hidden and weight probes ok
core/vit.test.mjs: all checks passed
```
Then run `node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs`: all pass.

- [ ] **Step 6: Commit**

```bash
git add core/vit.js core/vit.test.mjs
git commit -m "feat(core): activation gradients, the nudge check and the backward green" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The racer and the CNN-against-ViT measurement

**Files:**
- Create: `core/race.js`, `core/race.test.mjs`, `tools/vit_vs_cnn.mjs`
- Create: `models/vit_vs_cnn.json` (generated, about 430 bytes)

**Interfaces:**
- Consumes: `initWeights`, `lossAndGrad`, `sgdStep`, `accuracy`, `shuffledOrder` from `core/train.js`; `init`, `lossAndGrad`, `sgdStep`, `accuracy` from `core/vit.js`; `image` from `core/data.js`.
- Produces:
  - `RACE` = `{ sizes: [100, 300, 1000, 3000], steps: 30000, lr: { cnn: 0.1, vit: 0.003 } }`;
  - `balancedSubset(set, n)` returns an array of digit indices;
  - `racer(kind, set, idx, seed)` returns `{ kind, seed, steps, weights, step(), accuracy(testSet, testIdx) }`;
  - `models/vit_vs_cnn.json` = `{ version: 1, sizes, seeds: [1..5], steps, lr, cnn: [[acc per seed] per size], vit: [...], made, date }`.

- [ ] **Step 1: Write the test**

Create `core/race.test.mjs`:

```js
import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { initWeights, lossAndGrad, sgdStep, shuffledOrder } from './train.js';
import { RACE, balancedSubset, racer } from './race.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));

// a balanced subset: n/10 of each class, the first ones in file order, no repeats
for (const n of RACE.sizes) {
  const idx = balancedSubset(data.train, n), counts = new Array(10).fill(0);
  idx.forEach((i) => counts[data.train.labels[i]]++);
  assert(idx.length === n && new Set(idx).size === n && counts.every((k) => k === n / 10), `balanced subset of ${n}`);
}
const first = balancedSubset(data.train, 100);
assert(first.every((i, k) => k === 0 || i > first[k - 1]), 'in file order');

// a racer is a plain SGD loop over seeded shuffled passes: 150 CNN steps on 100 digits (one and a half passes)
const r = racer('cnn', data.train, first, 2);
for (let s = 0; s < 150; s++) r.step();
let w = initWeights({ seed: 2 }), steps = 0;
for (let pass = 1; steps < 150; pass++) for (const k of shuffledOrder(100, 2000 + pass)) {
  if (steps++ >= 150) break;
  const i = first[k]; w = sgdStep(w, lossAndGrad(w, image(data.train, i), data.train.labels[i]).grads, RACE.lr.cnn);
}
assert(r.steps === 150 && JSON.stringify(r.weights) === JSON.stringify(w), 'racer equals the plain loop');
const v = racer('vit', data.train, first, 2); v.step(); v.step();
assert(v.steps === 2 && Array.isArray(v.weights.Wq) && v.accuracy(data.test, [0, 1, 2]) >= 0, 'ViT racer steps');
console.log('core/race.test.mjs: balanced subsets, racer = plain loop, ViT racer ok');
```

- [ ] **Step 2: Run it to see it fail**

Run: `node core/race.test.mjs`. Expected: FAIL, `ERR_MODULE_NOT_FOUND … core/race.js`.

- [ ] **Step 3: Write the racer**

Create `core/race.js`:

```js
// core/race.js — the CNN and the ViT trained on the same small set of digits, for the data-hunger comparison. The
// offline measurement (tools/vit_vs_cnn.mjs) and the live race on vit-train/ both step these racers, so a live run at a
// measured seed reproduces that dot of the chart exactly. Pure, no DOM.
import { image } from './data.js';
import { initWeights, lossAndGrad as cnnGrad, sgdStep as cnnStep, accuracy as cnnAccuracy, shuffledOrder } from './train.js';
import { init, lossAndGrad as vitGrad, sgdStep as vitStep, accuracy as vitAccuracy } from './vit.js';

/** The settings shared by the measurement and the live race: training-set sizes, steps per run, learning rates. */
export const RACE = { sizes: [100, 300, 1000, 3000], steps: 30000, lr: { cnn: 0.1, vit: 0.003 } };

/** The first n/10 digits of each class in the set, in file order: a balanced training set of n digits. */
export function balancedSubset(set, n) {
  const per = n / 10, seen = new Array(10).fill(0), out = [];
  for (let i = 0; i < set.n && out.length < n; i++) { const l = set.labels[i]; if (seen[l] < per) { seen[l]++; out.push(i); } }
  return out;
}

/** One model ('cnn' or 'vit') training with SGD on the digits idx of a set, in seeded shuffled passes. */
export function racer(kind, set, idx, seed) {
  const lr = RACE.lr[kind];
  let w = kind === 'cnn' ? initWeights({ seed }) : init(seed), steps = 0, pass = 0, order = [], pos = 0;
  return {
    kind, seed,
    get steps() { return steps; },
    get weights() { return w; },
    /** One training digit: forward, backward, update. */
    step() {
      if (pos >= order.length) { pass++; order = shuffledOrder(idx.length, seed * 1000 + pass).map((k) => idx[k]); pos = 0; }
      const i = order[pos++], img = image(set, i), y = set.labels[i];
      w = kind === 'cnn' ? cnnStep(w, cnnGrad(w, img, y).grads, lr) : vitStep(w, vitGrad(w, img, y).grads, lr);
      steps++;
    },
    accuracy(testSet, testIdx) { return kind === 'cnn' ? cnnAccuracy(w, testSet, testIdx) : vitAccuracy(w, testSet, testIdx); },
  };
}
```

- [ ] **Step 4: Run the test**

Run: `node core/race.test.mjs`. Expected: `core/race.test.mjs: balanced subsets, racer = plain loop, ViT racer ok`.

- [ ] **Step 5: Write the measurement script**

Create `tools/vit_vs_cnn.mjs`:

```js
// tools/vit_vs_cnn.mjs — measure the CNN and the ViT on small training sets, for the chart on vit-train/: held-out
// accuracy after RACE.steps SGD steps on balanced subsets of RACE.sizes digits, seeds 1 to 5, both models.
//   node tools/vit_vs_cnn.mjs            runs the 40 trainings in parallel and writes models/vit_vs_cnn.json
//   node tools/vit_vs_cnn.mjs --job cnn 100 1    one training; prints its held-out accuracy
import { readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { fileURLToPath } from 'node:url';
import { parseMnist16 } from '../core/data.js';
import { RACE, balancedSubset, racer } from '../core/race.js';

const SEEDS = [1, 2, 3, 4, 5];
const argv = process.argv.slice(2);

if (argv[0] === '--job') {
  const [kind, n, seed] = [argv[1], +argv[2], +argv[3]];
  const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
  const r = racer(kind, data.train, balancedSubset(data.train, n), seed);
  while (r.steps < RACE.steps) r.step();
  console.log(JSON.stringify({ kind, n, seed, acc: +r.accuracy(data.test, [...Array(data.test.n).keys()]).toFixed(4) }));
} else {
  const jobs = [];
  for (const kind of ['cnn', 'vit']) for (const n of RACE.sizes) for (const seed of SEEDS) jobs.push([kind, n, seed]);
  const me = fileURLToPath(import.meta.url), results = [], t0 = Date.now();
  const run = ([kind, n, seed]) => new Promise((done, fail) => {
    let out = '';
    const p = spawn(process.execPath, [me, '--job', kind, String(n), String(seed)]);
    p.stdout.on('data', (d) => { out += d; });
    p.on('close', (code) => (code === 0 ? done(JSON.parse(out)) : fail(new Error(`${kind} ${n} ${seed} exited ${code}`))));
  });
  const workers = Array.from({ length: Math.max(1, availableParallelism() - 1) }, async () => {
    while (jobs.length) { const r = await run(jobs.shift()); results.push(r); console.log(`${r.kind} ${r.n} digits, seed ${r.seed}: ${(100 * r.acc).toFixed(1)}%`); }
  });
  await Promise.all(workers);
  const table = (kind) => RACE.sizes.map((n) => SEEDS.map((seed) => results.find((r) => r.kind === kind && r.n === n && r.seed === seed).acc));
  const out = { version: 1, sizes: RACE.sizes, seeds: SEEDS, steps: RACE.steps, lr: RACE.lr, cnn: table('cnn'), vit: table('vit'),
    made: 'node tools/vit_vs_cnn.mjs', date: new Date().toLocaleDateString('en-CA') };
  writeFileSync(new URL('../models/vit_vs_cnn.json', import.meta.url), JSON.stringify(out) + '\n');
  console.log(`wrote models/vit_vs_cnn.json (${((Date.now() - t0) / 1000) | 0} s)`);
}
```

- [ ] **Step 6: Two single runs, against the earlier measurement**

Run: `node tools/vit_vs_cnn.mjs --job cnn 100 1 && node tools/vit_vs_cnn.mjs --job vit 100 1`
Expected: `{"kind":"cnn","n":100,"seed":1,"acc":0.725}` and `{"kind":"vit","n":100,"seed":1,"acc":0.476}`. These are the scratch measurements the spec reports, reproduced exactly.

- [ ] **Step 7: The full measurement**

Run (about 2 minutes; Bash timeout at least 600000 ms): `node tools/vit_vs_cnn.mjs`
Expected: 40 lines, then `wrote models/vit_vs_cnn.json (… s)`. Print the table:
```bash
node -e "const j=require('./models/vit_vs_cnn.json');j.sizes.forEach((n,k)=>{const f=a=>a.map(x=>(100*x).toFixed(1)).join(' '),m=a=>(100*a.reduce((s,x)=>s+x)/a.length).toFixed(1);console.log(n,'CNN',f(j.cnn[k]),'mean',m(j.cnn[k]),'| ViT',f(j.vit[k]),'mean',m(j.vit[k]))})"
```
Expected (the prototype's run):
```text
100 CNN 72.5 74.7 71.9 67.7 76.0 mean 72.6 | ViT 47.6 45.3 50.4 41.8 42.3 mean 45.5
300 CNN 82.7 66.4 72.9 75.8 75.1 mean 74.6 | ViT 62.9 64.1 59.6 60.9 60.9 mean 61.7
1000 CNN 78.6 73.8 76.7 69.2 70.5 mean 73.8 | ViT 75.2 76.1 72.7 72.2 69.2 mean 73.1
3000 CNN 78.9 79.1 72.4 71.2 84.8 mean 77.3 | ViT 82.0 72.3 70.1 77.5 76.1 mean 75.6
```

- [ ] **Step 8: Commit**

```bash
git add core/race.js core/race.test.mjs tools/vit_vs_cnn.mjs models/vit_vs_cnn.json
git commit -m "feat: CNN against ViT on 100 to 3,000 digits, 5 seeds, one shared racer" -m "Held-out after 30,000 steps, mean of 5 seeds: 100 digits CNN 72.6% ViT 45.5%; 300: 74.6 / 61.7; 1,000: 73.8 / 73.1; 3,000: 77.3 / 75.6." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
If Step 7's numbers differ, put the measured ones in the commit body instead, and say so in the report.

---

### Task 3: The shared tensor drawing, and `vit/` moved onto it unchanged

**Files:**
- Create: `ui/vitline.js`, `ui/vitline.css`
- Modify: `vit/index.html` (edit script)
- Modify: `.superpowers/check/vitcheck.mjs` (gains `--page`)
- Create: `.superpowers/check/pngdiff.py` (helper)

**Interfaces:**
- Produces, from `ui/vitline.js`:
  - `el(cls, text)`;
  - `createViews({ shapeOf, onProbe })` returns `{ views, view(key, cell, o), kOf }`;
  - `line(box, rows, cell, { labels })`: `rows` is one item list, or several lists that share their columns, with sticky row labels; an item is `{ op, id }`, `{ v, up, lb, w, grid, sec, id }`, or `null`;
  - `arcs(box, pairs)`; `drawBars`, `barAt`;
  - `paint(v, M, C, G, scale, y, { gray, faint, outline })`.
- `vit/index.html` keeps all its behaviour.

- [ ] **Step 1: The harness learns `--page`, and a baseline of `vit/` before the move**

Replace `.superpowers/check/vitcheck.mjs` with:

```js
// vitcheck.mjs — drive vit/index.html in headless Edge and print what a scenario recorded.
//   node vitcheck.mjs <root> <scenario.js> [--page vit|vit-train] [--shot out.png] [--width 1400] [--height 1000] [--dark] [--no-weights]
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

const pageDir = opt('page', 'vit');
let html = readFileSync(join(root, pageDir, 'index.html'), 'utf8');
const end = html.lastIndexOf('</script>');
html = html.slice(0, end) + `\n;(async () => { const R = {}; try { ${scenario}\n } catch (e) { R.error = String(e && e.stack || e); } document.body.dataset.check = JSON.stringify(R); })();\n` + html.slice(end);
if (flag('dark')) html = html.replace('<html lang="en">', '<html lang="en" data-theme="dark">');
if (flag('no-weights')) html = html.replace("'../models/vit16.json'", "'../models/missing.json'");
const page = join(root, pageDir, '_check.html');
writeFileSync(page, html);

const server = spawn('python', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', root], { stdio: 'ignore' });
const url = `http://127.0.0.1:${PORT}/${pageDir}/_check.html`;
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

Create `.superpowers/check/pngdiff.py` (reports the box of the pixels that differ between two PNGs):

```python
# pngdiff.py a.png b.png — standard-library PNG decode (8-bit RGB/RGBA, non-interlaced) and the bounding box of the
# pixels that differ, plus their count.
import sys, zlib, struct

def load(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos, idat, w = 8, b'', None
    while pos < len(data):
        n, typ = struct.unpack('>I4s', data[pos:pos + 8]); body = data[pos + 8:pos + 8 + n]; pos += 12 + n
        if typ == b'IHDR': w, h, depth, ctype = struct.unpack('>IIBB', body[:10]); assert depth == 8
        elif typ == b'IDAT': idat += body
    bpp = {2: 3, 6: 4}[ctype]; raw = zlib.decompress(idat); stride = w * bpp; rows, prev = [], bytearray(stride); i = 0
    for _ in range(h):
        f = raw[i]; line = bytearray(raw[i + 1:i + 1 + stride]); i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0; b = prev[x]; c = prev[x - bpp] if x >= bpp else 0
            if f == 1: line[x] = (line[x] + a) & 255
            elif f == 2: line[x] = (line[x] + b) & 255
            elif f == 3: line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(bytes(line)); prev = line
    return w, h, bpp, rows

wa, ha, ba, A = load(sys.argv[1]); wb, hb, bb, B = load(sys.argv[2])
print('sizes', (wa, ha), (wb, hb))
x0 = y0 = 10**9; x1 = y1 = -1; n = 0
for y in range(min(ha, hb)):
    if A[y] == B[y]: continue
    for x in range(min(wa, wb)):
        if A[y][x * ba:x * ba + 3] != B[y][x * bb:x * bb + 3]:
            n += 1; x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
print('differing pixels', n, 'box', (x0, y0, x1, y1) if n else None)
```

Run, before any change to `vit/`:
```bash
C=.superpowers/check
node $C/vitcheck.mjs . $C/scen_basic.js --shot $C/base_basic.png --height 4200 > $C/base_basic.json 2>&1
node $C/vitcheck.mjs . $C/scen_hover.js --shot $C/base_hover.png --height 900 > /dev/null 2>&1
node $C/vitcheck.mjs . $C/scen_hover.js --dark --width 1000 --shot $C/base_dark.png --height 900 > /dev/null 2>&1
```

- [ ] **Step 2: Write the shared module and its styles**

Create `ui/vitline.js`:

```js
// ui/vitline.js — how the ViT pages draw tensors. A view is a canvas showing all or part of one tensor; a line lays views
// out as columns, token i always on row i, with keys and values above the token rows and labels under them; paint draws
// a view from its tensor. Shared by vit/ and vit-train/; styles in ui/vitline.css.
import { T, NC } from '../core/vit.js';
import { drawGrid, cellAt, cssVar } from './grid.js';

export const el = (cls, text) => { const d = document.createElement('div'); d.className = cls; if (text != null) d.textContent = text; return d; };
const pct = (x) => (100 * x).toFixed(1) + '%';

/** A view factory. shapeOf(key) gives a tensor's [rows, cols]; onProbe({ t, r, c } | null) is called as the pointer
 *  moves over a view's cells, with the tensor cell under it. */
export function createViews({ shapeOf, onProbe }) {
  const views = []; // { cv, key, rows, cols, cell, map, yl, own, bars, blocks }
  const kOf = (v) => (typeof v.key === 'function' ? v.key() : v.key);
  /** A canvas for tensor `key` (or a function giving it) at `cell` px a cell. o.rows, o.cols and o.map(r, c) → [row, col]
   *  in the tensor show a part of it; o.yl() gives the yellow ({ win, cell }); o.own() replaces the hover's green;
   *  o.bars draws probabilities as bars on a canvas of o.px = [width, height]; o.blocks draws the patch grid. */
  function view(key, cell, o = {}) {
    const [R, C] = shapeOf(typeof key === 'function' ? key() : key);
    const v = { key, cell, rows: o.rows ?? R, cols: o.cols ?? C, map: o.map || ((r, c) => [r, c]), yl: o.yl, own: o.own, bars: o.bars, blocks: o.blocks };
    const cv = document.createElement('canvas'); v.cv = cv;
    cv.width = o.px ? o.px[0] : v.cols * cell; cv.height = o.px ? o.px[1] : v.rows * cell;
    cv.style.width = cv.width + 'px'; cv.style.height = cv.height + 'px';
    const probeAt = (ev) => { const p = v.bars ? barAt(v, ev) : cellAt(cv, ev); onProbe(p ? (([r, c]) => ({ t: kOf(v), r, c }))(v.map(p[0], p[1])) : null); };
    cv.addEventListener('pointermove', probeAt);
    cv.addEventListener('pointerdown', probeAt);
    cv.addEventListener('pointerleave', (ev) => { if (ev.pointerType !== 'touch') onProbe(null); });
    views.push(v); return cv;
  }
  return { views, view, kOf };
}

/**
 * Lay out columns on a grid. `rows` is one list of items, or several lists that share their columns (the training
 * page's forward, backward and update rows), with `labels` naming them in a sticky first column. An item is
 * { op, id }: an operator; { v: [elements], up, lb, w, grid, sec, id }: a column, `up` above the token rows (keys and
 * values), `w` tinted as learned, `grid` its number of sub-columns, `sec` the section a click on it scrolls to; or null:
 * an empty place. Shorter columns sit level with the middle of the token rows.
 */
export function line(box, rows, cell, { labels = null } = {}) {
  const bands = Array.isArray(rows[0]) ? rows : [rows], off = labels ? 1 : 0;
  box.className = 'line'; box.innerHTML = ''; box.style.setProperty('--mid', Math.max(0, T * cell / 2 - 9) + 'px');
  bands.forEach((items, b) => {
    const gr = (k) => String(3 * b + k);
    if (labels) { const l = el('rl', labels[b]); l.style.gridColumn = '1'; l.style.gridRow = gr(2); box.append(l); }
    items.forEach((it, x) => {
      if (!it) return;
      const col = String(x + 1 + off);
      if (it.op != null) { const o = el('op', it.op); o.style.gridColumn = col; o.style.gridRow = gr(2); if (it.id) o.dataset.id = it.id; box.append(o); return; }
      const parts = [];
      if (it.up) { const u = el('up'); u.append(...it.up); u.style.gridRow = gr(1); parts.push(u); }
      const a = el('at' + (it.w ? ' w' : '')); if (it.grid) a.style.gridTemplateColumns = `repeat(${it.grid}, auto)`; a.append(...it.v); a.style.gridRow = gr(2); if (it.id) a.dataset.id = it.id; parts.push(a);
      const l = el('lb', it.lb || ''); l.style.gridRow = gr(3); parts.push(l);
      for (const p of parts) { p.style.gridColumn = col; if (it.sec) p.dataset.sec = it.sec; box.append(p); }
      a.style.marginTop = Math.max(0, (T * cell - a.offsetHeight) / 2) + 'px';
    });
  });
}

/** Dashed residual arcs under a line's labels, from a tensor's column to the + that adds it back. */
export function arcs(box, pairs) {
  box.querySelector('svg.arcs')?.remove();
  const b = box.getBoundingClientRect(); if (!b.width) return;
  const ids = pairs.flat().map((id) => box.querySelector(`[data-id="${id}"]`));
  if (ids.some((e) => !e)) return;
  const labRow = +ids[0].style.gridRow + 1; // the labels of the arcs' own row of the line
  const labs = [...box.querySelectorAll('.lb')].filter((l) => +l.style.gridRow === labRow).map((l) => l.getBoundingClientRect().bottom - b.top);
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

/** Probabilities as bars, labelled when there is room; glow outlines the lit classes. */
export function drawBars(v, vals, glow) {
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
export function barAt(v, ev) { const b = v.cv.getBoundingClientRect(), r = Math.floor((ev.clientY - b.top) / (b.height / NC)); return r >= 0 && r < NC ? [r, 0] : null; }

/** Draw view v from its tensor M (rows of numbers, C columns). G: a green strength per tensor cell, or null; scale: the
 *  colour scale; y: the yellow { win, cell }; o.gray draws grey values (the digit), o.faint dims a view not computed
 *  yet, o.outline frames it in green (what the last step computed). */
export function paint(v, M, C, G, scale, y = {}, o = {}) {
  const vals = new Float32Array(v.rows * v.cols), glow = G ? new Float32Array(v.rows * v.cols) : null;
  for (let r = 0; r < v.rows; r++) for (let c = 0; c < v.cols; c++) {
    const [mr, mc] = v.map(r, c); vals[r * v.cols + c] = M[mr][mc]; if (glow) glow[r * v.cols + c] = G[mr * C + mc];
  }
  if (v.bars) drawBars(v, vals, glow);
  else drawGrid(v.cv, vals, v.cols, v.rows, { mode: o.gray ? null : 'div', scale, glow, lines: v.cell >= 7, blocks: v.blocks, win: y.win, cell: y.cell, outline: o.outline });
  v.cv.style.opacity = o.faint ? '0.22' : '';
}
```

Create `ui/vitline.css`:

```css
/* ui/vitline.css — lines of tensors on the ViT pages (ui/vitline.js): keys and values above the token rows, labels under
   them; operators level with the middle of the tokens; learned weights tinted. Uses the page's colour tokens. */
.wrap{overflow-x:auto;padding-bottom:4px}
.line{display:grid;grid-template-rows:auto auto auto;column-gap:8px;row-gap:3px;width:max-content;align-items:start;position:relative;padding-bottom:14px}
.line .up{grid-row:1;align-self:end;justify-self:center;display:grid;gap:3px}
.line .at{grid-row:2;display:grid;gap:3px;justify-items:start;justify-self:center}
.line .at.w{background:var(--wtint);border:1px solid var(--accent);border-radius:4px;padding:2px}
.line .lb{grid-row:3;font-size:11.5px;color:var(--ink2);text-align:center;white-space:nowrap;line-height:1.25}
.line .op{grid-row:2;margin-top:var(--mid);font-family:var(--mono);font-size:12px;color:var(--ink2);white-space:nowrap}
.line .sl{font-size:11px;color:var(--ink2)}
.line .rl{position:sticky;left:0;z-index:2;align-self:stretch;background:var(--bg);padding:4px 10px 0 4px;font-size:12.5px;font-weight:600;color:var(--ink2)}
.line [data-sec]{cursor:pointer}
.kg{display:grid;gap:3px}
.kg canvas.pick{cursor:pointer}
svg.arcs{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
```

- [ ] **Step 3: Move `vit/` onto it**

Create `.superpowers/check/edit_vit.py`:

```python
# Move the explorer's tensor drawing into ui/vitline.js and ui/vitline.css (vit/index.html).
import sys, re
p = sys.argv[1]
s = open(p, encoding='utf-8').read()

def cut(start, end):
    """Remove from the line starting with `start` up to (not including) the line starting with `end`."""
    global s
    a = s.index(start); b = s.index(end, a)
    s = s[:a] + s[b:]

def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)

rep('<link rel="stylesheet" href="../ui/terms.css">\n', '<link rel="stylesheet" href="../ui/terms.css">\n<link rel="stylesheet" href="../ui/vitline.css">\n')
cut('/* a line of tensors: storey (keys and values) over the token rows, labels under */\n', '@media (prefers-reduced-motion: reduce)')
rep("import { drawGrid, cellAt, fmt, maxAbs, cssVar } from '../ui/grid.js';\n",
    "import { fmt, maxAbs } from '../ui/grid.js';\nimport { createViews, line, arcs, paint, el } from '../ui/vitline.js';\n")
cut('// ---------- views: one canvas showing (part of) a tensor, by its SHAPES key ----------\n', 'const kernels = ')
rep('const kernels = ', """// ---------- views and lines (ui/vitline.js): every copy of a tensor lights together ----------
const { views, view, kOf } = createViews({ shapeOf: (key) => SHAPES[key], onProbe: setProbe });
function setProbe(p) { if (JSON.stringify(p) !== JSON.stringify(probe)) { probe = p; render(); } }
const kernels = """)
cut('// ---------- probabilities as bars ----------\n', '// ---------- render ----------')
rep("""  for (const v of views) {
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
  }""", """  for (const v of views) {
    const key = kOf(v), M = tensor(key, S, w), s = key === 'digit' ? 1 : (scale[key] ??= maxAbs(M.flat()));
    paint(v, M, SHAPES[key][1], v.own ? v.own() : green[key], s, v.yl ? v.yl() : {}, { gray: key === 'digit' });
  }""")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('vit/index.html refactored')
```

Run: `python .superpowers/check/edit_vit.py vit/index.html`. Expected: `vit/index.html refactored`, with the file now about 408 lines.

- [ ] **Step 4: Prove `vit/` unchanged**

Run:
```bash
C=.superpowers/check
node $C/vitcheck.mjs . $C/scen_basic.js --shot $C/new_basic.png --height 4200 > $C/new_basic.json 2>&1
node $C/vitcheck.mjs . $C/scen_hover.js --shot $C/new_hover.png --height 900 > /dev/null 2>&1
node $C/vitcheck.mjs . $C/scen_hover.js --dark --width 1000 --shot $C/new_dark.png --height 900 > /dev/null 2>&1
diff <(grep -v screenshot $C/base_basic.json) <(grep -v screenshot $C/new_basic.json) && echo "scenario identical"
for f in basic hover dark; do cmp -s $C/base_$f.png $C/new_$f.png && echo "$f identical" || python $C/pngdiff.py $C/base_$f.png $C/new_$f.png; done
```
Expected: `scenario identical` and the three screenshots identical.

Edge's first capture in a session can differ by a few pixels at the page edge. If one screenshot differs, capture the baseline again from the unchanged file before deciding: `git stash`, rerun Step 1's line for that screenshot, `git stash pop`, compare again. In the prototype, the refactored dark screenshot equalled the original's in two fresh captures.

- [ ] **Step 5: Commit**

```bash
git add ui/vitline.js ui/vitline.css vit/index.html
git commit -m "refactor(vit): the tensor drawing moves to ui/vitline.js" -m "Lines can now stack several rows that share columns, for the training page. vit/ is unchanged: same scenario values, byte-identical screenshots." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The ViT hand-off, the training terms, and `vit/` reading stored weights

**Files:**
- Modify: `ui/trained.js` (append), `ui/terms.js` (edit script), `vit/index.html` (edit script)

**Interfaces:**
- Produces:
  - in `ui/trained.js`: `VIT_KEY = 'dl-explorer.vit-weights.v1'`, `saveTrainedViT(o)`, `loadTrainedViT(isWeights)`, `clearTrainedViT()`, `showTrainedViTBanner(anchor, o)`;
  - the stored object: `{ version: 1, arch: 'vit16', weights, meta: { source: 'vit-train', digits, passes, optimiser: 'sgd', lr, batch, testAcc, testAccNoPos, date } }`;
  - in `ui/terms.js`: `VIT_TRAIN_TERMS`.
- Consumers: `vit-train/` (Task 5) calls `saveTrainedViT` and uses `VIT_TRAIN_TERMS`; `vit/` calls `loadTrainedViT` and the banner.

- [ ] **Step 1: The CNN pages' DOM, before**

Run: `node .superpowers/check/domcheck.mjs . .superpowers/check/dom-before-t4`
Expected: three lines, with 30, 4 and 18 dotted terms. If `cnn-3d` shows 0, rerun: three.js did not load.

- [ ] **Step 2: The hand-off functions**

Append to `ui/trained.js` (after `showTrainedBanner`):

```js

// The ViT's own hand-off, from vit-train/ to vit/: { version: 1, arch: 'vit16', weights, meta }, the shape of
// models/vit16.json, with meta.source 'vit-train'. The ViT pages pass core/vit.js's isWeights to check the weights,
// so this module stays free of the model code.
export const VIT_KEY = 'dl-explorer.vit-weights.v1';

export function saveTrainedViT(o) { localStorage.setItem(VIT_KEY, JSON.stringify(o)); }

/** The stored ViT weights, or null when there are none or isWeights rejects them. */
export function loadTrainedViT(isWeights) {
  try {
    const s = localStorage.getItem(VIT_KEY); if (!s) return null;
    const o = JSON.parse(s);
    return o && o.version === 1 && o.arch === 'vit16' && o.meta && isWeights(o.weights) ? o : null;
  } catch { return null; }
}

export function clearTrainedViT() { try { localStorage.removeItem(VIT_KEY); } catch { /* storage blocked: nothing to clear */ } }

/** A banner after `anchor` saying the ViT page uses weights from the training page; its link goes back to the shipped ones. */
export function showTrainedViTBanner(anchor, o) {
  const d = document.createElement('div'); d.className = 'banner';
  const acc = Number.isFinite(o.meta.testAcc) ? `, held-out accuracy ${Math.round(o.meta.testAcc * 100)}%` : '';
  d.innerHTML = `Using weights trained on the <a href="../vit-train/">ViT training page</a> (${o.meta.digits.toLocaleString('en-NZ')} digits${acc}). <a href="#" data-clear>Back to the shipped weights</a>.`;
  d.querySelector('[data-clear]').onclick = (e) => { e.preventDefault(); clearTrainedViT(); location.reload(); };
  anchor.after(d);
}
```

- [ ] **Step 3: The terms**

Create `.superpowers/check/edit_terms.py`:

```python
# ui/terms.js: "classifier head" is not an attention head; the ViT training page's term list.
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("  ['head', /(?<!W )\\b(attention )?heads?\\b/i,", "  ['head', /(?<!W |classifier )\\b(attention )?heads?\\b/i,")
rep("""  ['heldout', /\\bheld-out\\b/i, 'Held-out test set', 'Digits the network never trained on, kept only to measure accuracy.'],
];
""", """  ['heldout', /\\bheld-out\\b/i, 'Held-out test set', 'Digits the network never trained on, kept only to measure accuracy.'],
];

/** The ViT training page's terms: the ViT's, the training terms the CNN pages define (their wording fits both), and two
 *  of its own. */
const TRAINING = ['loss', 'gradient', 'backprop', 'lr', 'update', 'batch', 'pass', 'init'];
export const VIT_TRAIN_TERMS = [
  ...VIT_TERMS,
  ...TERMS.filter((t) => TRAINING.includes(t[0])),
  ['seed', /\\bseeds?\\b/i, 'Seed', 'The number that starts the random number generator, which sets the starting weights and the order of the digits. Two runs with the same seed are identical; runs with different seeds show how much of a result is luck.'],
  ['locality', /\\blocality\\b/i, 'Locality', 'A convolution only ever combines neighbouring pixels: an assumption built into its shape, true of images, that a CNN never has to learn. Attention can join any two patches, so a ViT has to learn from examples which ones matter.'],
];
""")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ui/terms.js edited')
```

Run: `python .superpowers/check/edit_terms.py ui/terms.js`. Expected: `ui/terms.js edited`.

- [ ] **Step 4: `vit/` reads handed-off weights, links to the training page, and colours links**

Create `.superpowers/check/edit_vit2.py`:

```python
# vit/: load weights handed off by vit-train/, say where the weights came from, link to the training page.
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("import { installTerms, VIT_TERMS } from '../ui/terms.js';\n",
    "import { installTerms, VIT_TERMS } from '../ui/terms.js';\nimport { loadTrainedViT, showTrainedViTBanner } from '../ui/trained.js';\n")
rep("  if (js && js.version === 1 && isWeights(js.weights)) { wTrained = js.weights; meta = js.meta; }\n",
    "  const stored = loadTrainedViT(isWeights);\n"
    "  if (stored) { wTrained = stored.weights; meta = stored.meta; showTrainedViTBanner(document.querySelector('h1'), stored); }\n"
    "  else if (js && js.version === 1 && isWeights(js.weights)) { wTrained = js.weights; meta = js.meta; }\n")
rep("    ? `These weights were trained offline by tools/train_vit.mjs on the ${n(data.train)} training digits (${meta.passes} passes, ${opt}); with",
    "    ? `These weights were trained ${meta.source === 'vit-train' ? `on the ViT training page (${meta.digits.toLocaleString('en-NZ')} digits seen, ${opt}, learning rate ${meta.lr})` : `offline by tools/train_vit.mjs on the ${n(data.train)} training digits (${meta.passes} passes, ${opt})`}; with")
rep("so this toy uses the average of the tokens, as many current ViTs do.</p>",
    "so this toy uses the average of the tokens, as many current ViTs do. The <a href=\"../vit-train/\">ViT training page</a> trains it live and races it against the CNN.</p>")
rep(".crumb{font-size:13px;color:var(--ink2);margin:0 0 10px}.crumb a{color:var(--accent);text-decoration:none}.crumb a:hover{text-decoration:underline}\n",
    ".crumb{font-size:13px;color:var(--ink2);margin:0 0 10px}.crumb a{color:var(--accent);text-decoration:none}.crumb a:hover{text-decoration:underline}\nmain a{color:var(--accent)}\n")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('vit/index.html: hand-off loaded')
```

Run: `python .superpowers/check/edit_vit2.py vit/index.html`. Expected: `vit/index.html: hand-off loaded`.

- [ ] **Step 5: Checks**

Create `.superpowers/check/scen_stored.js`:

```js
await ready;
// First load: store a hand-off as vit-train/ would, and reload. Second load: what vit/ makes of it.
if (!localStorage.getItem('dl-explorer.vit-weights.v1')) {
  const j = await (await fetch('../models/vit16.json')).json();
  j.meta = { ...j.meta, source: 'vit-train', digits: 1234, lr: 0.003 };
  localStorage.setItem('dl-explorer.vit-weights.v1', JSON.stringify(j));
  location.reload();
} else {
  R.banner = (document.querySelector('.banner') || {}).textContent || null;
  R.wnote = $('wnote').textContent.slice(0, 110);
  R.trained = !!wTrained && meta.source === 'vit-train';
}
```

Run:
```bash
C=.superpowers/check
node $C/vitcheck.mjs . $C/scen_stored.js
node $C/vitcheck.mjs . $C/scen_basic.js
node $C/domcheck.mjs . $C/dom-after-t4 && diff -r $C/dom-before-t4 $C/dom-after-t4 && echo IDENTICAL
```
Expected:
- **The stored-weights scenario:** `"banner": "Using weights trained on the ViT training page (1,234 digits, held-out accuracy 82%). Back to the shipped weights."`, a `"wnote"` starting "These weights were trained on the ViT training page (1,234 digits seen, SGD, learning rate 0.003)", `"trained": true`, and no console errors.
- **The basic scenario:** as in Task 3 (`"dfn": 41`, no console errors).
- **The CNN DOM check:** `IDENTICAL`.

- [ ] **Step 6: Commit**

```bash
git add ui/trained.js ui/terms.js vit/index.html
git commit -m "feat: ViT weights hand off from the training page to the explorer" -m "ui/trained.js gains a ViT key; ui/terms.js the training page's terms; vit/ loads stored weights with a banner and links to vit-train/." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The training page

**Files:**
- Create: `vit-train/index.html`
- Create: `.superpowers/check/scen_train.js`, `.superpowers/check/scen_load.js`

**Interfaces:**
- Consumes: everything from Tasks 1 to 4, plus `models/vit_vs_cnn.json`, `data/mnist16.bin`, `ui/grid.js`.
- Produces: module-scope names the scenarios use: `ready`, `views`, `kOf`, `setProbe`, `cur`, `probe`, `step`, `advance`, `reset`, `seed`, `lr`, `weights`, `digitsSeen`, `snaps`, `accTest`, `renderAll`, `measured`, `RACE`, `raceSeed`, `newRace`, `racing`, `race`, `raceFrame`, `raceDots`, `chartDots`, plus the imports (`init`, `lossAndGrad`, `sgdStep`, `shuffledOrder`, `image`, `data`, `gradTrace`, `vitTrace`).

How the page works, to read before the code:
- **One step.** `build()` lays out the ViT line three times with `line(…, { labels })`. The forward row uses SHAPES keys, the backward row `'d'` + key, the update row `'n'` + key.
  - `renderNet()` paints every view: faint until its stage is revealed, framed in green while it is the front.
  - Green comes from `vitTrace` for a forward activation, and from `gradTrace` for a gradient or a weight.
  - The phases and the play loop follow `cnn-train/`.
- **Many steps.** `record()` keeps the curves, evaluates the held-out digits every 500 digits, and takes a `snapshot()` at 0, 300, 1,000, 3,000, 10,000 and 30,000 digits: both heads' maps for the first held-out 7, and the position table's similarities.
- **CNN against ViT.**
  - `drawChart()` draws the measured dots, the mean lines and the race's hollow dots. `chartDots` feeds the hover tooltip.
  - The race steps two `racer`s in a frame loop and evaluates every 2,000 steps. When it finishes it adds its result to the chart.

- [ ] **Step 1: The scenarios**

Create `.superpowers/check/scen_train.js`:

```js
await ready;
R.views = views.length;
R.overflow = document.documentElement.scrollWidth - innerWidth;
R.glossary = $('gloss').children.length; R.dfn = document.querySelectorAll('dfn').length;
R.arcs = !!document.querySelector('#net svg.arcs');
// the phases: faint views fall to 0, the green front moves stage by stage
const faint = () => views.filter((v) => v.cv.style.opacity === '0.22').length;
const front = () => views.filter((v) => v.cv.style.outline).length;
R.phases = [];
for (let k = 0; k < 7; k++) { R.phases.push([cur.phase, faint(), front(), $('stepBtn').textContent]); if (k < 6) $('stepBtn').click(); }
// hovers: a value's gradient lights the queries that read it; a weight lights its three rows and its two columns
setProbe({ t: 'dv', r: 4, c: 1 });
const lit = (tr) => Object.keys(tr).filter((key) => views.some((v) => kOf(v) === key)).sort();
R.dvLit = lit(gradTrace(cur.S, probe));
R.dvReadout = $('readout').textContent.slice(0, 90);
setProbe({ t: 'nWq', r: 1, c: 2 });
R.wLit = lit(gradTrace(cur.S, probe));
R.wReadout = $('readout').textContent.replace(/\s+/g, ' ').slice(0, 160);
setProbe({ t: 'z1', r: 3, c: 0 });
R.fwdLit = lit(vitTrace(cur.S, probe)).length;
setProbe(null);
// play, digit by digit: the weights equal a plain core/vit.js loop over the same digits
reset();
const K = 300;
for (let k = 0; k < K; k++) advance();
let w = init(seed); const ord = shuffledOrder(data.train.n, seed * 1000);
for (let k = 0; k < K; k++) { const i = ord[k]; w = sgdStep(w, lossAndGrad(w, image(data.train, i), data.train.labels[i]).grads, lr); }
R.playEqualsLoop = JSON.stringify(w) === JSON.stringify(weights) && digitsSeen === K && cur.i === ord[K];
R.snaps = snaps.map((s) => s.at); R.tests = accTest.length;
renderAll(); R.snapCanvases = document.querySelectorAll('#snaps canvas').length; R.thumbs = document.querySelectorAll('#thumbs canvas').length;
// the measured chart and a short race, driven frame by frame
R.measured = measured && [measured.sizes.length, measured.seeds.length];
R.chartNote = $('chartNote').textContent.slice(0, 120);
RACE.steps = 2000; raceSeed = 1; newRace(); racing = true;
for (let ts = 0, n = 0; !race.done && n < 2000; n++) { ts += 100; raceFrame(ts); }
R.raceDone = race.done; R.raceDots = raceDots.length; R.chartDots = chartDots.length; R.raceNote = $('raceNote').textContent.slice(0, 80);
// the hand-off
$('useBtn').click(); R.stored = !!localStorage.getItem('dl-explorer.vit-weights.v1'); R.useMsg = $('useMsg').textContent.slice(0, 40);
localStorage.removeItem('dl-explorer.vit-weights.v1');
```

Create `.superpowers/check/scen_load.js`:

```js
await ready;
R.overflow = document.documentElement.scrollWidth - innerWidth;
R.views = views.length; R.measured = !!measured; R.status = $('status').textContent;
```

Run: `node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_train.js --page vit-train`. Expected: FAIL, `ENOENT … vit-train\index.html`.

- [ ] **Step 2: Write the page**

Create `vit-train/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ViT training — how attention learns</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<link rel="stylesheet" href="../ui/terms.css">
<link rel="stylesheet" href="../ui/vitline.css">
<style>
:root{
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
  --bg:#f7f8f6; --panel:#ffffff; --ink:#1d2430; --ink2:#5b6474; --line:#d9ddd6; --line2:#eef0ec;
  --accent:#2457a8; --accent-soft:#dde7f7;
  --pos:#e5641e; --neg:#2a6fd6; --zero:#f3f4f1;
  --hl:#ffd94a; --hl2:#3fae4b; --wtint:rgba(36,87,168,.07);
  --cnn:#e5641e; --vit:#2457a8;
  --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
  --cnn:#e5641e; --vit:#5f8fe0;
}}
:root[data-theme="dark"]{
  --bg:#171a1f; --panel:#1f232a; --ink:#e8eaee; --ink2:#9aa3b2; --line:#343a44; --line2:#272c35;
  --accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a; --wtint:rgba(127,167,234,.09);
  --cnn:#e5641e; --vit:#5f8fe0;
}
html{scroll-padding-top:env(safe-area-inset-top,0px);}
*{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
main{max-width:1240px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:28px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{margin:0 0 10px;max-width:68ch}
.lede{color:var(--ink2);font-size:15.5px;margin-bottom:10px;max-width:none}
.keys{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-bottom:18px}
.keys span{display:inline-flex;align-items:center;gap:6px}
.keys i{display:inline-block;width:14px;height:14px;border-radius:3px;border:3px solid var(--hl2);background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--hl2) 45%,transparent))}
.keys i.front{border-width:2px;background:transparent}
.keys i.wt{border:1px solid var(--accent);background:var(--wtint)}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:22px}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.head .n{font-family:var(--mono);color:var(--ink2);font-size:13px}
.head p{color:var(--ink2);margin:0;flex:0 0 100%;font-size:14px;max-width:none;padding-right:max(0px,calc(100% - 92ch))}
canvas{display:block;touch-action:pan-x pan-y;border-radius:3px}
.cap{font-size:13px;color:var(--ink2);margin-top:6px;font-variant-numeric:tabular-nums}
.cap b{color:var(--ink);font-weight:600}
.ctrls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
button,select{font:inherit;font-size:13.5px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
label.ctl{display:inline-flex;align-items:center;gap:6px;font-size:13.5px;color:var(--ink2)}
input[type=range]{accent-color:var(--accent)}
.math{font-family:var(--mono);font-size:12.5px;line-height:1.7;background:var(--line2);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre-wrap;min-height:3.2em;margin-top:10px}
.math .sum{color:var(--accent);font-weight:600}
.note{font-size:13.5px;color:var(--ink2);border-left:3px solid var(--accent-soft);padding:4px 12px;margin:10px 0 0;max-width:92ch}
.crumb{font-size:13px;color:var(--ink2);margin:0 0 10px}.crumb a{color:var(--accent);text-decoration:none}.crumb a:hover{text-decoration:underline}
main a{color:var(--accent)}
.phase{display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:12.5px;color:var(--ink2)}
.phase b{font-weight:500;padding:2px 7px;border-radius:10px;border:1px solid var(--line)}
.phase b.done{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
.phase b.next{border-style:dashed;border-color:var(--accent)}
.netwrap{overflow-x:auto;margin-top:12px;border:1px solid var(--line2);border-radius:10px;background:var(--bg);padding:8px 0 4px}
.netwrap .line .rl{background:var(--bg)}
.curves{display:flex;gap:18px;flex-wrap:wrap}
.curves > div{min-width:0;max-width:100%}
.curves canvas{max-width:100%;height:auto}
.snaps{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;margin-top:6px}
.snaps > div{text-align:center;font-size:12px;color:var(--ink2)}
.snaps .trio{display:flex;gap:4px;margin-bottom:4px}
.thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.thumbs > div{text-align:center;font-size:11px;color:var(--ink2);width:34px;font-family:var(--mono)}
.thumbs .ok{color:var(--hl2)} .thumbs .no{color:var(--pos)}
.lrv{font-family:var(--mono);font-size:12.5px;min-width:3.5em}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px}
.legend i.hollow{width:10px;height:10px;border-radius:50%;border:2px solid var(--ink2);background:transparent}
.chartbox{position:relative;max-width:100%;overflow-x:auto}
#ctip{position:fixed;z-index:9;display:none;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;padding:5px 9px;border-radius:6px;font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion: reduce){ *{transition:none!important;scroll-behavior:auto!important} }
</style>
</head>
<body>
<main>
<nav class="crumb"><a href="../">Visualising AI</a> › Vision Transformer, training</nav>
<h1>ViT training</h1>
<p class="lede">The toy Vision Transformer of the <a href="../vit/">explorer</a>, now learning. Real MNIST digits shrunk to 16×16 come in one at a time; the loss says how wrong the guess was, and the error flows back from the loss through the classifier head, the MLP, attention and the patch embedding, giving every weight a gradient. One step is shown in full, then thousands; at the end the ViT races the CNN on small sets of digits, where a CNN's built-in locality is worth the most.</p>
<div class="keys"><span><i></i> what a hovered value is computed from: upstream for an activation (by a row of attention), downstream for a gradient (by a column), the two columns of a weight's gradient</span><span><i class="front"></i> what the last backward step computed</span><span><i class="wt"></i> learned weights, where they act</span></div>
<div class="note" id="status">Loading 4,000 digits (1 MB)…</div>

<section id="s1">
  <div class="head"><h2>One step</h2><span class="n">digit → forward → backward, a stage at a time → update</span>
    <p>The ViT is laid out as on the explorer, token i on row i, with three rows: forward (activations, and the weights where they act), backward (the gradient of each, under it) and update (w′ = w − learning rate × gradient, under each weight). Step runs one phase. Backward starts at the class scores as p − y and flows right to left, a stage per step: the classifier head, the MLP, attention, the patch embedding; each step frames in green what it computed. Through attention the error flows the other way round from the forward pass: a token's output read a row of the attention map, so the gradient of a token's key and value collects from a column, every query that read it, in proportion. A weight's gradient sums, over the 16 tokens, what came forward into it times what came back out of it. Next digit finishes this digit and brings the next. While Many steps plays, this band follows the training.</p></div>
  <div class="ctrls">
    <span class="phase" id="phase"><b data-p="0">digit</b>→<b data-p="1">forward</b>→<b data-p="2">back: head</b>→<b data-p="3">back: MLP</b>→<b data-p="4">back: attention</b>→<b data-p="5">back: embedding</b>→<b data-p="6">update</b></span>
    <button id="stepBtn" class="primary">Step</button><button id="nextBtn">Next digit</button>
    <span class="cap" id="digitInfo"></span>
  </div>
  <div class="netwrap"><div id="net"></div></div>
  <div class="math" id="readout"></div>
</section>

<section id="s2">
  <div class="head"><h2>Many steps</h2><span class="n">play · curves · attention and position over training</span>
    <p>Play repeats the step for digit after digit, with SGD. The loss curve shows the loss per digit and its running average; accuracy is measured on the last 200 training digits and, every 500 digits, on the 1,000 held-out digits the network never trains on. This transformer is sensitive to the learning rate: from about 0.1 up it stops learning (measured: 0.1 reached 30% after ten passes, 0.3 nothing), where the CNN page trains well at 0.1. Below, at six moments of training: both heads' attention maps for one fixed digit, and the position table as similarities.</p></div>
  <div class="ctrls">
    <button id="playBtn" class="primary">Play</button><button id="resetBtn" title="new random weights; curves, snapshots and digits cleared">Restart</button>
    <label class="ctl">speed <select id="speed"><option value="2">2 / s</option><option value="10">10 / s</option><option value="50">50 / s</option><option value="200" selected>200 / s</option><option value="500">500 / s</option></select></label>
    <label class="ctl">learning rate <input type="range" id="lr" min="0" max="100" value="33" style="width:120px"><span class="lrv" id="lrv">0.003</span></label>
    <label class="ctl">batch <select id="batch"><option>1</option><option>8</option><option>32</option></select></label>
    <span class="cap" id="counts"></span>
  </div>
  <div class="curves">
    <div><canvas id="cLoss" width="560" height="170"></canvas><div class="cap">loss per digit (faint) and its running average</div></div>
    <div><canvas id="cAcc" width="560" height="170"></canvas><div class="cap">accuracy: <b>training</b>, last 200 digits (thin) · <b>held-out</b>, 1,000 test digits every 500 (thick)</div></div>
  </div>
  <div class="cap" style="margin-top:14px">Over training, for the first held-out 7: head 1's attention map · head 2's · the position table as similarities</div>
  <div class="snaps" id="snaps"></div>
  <div class="note">Each attention map is 16 × 16: row i is where token i looks. The similarity picture puts, at each patch slot, a 4 × 4 map of how alike that slot's row of the position table is to every other slot's (orange alike, blue opposite). In the ViT paper, trained on millions of images, these maps form a 2D grid: neighbours alike. Here, on 3,000 digits, no grid forms (measured: neighbouring slots are no more alike than distant ones), yet the model leans on position: switched off, its held-out accuracy falls from 82% to 34%. It uses each row as a label for its slot, not as a map.</div>
  <div class="cap" style="margin-top:14px">Recent digits, with the guess made at the time</div>
  <div class="thumbs" id="thumbs"></div>
  <div class="ctrls" id="handoff"><button id="useBtn">Use these weights in the ViT explorer</button><button id="dlBtn">Download JSON</button><span class="cap" id="useMsg"></span></div>
</section>

<section id="s3">
  <div class="head"><h2>CNN against ViT</h2><span class="n">held-out accuracy after 30,000 steps, by number of training digits</span>
    <p>Both toys trained on the same few digits, 30,000 SGD steps each, at their own best learning rate. A convolution only combines neighbouring pixels, so a CNN starts out knowing that nearby pixels belong together; attention can join any two patches, so a ViT has to learn locality from the examples. With few examples that costs it a lot. Each dot is one run with its own seed; the lines join the averages.</p></div>
  <div class="legend" id="legend"><span><i style="background:var(--cnn)"></i>CNN</span><span><i style="background:var(--vit)"></i>ViT</span><span><i class="hollow"></i>this page's race</span></div>
  <div class="chartbox"><canvas id="cChart" width="640" height="300"></canvas></div>
  <div class="note" id="chartNote"></div>
  <div class="ctrls" style="margin-top:14px">
    <label class="ctl">race on <select id="raceN"><option value="100" selected>100 digits</option><option value="300">300 digits</option><option value="1000">1,000 digits</option><option value="3000">3,000 digits</option></select></label>
    <button id="racePlay" class="primary">Race</button><button id="raceReset">Restart</button>
    <label class="ctl">speed <select id="raceSpeed"><option value="50">50 / s</option><option value="200" selected>200 / s</option></select></label>
    <span class="cap" id="raceCounts"></span>
  </div>
  <div class="legend"><span><i style="background:var(--cnn)"></i>CNN</span><span><i style="background:var(--vit)"></i>ViT</span></div>
  <div class="chartbox"><canvas id="cRace" width="640" height="220"></canvas></div>
  <div class="note" id="raceNote"></div>
</section>

<section id="s4">
  <div class="head"><h2>Terms</h2><span class="n">dotted words above point here</span>
    <p>Hover or tap a dotted term anywhere above for the same definition in place.</p></div>
  <div class="gloss" id="gloss"></div>
</section>
</main>
<div id="tip" role="tooltip"></div>
<div id="ctip"></div>

<script type="module">
import { SHAPES, T, D, DH, NC, P as PS, init, forward, tensor, lossAndGrad, sgdStep, averageGrads, accuracy, vitTrace, gradTrace, WEIGHT_KEYS, BACK_STAGES } from '../core/vit.js';
import { parseMnist16, image } from '../core/data.js';
import { shuffledOrder } from '../core/train.js';
import { RACE, balancedSubset, racer } from '../core/race.js';
import { drawGrid, fmt, maxAbs, cssVar } from '../ui/grid.js';
import { createViews, line, arcs, paint, el } from '../ui/vitline.js';
import { installTerms, VIT_TRAIN_TERMS } from '../ui/terms.js';
import { saveTrainedViT } from '../ui/trained.js';
const $ = (id) => document.getElementById(id);
const pct = (x) => (100 * x).toFixed(1) + '%';
const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
const MILESTONES = [0, 300, 1000, 3000, 10000, 30000];
const nz = (n) => n.toLocaleString('en-NZ');

// ---------- state ----------
let data = null, testIdx = [], fixedImg = null;
let seed = 1, lr = 0.003, batch = 1, speed = 200;
let weights = null, order = [], orderPos = 0, passes = 0, digitsSeen = 0;
let cur = null;        // { i, input, label, phase (see P), S, loss, grads, back, pred, applied }
let collected = [], shown = null, prevWeights = null;
let probe = null;
let lossRaw = [], lossEma = [], ema = null, trainFlags = [], accTrain = [], accTest = [], lastTest = null, snaps = [], recent = [];
let playing = false, carry = 0, lastFrame = 0, lastDraw = 0, drawn = null;

// ---------- the training step, in phases ----------
// A digit goes through seven phases. Its gradient is computed in one go when the backward pass starts (core/vit.js);
// the band then reveals it a stage at a time (BACK_STAGES), from the loss back towards the input.
const P = { DIGIT: 0, FWD: 1, B0: 2, UPD: 6 };
function loadDigit() {
  if (orderPos >= order.length) { passes++; order = shuffledOrder(data.train.n, seed * 1000 + passes); orderPos = 0; }
  const i = order[orderPos++];
  cur = { i, input: image(data.train, i), label: data.train.labels[i], phase: P.DIGIT };
}
function doForward() {
  const r = lossAndGrad(weights, cur.input, cur.label);
  Object.assign(cur, { S: r.S, loss: r.loss, grads: r.grads, back: r.back, pred: argmax(r.S.prob), phase: P.FWD });
}
function doBackward() { collected.push(cur.grads); shown = collected.length === 1 ? cur.grads : averageGrads(collected); cur.phase = P.B0; }
function doUpdate() {
  if (collected.length >= batch) { prevWeights = weights; weights = sgdStep(weights, shown, lr); collected = []; cur.applied = true; }
  else cur.applied = false;
  cur.phase = P.UPD; digitsSeen++; record(cur);
}
function record(d) {
  lossRaw.push([digitsSeen, d.loss]); ema = ema == null ? d.loss : ema + 0.02 * (d.loss - ema); lossEma.push([digitsSeen, ema]);
  if (lossRaw.length > 4000) { lossRaw = lossRaw.filter((_, i) => i % 2 === 0); lossEma = lossEma.filter((_, i) => i % 2 === 0); }
  trainFlags.push(d.pred === d.label); if (trainFlags.length > 200) trainFlags.shift();
  recent.push({ input: d.input, label: d.label, pred: d.pred }); if (recent.length > 20) recent.shift();
  if (digitsSeen % 20 === 0) accTrain.push([digitsSeen, trainFlags.filter(Boolean).length / trainFlags.length]);
  if (digitsSeen % 500 === 0) evalTest();
  if (MILESTONES.includes(digitsSeen)) snaps.push(snapshot(digitsSeen));
}
function evalTest() { lastTest = accuracy(weights, data.test, testIdx); accTest.push([digitsSeen, lastTest]); }
const finishDigit = () => { if (cur.phase === P.DIGIT) doForward(); if (cur.phase === P.FWD) doBackward(); if (cur.phase < P.UPD) doUpdate(); };
function advance() { finishDigit(); loadDigit(); doForward(); doBackward(); cur.phase = P.UPD - 1; } // play: each digit shown at the end of its backward pass
function step() {
  if (cur.phase === P.UPD) loadDigit(); else if (cur.phase === P.DIGIT) doForward(); else if (cur.phase === P.FWD) doBackward();
  else if (cur.phase < P.UPD - 1) cur.phase++; else doUpdate();
  renderAll();
}
function nextDigit() { finishDigit(); loadDigit(); doForward(); renderAll(); }
function reset({ newSeed = false } = {}) {
  if (newSeed) seed++;
  weights = init(seed); order = shuffledOrder(data.train.n, seed * 1000); orderPos = 0; passes = 0; digitsSeen = 0;
  collected = []; shown = null; prevWeights = null; probe = null;
  lossRaw = []; lossEma = []; ema = null; trainFlags = []; accTrain = []; accTest = []; recent = [];
  snaps = [snapshot(0)];
  evalTest(); loadDigit(); renderAll();
}
/** Both heads' attention maps for the fixed digit, and the position table's similarities laid out on the patch grid. */
function snapshot(at) {
  const S = forward(weights, fixedImg), pos = weights.pos;
  const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let d = 0; d < D; d++) { s += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; } return s / Math.sqrt(na * nb || 1); };
  const sim = new Float32Array(T * T); // slot i's 4 × 4 map sits at the slot's place on the patch grid
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) sim[((i >> 2) * 4 + (j >> 2)) * T + (i & 3) * 4 + (j & 3)] = cos(pos[i], pos[j]);
  return { at, att: S.att.map((A) => Float32Array.from(A.flat())), sim };
}

// ---------- play ----------
function frame(ts) {
  if (!playing) return;
  const dt = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0; lastFrame = ts;
  carry = Math.min(carry + speed * dt, Math.max(1, speed / 10));
  const t0 = performance.now();
  while (carry >= 1 && performance.now() - t0 < 12) { advance(); carry -= 1; }
  if (ts - lastDraw > 100) { if (cur !== drawn) { drawn = cur; renderAll(); } else renderPlay(); lastDraw = ts; } // the band follows, at most 10 times a second
  requestAnimationFrame(frame);
}
function setPlaying(p) {
  playing = p; $('playBtn').textContent = p ? 'Pause' : 'Play';
  if (p) { setRacing(false); finishBackward(); lastFrame = 0; carry = 0; drawn = cur; renderAll(); requestAnimationFrame(frame); }
  else renderAll();
}
function finishBackward() { if (cur.phase === P.DIGIT) doForward(); if (cur.phase === P.FWD) doBackward(); if (cur.phase < P.UPD - 1) cur.phase = P.UPD - 1; }
// the weights the gradient is applied to (before the update; once applied, the ones it was applied to), and w′
const wBefore = () => (cur.phase === P.UPD && cur.applied && prevWeights ? prevWeights : weights);
const wPrime = () => (cur.phase === P.UPD && cur.applied ? weights : shown ? sgdStep(weights, shown, lr) : weights);

// ---------- One step: the ViT line in three rows ----------
// Keys are namespaced: a SHAPES key in the forward row, 'd' + key for its gradient, 'n' + key for an updated weight.
const base = (key) => (SHAPES[key] ? key : key.slice(1));
const { views, view, kOf } = createViews({ shapeOf: (key) => SHAPES[base(key)], onProbe: setProbe });
function setProbe(p) { if (JSON.stringify(p) !== JSON.stringify(probe)) { probe = p; renderNet(); } }
const A = 8, Wc = 6; // cell sizes, px: activations, weights (the explorer overview's)
const STAGE_OF = {}; BACK_STAGES.forEach((s, k) => [...s.act, ...s.w].forEach((key) => { STAGE_OF[key] = k; }));
function build() {
  const R = [[], [], []]; // forward, backward, update
  const add = (f, b, u) => { R[0].push(f); R[1].push(b); R[2].push(u); };
  const op = (s, id) => add({ op: s, id }, { op: s ? '←' : '' }, null);
  const act = (key, lb, cell = A, o = {}) => add({ v: [view(key, cell, o)], lb, id: o.id }, key === 'digit' ? null : { v: [view('d' + key, cell)], lb: '∂L/∂' + lb }, null);
  const wts = (keys, lb, cell = Wc, grid) => add(...['', 'd', 'n'].map((ns, row) => ({ v: keys.map((k) => view(ns + k, cell)), w: true, grid, lb: row === 1 ? '∂L/∂' + lb : row === 2 ? lb + '′' : lb })));
  const kernels = (ns) => { const g = el('kg'); g.style.gridTemplateColumns = 'repeat(2, auto)'; for (let d = 0; d < D; d++) g.append(view(ns + 'Wp', Wc, { rows: PS, cols: PS, map: (r, c) => [d, r * PS + c] })); return [g, view(ns + 'bp', Wc)]; };
  const hc = (key, h) => view(key, A, { cols: DH, map: (r, c) => [r, h * DH + c] });
  const hr = (key, h) => view(key, A, { rows: DH, cols: T, map: (r, c) => [c, h * DH + r] });
  act('digit', 'digit', A, { blocks: PS }); op('⋆');
  add(...['', 'd', 'n'].map((ns, row) => ({ v: kernels(ns), w: true, lb: ['kernels', '∂L/∂kernels', 'kernels′'][row] }))); op('=');
  act('xp', 'tokens'); op('+'); wts(['pos'], 'position', A); op('=');
  act('z0', 'z₀', A, { id: 'z0' }); op('LN'); wts(['ln1'], 'LN₁'); op('='); act('h1', 'LN₁(z₀)'); op('·');
  wts(['Wq', 'Wk', 'Wv', 'bq', 'bk', 'bv'], 'Wq Wk Wv', Wc, 3); op('=');
  for (let h = 0; h < 2; h++) {
    add({ v: [hc('q', h)], lb: `Q${h + 1}` }, { v: [hc('dq', h)], lb: `∂L/∂Q${h + 1}` }, null); op('·');
    add({ up: [hr('k', h), hr('v', h)], v: [view('att' + h, A)], lb: `head ${h + 1}` }, { up: [hr('dk', h), hr('dv', h)], v: [view('datt' + h, A)], lb: `∂L/∂A, head ${h + 1}` }, null); op('=');
    add({ v: [hc('o', h)], lb: 'A·V' }, { v: [hc('do', h)], lb: '∂L/∂(A·V)' }, null); op(h ? '·' : '');
  }
  wts(['Wo', 'bo'], 'Wo'); op('+', 'add1');
  act('z1', 'z₁', A, { id: 'z1' }); op('LN'); wts(['ln2'], 'LN₂'); op('='); act('h2', 'LN₂(z₁)'); op('·');
  wts(['W1', 'b1'], 'W₁'); op('='); act('hid', 'hidden'); op('·');
  wts(['W2', 'b2'], 'W₂', 4); op('+', 'add2'); act('z2', 'z₂'); op('mean');
  act('avg', 'average'); op('LN'); wts(['lnf'], 'LN'); op('=');
  act('avgNorm', 'normalised'); op('·'); wts(['Wh', 'bh'], 'W head'); op('=');
  add({ v: [view('prob', 1, { rows: NC, cols: 1, map: (r) => [0, r], bars: true, px: [120, T * A] })], lb: 'classes' },
    { v: [view('dprob', A, { rows: NC, cols: 1, map: (r) => [0, r] })], lb: '∂L/∂scores = p − y' }, null);
  line($('net'), R, A, { labels: ['forward', 'backward', 'update'] });
}
const layout = () => arcs($('net'), [['z0', 'add1'], ['z1', 'add2']]);

function renderNet() {
  if (!cur) return;
  const ph = cur.phase, S = cur.S, wB = wBefore(), wP = ph >= P.B0 ? wPrime() : null, scale = {};
  const shownStage = ph >= P.B0 ? Math.min(ph, P.UPD - 1) - P.B0 : -1; // the last stage revealed
  const revealed = (key) => STAGE_OF[key] <= shownStage, front = (key) => !playing && ph < P.UPD && STAGE_OF[key] === shownStage;
  const img2d = S ? S.img : Array.from({ length: 16 }, (_, r) => Array.from(cur.input.slice(r * 16, r * 16 + 16)));
  const isGrad = probe && probe.t !== base(probe.t) && probe.t[0] === 'd', isW = probe && WEIGHT_KEYS.includes(base(probe.t));
  const green = !probe || !S ? {} : isGrad || isW ? gradTrace(S, probe) : vitTrace(S, probe);
  for (const v of views) {
    const key = kOf(v), b = base(key), ns = key === b ? '' : key[0], weight = WEIGHT_KEYS.includes(b), [R, C] = SHAPES[b];
    let M = null, faint = false;
    if (ns === '') M = b === 'digit' ? img2d : weight ? tensor(b, null, wB) : S ? tensor(b, S, wB) : null;
    else if (revealed(b)) M = ns === 'n' ? tensor(b, null, wP) : weight ? tensor(b, null, shown) : backTensor(b);
    if (!M) { M = Array.from({ length: R }, () => new Array(C).fill(0)); faint = true; }
    const sk = ns === 'd' ? key : b; // w and w′ share a colour scale; each gradient has its own
    const s = b === 'digit' ? 1 : (scale[sk] ??= maxAbs((ns === 'n' ? tensor(b, null, wB) : M).flat()));
    paint(v, M, C, green[key], s, {}, { gray: b === 'digit', faint, outline: ns !== '' && !faint && front(b) });
  }
  $('readout').innerHTML = describe(probe);
}
const backTensor = (b) => { const M = cur.back[b]; return Array.isArray(M[0]) ? M : [M]; };

// ---------- readouts ----------
const NAME = { digit: 'pixel', xp: 'patch token', z0: 'z₀', h1: 'LayerNorm 1 output', q: 'query', k: 'key', v: 'value', att0: 'head 1 attention', att1: 'head 2 attention',
  o: 'heads joined', z1: 'z₁', h2: 'LayerNorm 2 output', hid: 'MLP hidden', z2: 'z₂', avg: 'average', avgNorm: 'normalised average', prob: 'class',
  Wp: 'patch kernels', bp: 'patch bias', pos: 'position table', ln1: 'LayerNorm 1', ln2: 'LayerNorm 2', lnf: 'final LayerNorm', Wq: 'Wq', bq: 'bq', Wk: 'Wk', bk: 'bk',
  Wv: 'Wv', bv: 'bv', Wo: 'Wo', bo: 'bo', W1: 'W₁', b1: 'b₁', W2: 'W₂', b2: 'b₂', Wh: 'W head', bh: 'head bias' };
const HOW = { // how each activation's gradient is computed, from what is downstream of it
  prob: 'p − y: this class\'s probability minus 1 if it is the label, 0 otherwise',
  avgNorm: 'Σ over classes of W head[d, c] × ∂L/∂score(c)', avg: 'the final LayerNorm, backwards, from all 8 of ∂L/∂(normalised average)',
  z2: '∂L/∂average[c] / 16: every token gets a sixteenth of the average\'s gradient', hid: 'Σ over columns of W₂[f, c] × ∂L/∂z₂[i, c]',
  h2: 'Σ over the 32 hidden of W₁[d, f] × GELU′ × ∂L/∂hidden[i, f]', z1: '∂L/∂z₂[i, d] through the residual, plus LayerNorm 2 backwards from ∂L/∂h₂, row i',
  o: 'Σ over columns of Wo[c, d] × ∂L/∂z₁[i, d]', att0: 'Σ over its head\'s 4 columns of ∂L/∂(A·V)[i, c] × value[j, c]', att1: 'Σ over its head\'s 4 columns of ∂L/∂(A·V)[i, c] × value[j, c]',
  v: 'Σ over queries i of A[i, j] × ∂L/∂(A·V)[i, c]: every query that read this token, in proportion to how much it read',
  k: 'Σ over queries i of ∂L/∂s[i, j] × query[i, c] / 2, with ∂L/∂s = A ⊙ (∂L/∂A − Σ A·∂L/∂A), the softmax backwards: a column of the map',
  q: 'Σ over keys j of ∂L/∂s[i, j] × key[j, c] / 2: its own row of the map', h1: '∂L/∂Q · Wqᵀ + ∂L/∂K · Wkᵀ + ∂L/∂V · Wvᵀ, row i',
  z0: '∂L/∂z₁[i, d] through the residual, plus LayerNorm 1 backwards from ∂L/∂h₁, row i', xp: 'equal to ∂L/∂z₀: z₀ = tokens + position' };
const IN_OUT = { Wq: ['LayerNorm 1 output', 'query'], Wk: ['LayerNorm 1 output', 'key'], Wv: ['LayerNorm 1 output', 'value'], Wo: ['heads joined', 'z₁'], W1: ['LayerNorm 2 output', 'hidden'], W2: ['hidden', 'z₂'], Wh: ['normalised average', 'score'] };
function describe(p) {
  if (!p) {
    if (playing) return `Playing at ${speed} digits a second: the band shows a digit at the end of its backward pass${speed > 10 ? `, about one in ${Math.round(speed / 10)}` : ''}. Hover any value to follow it from digit to digit.`;
    if (cur.phase === P.DIGIT) return 'A new digit. Step runs the forward pass.';
    if (cur.phase === P.FWD) return `Forward done: the guess is ${cur.pred}, the label ${cur.label}, the loss −log p(${cur.label}) = ${fmt(cur.loss)}. Step starts the backward pass at the scores.`;
    if (cur.phase < P.UPD) return `Backward through the ${BACK_STAGES[cur.phase - P.B0].name}: the green frames mark what this step computed. Hover any value for how.`;
    return cur.applied ? `Updated: every weight moved by −${lr} × its gradient. Step loads the next digit.` : `Gradient collected (${collected.length}/${batch}); the weights move once the batch is full.`;
  }
  const b = base(p.t), ns = p.t === b ? '' : p.t[0], name = NAME[b] || b, S = cur.S;
  if (WEIGHT_KEYS.includes(b)) {
    const w0 = tensor(b, null, wBefore())[p.r][p.c], g = shown ? tensor(b, null, shown)[p.r][p.c] : null, io = IN_OUT[b];
    const how = io ? `Σ over the 16 tokens of ${io[0]}[t, ${p.r}] × ∂L/∂${io[1]}[t, ${p.c}]` : 'the sum of the gradients it was added to';
    const f4 = (x) => (x === 0 ? '0' : Math.abs(x) < 1e-4 ? x.toExponential(1) : x.toFixed(4)); // a step is often under 0.01
    return `<b>${name} [${p.r}, ${p.c}]</b> w = ${f4(w0)}` + (g == null ? '' : `\n∂L/∂w = ${how} = <span class="sum">${fmt(g)}</span>\nw′ = w − ${lr} × ∂L/∂w = ${f4(w0)} − ${f4(lr * g)} = <span class="sum">${f4(w0 - lr * g)}</span>`);
  }
  if (!S) return 'Step runs the forward pass first.';
  if (ns === 'd') return `<b>∂L/∂${name}</b> [${p.r}, ${p.c}] = <span class="sum">${fmt(backTensor(b)[p.r][p.c])}</span>\n= ${HOW[b]}`;
  const val = b === 'digit' ? S.img[p.r][p.c] : tensor(b, S, wBefore())[p.r][p.c];
  return `<b>${name}</b> [${p.r}, ${p.c}] = ${fmt(val)}` + (b === 'prob' ? ` (${pct(val)})` : '') + '. Green: what it is computed from, upstream.';
}
const STEP_LABEL = ['Step: forward', 'Step: backward from the loss, through the head', 'Step: backward through the MLP', 'Step: backward through attention', 'Step: backward to the embedding', 'Step: update', 'Step: load the next digit'];
function renderPhase() {
  $('phase').querySelectorAll('b').forEach((b) => { const p = +b.dataset.p; b.classList.toggle('done', p <= cur.phase); b.classList.toggle('next', p === cur.phase + 1); });
  $('digitInfo').textContent = `training digit ${cur.i} · label ${cur.label}` + (cur.phase >= P.FWD ? ` · guess ${cur.pred} ${cur.pred === cur.label ? '✓' : '✗'}` : '') + (batch > 1 ? ` · batch ${collected.length}/${batch} collected` : '');
  $('stepBtn').textContent = STEP_LABEL[cur.phase];
}

// ---------- Many steps: curves, snapshots, recent digits ----------
const niceStep = (xmax) => { const raw = Math.max(1, xmax / 5); const p = Math.pow(10, Math.floor(Math.log10(raw))); const m = raw / p; return (m < 2 ? 2 : m < 5 ? 5 : 10) * p; };
/** Lines on a canvas: series [{ pts: [[x, y]], color, width, alpha }], x from 0 to xmax, y from 0 to ymax; recessive axes. */
function curve(cv, series, { xmax, ymax, yfmt, ylines, right = 10 }) {
  const ctx = cv.getContext('2d'), W = cv.width, H = cv.height, L = 38, R = right, Tm = 8, B = 22;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
  const X = (x) => L + (W - L - R) * x / xmax, Y = (y) => H - B - (H - Tm - B) * Math.min(1, y / ymax);
  ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`; ctx.fillStyle = cssVar('--ink2'); ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
  for (const y of ylines) { ctx.beginPath(); ctx.moveTo(L, Y(y) + .5); ctx.lineTo(W - R, Y(y) + .5); ctx.stroke(); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(yfmt(y), L - 5, Y(y)); }
  const st = niceStep(xmax); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let x = 0; x <= xmax; x += st) ctx.fillText(nz(x), X(x), H - B + 5);
  for (const s of series) {
    if (!s.pts.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width; ctx.globalAlpha = s.alpha ?? 1; ctx.beginPath();
    s.pts.forEach(([x, y], k) => (k ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke(); ctx.globalAlpha = 1;
    if (s.dots) { ctx.fillStyle = s.color; s.pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, 7); ctx.fill(); }); }
    if (s.label) { const [x, y] = s.pts[s.pts.length - 1]; ctx.fillStyle = cssVar('--ink2'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(s.label, X(x) + 6, Y(y) + (s.nudge || 0)); }
  }
}
function renderPlay() {
  const xmax = Math.max(500, digitsSeen);
  curve($('cLoss'), [{ pts: lossRaw, color: cssVar('--accent'), width: 1, alpha: 0.25 }, { pts: lossEma, color: cssVar('--accent'), width: 2 }], { xmax, ymax: 3, yfmt: (y) => fmt(y), ylines: [0, 1, 2, 3] });
  curve($('cAcc'), [{ pts: accTrain, color: cssVar('--accent'), width: 1 }, { pts: accTest, color: cssVar('--accent'), width: 2.5, dots: true }], { xmax, ymax: 1, yfmt: (y) => `${y * 100}%`, ylines: [0, 0.25, 0.5, 0.75, 1] });
  $('counts').textContent = `${nz(digitsSeen)} digits seen · pass ${passes + 1} · held-out ${lastTest == null ? '–' : pct(lastTest)}`;
}
function renderSnaps() {
  const box = $('snaps'); box.innerHTML = '';
  for (const s of [...snaps, { ...snapshot('now'), now: true }]) {
    const d = el(''), trio = el('trio');
    for (const A of s.att) { const cv = document.createElement('canvas'); cv.width = cv.height = 64; cv.style.width = cv.style.height = '64px'; drawGrid(cv, A, T, T, { mode: 'div', scale: Math.max(...A), lines: false }); trio.append(cv); }
    const cv = document.createElement('canvas'); cv.width = cv.height = 64; cv.style.width = cv.style.height = '64px'; drawGrid(cv, s.sim, T, T, { mode: 'div', scale: 1, lines: false, blocks: 4 }); trio.append(cv);
    d.append(trio, el('', s.now ? `now (${nz(digitsSeen)})` : `${nz(s.at)} digits`)); box.append(d);
  }
  const th = $('thumbs'); th.innerHTML = '';
  for (const r of recent) {
    const d = el(''), cv = document.createElement('canvas'); cv.width = cv.height = 32; cv.style.width = cv.style.height = '32px';
    drawGrid(cv, r.input, 16, 16, { lines: false });
    const g = el(r.pred === r.label ? 'ok' : 'no', `${r.pred}${r.pred === r.label ? '✓' : '✗'}`); d.append(cv, g); th.append(d);
  }
}
function renderAll() { if (!cur) return; renderPhase(); renderNet(); renderPlay(); renderSnaps(); }

// ---------- hand-off to the explorer ----------
function bundle() {
  const every = [...Array(data.test.n).keys()];
  return { version: 1, arch: 'vit16', weights, meta: { source: 'vit-train', digits: digitsSeen, passes, optimiser: 'sgd', lr, batch,
    testAcc: +accuracy(weights, data.test, every).toFixed(4), testAccNoPos: +accuracy(weights, data.test, every, { position: false }).toFixed(4), date: new Date().toLocaleDateString('en-CA') } };
}
$('useBtn').onclick = () => {
  const o = bundle();
  try { saveTrainedViT(o); $('useMsg').innerHTML = `Stored (held-out ${pct(o.meta.testAcc)}). The <a href="../vit/">ViT explorer</a> will use these weights until you go back to the shipped ones there.`; }
  catch { $('useMsg').textContent = 'Could not store the weights: this browser blocks local storage. Download JSON still works.'; }
};
$('dlBtn').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle())], { type: 'application/json' })); a.download = `vit16-${digitsSeen}-digits.json`; a.click(); URL.revokeObjectURL(a.href); };

// ---------- CNN against ViT: the measured chart and the live race ----------
let measured = null, raceDots = [], race = null, racing = false, raceSeed = 6, raceLast = 0, raceCarry = 0;
const chartDots = []; // for the hover tooltip: { x, y, text }
function drawChart() {
  const cv = $('cChart'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height, L = 44, R = 70, Tm = 12, B = 30;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H); chartDots.length = 0;
  if (!measured) return;
  const lo = Math.log10(measured.sizes[0]), hi = Math.log10(measured.sizes[measured.sizes.length - 1]);
  const X = (n) => L + (W - L - R) * (Math.log10(n) - lo) / (hi - lo), Y = (a) => H - B - (H - Tm - B) * (a - 0.3) / 0.6;
  ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`; ctx.strokeStyle = cssVar('--line'); ctx.fillStyle = cssVar('--ink2'); ctx.lineWidth = 1;
  for (const a of [0.3, 0.45, 0.6, 0.75, 0.9]) { ctx.beginPath(); ctx.moveTo(L, Y(a) + .5); ctx.lineTo(W - R, Y(a) + .5); ctx.stroke(); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(`${Math.round(a * 100)}%`, L - 6, Y(a)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const n of measured.sizes) ctx.fillText(`${nz(n)} digits`, X(n), H - B + 8);
  const surface = cssVar('--panel'), mean = (r) => r.reduce((a, b) => a + b, 0) / r.length;
  const above = mean(measured.cnn.at(-1)) >= mean(measured.vit.at(-1)) ? 'cnn' : 'vit';
  for (const [kind, k] of [['cnn', -1], ['vit', 1]]) {
    const color = cssVar('--' + kind), rows = measured[kind], means = rows.map(mean);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); means.forEach((m, i) => (i ? ctx.lineTo(X(measured.sizes[i]) + k * 7, Y(m)) : ctx.moveTo(X(measured.sizes[i]) + k * 7, Y(m)))); ctx.stroke();
    rows.forEach((r, i) => r.forEach((a, s) => {
      const x = X(measured.sizes[i]) + k * 7 + (s - 2) * 2.5, y = Y(a);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke();
      chartDots.push({ x, y, text: `${kind === 'cnn' ? 'CNN' : 'ViT'}, ${nz(measured.sizes[i])} digits, seed ${measured.seeds[s]}: ${pct(a)}` });
    }));
    means.forEach((m, i) => chartDots.push({ x: X(measured.sizes[i]) + k * 7, y: Y(m), text: `${kind === 'cnn' ? 'CNN' : 'ViT'} average over ${measured.seeds.length} seeds, ${nz(measured.sizes[i])} digits: ${pct(m)}` }));
    const last = means.length - 1; ctx.fillStyle = cssVar('--ink2'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(kind === 'cnn' ? 'CNN' : 'ViT', X(measured.sizes[last]) + 18, Y(means[last]) + (kind === above ? -8 : 8));
  }
  for (const d of raceDots) for (const [kind, k] of [['cnn', -1], ['vit', 1]]) {
    const x = X(d.n) + k * 7, y = Y(d[kind]);
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.lineWidth = 2; ctx.strokeStyle = cssVar('--' + kind); ctx.stroke();
    chartDots.push({ x, y, text: `this page's race, ${kind === 'cnn' ? 'CNN' : 'ViT'}, ${nz(d.n)} digits, seed ${d.seed}: ${pct(d[kind])}` });
  }
}
function chartNote() {
  if (!measured) return 'Could not load models/vit_vs_cnn.json.';
  const mean = (r) => r.reduce((a, b) => a + b, 0) / r.length;
  const parts = measured.sizes.map((n, i) => {
    const c = measured.cnn[i], v = measured.vit[i], ahead = c.filter((a, s) => a > v[s]).length;
    return `${nz(n)} digits: CNN ${pct(mean(c))}, ViT ${pct(mean(v))}, the CNN ahead in ${ahead} of ${c.length} seeds`;
  });
  return `Averages over ${measured.seeds.length} seeds (${measured.date}): ${parts.join('; ')}. With few digits the CNN's locality wins clearly; from about 1,000 digits the two are level within the spread of the seeds, and single runs scatter by 10 points or more. Hover a dot for its run.`;
}
$('cChart').addEventListener('pointermove', (ev) => {
  const b = ev.target.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top, tip = $('ctip');
  const hit = chartDots.map((d) => ({ d, r: Math.hypot(d.x - x, d.y - y) })).sort((a, c) => a.r - c.r)[0];
  if (hit && hit.r < 9) { tip.textContent = hit.d.text; tip.style.display = 'block'; tip.style.left = ev.clientX + 12 + 'px'; tip.style.top = ev.clientY + 12 + 'px'; } else tip.style.display = 'none';
});
$('cChart').addEventListener('pointerleave', () => { $('ctip').style.display = 'none'; });

function newRace() {
  const n = +$('raceN').value, idx = balancedSubset(data.train, n);
  race = { n, seed: raceSeed, cnn: racer('cnn', data.train, idx, raceSeed), vit: racer('vit', data.train, idx, raceSeed), pts: { cnn: [], vit: [] }, done: false };
  raceEval(); renderRace();
}
function raceEval() { for (const kind of ['cnn', 'vit']) race.pts[kind].push([race[kind].steps, race[kind].accuracy(data.test, testIdx)]); }
function raceFrame(ts) {
  if (!racing) return;
  const sp = +$('raceSpeed').value, dt = raceLast ? Math.min(0.1, (ts - raceLast) / 1000) : 0; raceLast = ts;
  raceCarry = Math.min(raceCarry + sp * dt, Math.max(1, sp / 10));
  const t0 = performance.now();
  while (raceCarry >= 1 && !race.done && performance.now() - t0 < 12) {
    race.cnn.step(); race.vit.step(); raceCarry -= 1;
    if (race.cnn.steps % 2000 === 0) raceEval();
    if (race.cnn.steps >= RACE.steps) { race.done = true; const last = (k) => race.pts[k][race.pts[k].length - 1][1]; raceDots.push({ n: race.n, seed: race.seed, cnn: last('cnn'), vit: last('vit') }); drawChart(); setRacing(false); }
  }
  renderRace();
  if (racing) requestAnimationFrame(raceFrame);
}
function setRacing(r) {
  if (r && race.done) { raceSeed++; newRace(); }
  racing = r; $('racePlay').textContent = r ? 'Pause' : race && race.cnn.steps > 0 && !race.done ? 'Resume' : 'Race';
  if (r) { if (playing) setPlaying(false); raceLast = 0; raceCarry = 0; requestAnimationFrame(raceFrame); }
}
function renderRace() {
  const lab = (k) => (race.pts[k].length ? `${k === 'cnn' ? 'CNN' : 'ViT'} ${pct(race.pts[k][race.pts[k].length - 1][1])}` : '');
  const hiK = race.pts.cnn.length && race.pts.cnn.at(-1)[1] >= race.pts.vit.at(-1)[1] ? 'cnn' : 'vit';
  curve($('cRace'), ['cnn', 'vit'].map((k) => ({ pts: race.pts[k], color: cssVar('--' + k), width: 2, dots: true, label: lab(k), nudge: k === hiK ? -7 : 7 })), { xmax: RACE.steps, ymax: 1, yfmt: (y) => `${y * 100}%`, ylines: [0, 0.25, 0.5, 0.75, 1], right: 80 });
  $('raceCounts').textContent = `${nz(race.cnn.steps)} of ${nz(RACE.steps)} steps on ${nz(race.n)} digits · seed ${race.seed}`;
  $('raceNote').textContent = race.done
    ? `This race (seed ${race.seed}, ${nz(race.n)} digits): CNN ${pct(race.pts.cnn.at(-1)[1])}, ViT ${pct(race.pts.vit.at(-1)[1])}, now hollow dots on the chart above: one more run among the measured ones. Race again for a new seed.`
    : `Both models train on the same ${nz(race.n)} digits in the same order, each at its own learning rate (CNN ${RACE.lr.cnn}, ViT ${RACE.lr.vit}); held-out accuracy every 2,000 steps. One run is one dot: at 100 digits the CNN won in every measured seed, at 1,000 and above either can win.`;
}
$('racePlay').onclick = () => setRacing(!racing);
$('raceReset').onclick = () => { setRacing(false); raceSeed++; newRace(); $('racePlay').textContent = 'Race'; };
$('raceN').onchange = () => { setRacing(false); newRace(); $('racePlay').textContent = 'Race'; };

// ---------- controls ----------
const lrOf = (v) => +(0.0003 * Math.pow(1000, v / 100)).toPrecision(1);
$('stepBtn').onclick = () => { if (playing) setPlaying(false); step(); };
$('nextBtn').onclick = () => { if (playing) setPlaying(false); nextDigit(); };
$('playBtn').onclick = () => setPlaying(!playing);
$('resetBtn').onclick = () => { setPlaying(false); reset({ newSeed: true }); };
$('speed').onchange = (e) => { speed = +e.target.value; };
$('lr').oninput = (e) => { lr = lrOf(+e.target.value); $('lrv').textContent = String(lr); if (!playing) renderNet(); };
$('batch').onchange = (e) => { batch = +e.target.value; collected = []; shown = null; if (cur && cur.phase >= P.B0 && cur.phase < P.UPD) cur.phase = P.FWD; renderAll(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderAll(); drawChart(); if (race) renderRace(); });
addEventListener('resize', layout);

// ---------- start ----------
async function start() {
  installTerms({ selectors: '.lede, .head p, .note:not(#status):not(#chartNote):not(#raceNote)', tip: $('tip'), gloss: $('gloss'), terms: VIT_TRAIN_TERMS });
  const [bin, js] = await Promise.all([
    fetch('../data/mnist16.bin').then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null),
    fetch('../models/vit_vs_cnn.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  if (!bin) { $('status').textContent = 'Could not load the digits (data/mnist16.bin); this page needs to be served over http.'; return; }
  data = parseMnist16(bin); testIdx = [...Array(data.test.n).keys()];
  fixedImg = image(data.test, testIdx.find((i) => data.test.labels[i] === 7));
  measured = js && js.version === 1 ? js : null;
  $('status').textContent = `${nz(data.train.n)} training and ${nz(data.test.n)} held-out digits loaded.`;
  build(); reset(); layout(); document.fonts.ready.then(layout);
  drawChart(); $('chartNote').textContent = chartNote(); newRace();
}
const ready = start();
</script>
</body>
</html>
```

- [ ] **Step 3: The training-page scenario**

Run: `node .superpowers/check/vitcheck.mjs . .superpowers/check/scen_train.js --page vit-train --width 1640 --height 1000`
Expected JSON, as in the prototype:
- **The page:** `"views": 122`, `"overflow": 0`, `"glossary": 28`, `"dfn": 34`, `"arcs": true`.
- **`"phases"`:** seven `[phase, faint views, framed views, button]` entries, with faint and framed counts going 94/0, 74/0, 64/10, 51/13, 24/27, 0/24, 0/0. The buttons run "Step: forward", "Step: backward from the loss, through the head", …, "Step: load the next digit".
- **Hover traces:**
  - `"dvLit": ["att0","davg","davgNorm","dh2","dhid","do","dprob","dv","dz1","dz2"]`;
  - `"wLit": ["Wq","dWq","dq","h1","nWq"]`;
  - `"wReadout"` writes w, ∂L/∂w as a sum over the 16 tokens, and w′ with 4 decimals.
- **Play:** `"playEqualsLoop": true` (300 digits of play equal a plain `core/vit.js` loop over the same digits), `"snaps": [0,300]`, `"snapCanvases": 9`, `"thumbs": 20`.
- **The comparison:** `"measured": [4,5]`; `"chartNote"` starting "Averages over 5 seeds (…): 100 digits: CNN 72.6%, ViT 45.5%, the CNN ahead in 5 of 5 seeds"; `"raceDone": true`, `"raceDots": 1`, `"chartDots": 50`.
- **The hand-off:** `"stored": true`.
- `"consoleErrors": []`.

- [ ] **Step 4: Dark, phone, screenshots**

Run:
```bash
C=.superpowers/check
node $C/vitcheck.mjs . $C/scen_load.js --page vit-train --dark --width 1000 --height 1400 --shot $C/train_dark.png
node $C/vitcheck.mjs . $C/scen_load.js --page vit-train --width 500 --height 1400 --shot $C/train_phone.png
node $C/vitcheck.mjs . $C/scen_train.js --page vit-train --width 1640 --height 3200 --shot $C/train_tall.png
```
Expected: `"overflow": 0` in each, `"measured": true`, and no console errors. Open the three PNGs with the Read tool and check:
- **One step:** three rows labelled forward, backward and update, with ← between backward columns. The forward row has the arc under the labels. Faint cells appear before their stage; green frames sit on the stage just computed.
- **Many steps:** the curves, the snapshot trios (two attention maps and a similarity picture with dashed 4×4 blocks), the recent digits.
- **CNN against ViT:**
  - the legend; dots per seed; mean lines, with CNN above at 100 and 300 digits and the two close at 1,000 and 3,000; the direct labels "CNN" and "ViT"; the hollow race dots;
  - the race chart with two labelled lines;
  - the dark colours readable, and the lede's link in the accent colour.

Fix anything that reads wrong with a minimal change, and rerun Steps 3 and 4.

- [ ] **Step 5: Commit**

```bash
git add vit-train/index.html
git commit -m "feat(vit-train): the ViT training page" -m "One step with the backward pass through attention a stage at a time; many steps with attention and position snapshots; CNN against ViT, measured over 5 seeds, and a live race." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Landing page, documentation, final checks

**Files:**
- Modify: `index.html` (edit script), `CLAUDE.md`, `HANDOFF.md`, `project_notes.md`

- [ ] **Step 1: The landing page**

Create `.superpowers/check/landing_train.py`:

```python
# Landing page: the ViT section gains its Training card (index.html). Each original must occur exactly once.
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
reps = [
('<p class="lede">The convolutional network is built, and the Vision Transformer has its explorer. Next: the Vision Transformer\'s training page, then U-Net, CoAtNet and ConvNeXt.</p>',
 '<p class="lede">The convolutional network and the Vision Transformer are built, each with an explorer and a training page. Next: U-Net, CoAtNet and ConvNeXt.</p>'),
('<a href="#vit">Vision Transformer <span class="badge live">1 page</span></a>', '<a href="#vit">Vision Transformer <span class="badge live">2 pages</span></a>'),
('<h2>Vision Transformer (ViT)</h2><span class="badge live">1 page</span>', '<h2>Vision Transformer (ViT)</h2><span class="badge live">2 pages</span>'),
('''    <div class="cards one">
      <a class="card" href="vit/">''', '''    <div class="cards">
      <a class="card" href="vit/">'''),
('''then the MLP, the residuals and the average. Trained weights; hover any value to see what it reads.</span><em>Open →</em></a>
    </div>''', '''then the MLP, the residuals and the average. Trained weights; hover any value to see what it reads.</span><em>Open →</em></a>
      <a class="card" href="vit-train/"><b><svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><path d="M3 4C7 5 8 15 12 17S20 20 23 21" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linecap="round"/><path d="M3 23H23M3 3V23" stroke="var(--ink2)" stroke-width="1.2" fill="none"/></svg>Training</b>
        <span>One step in full, the backward pass through attention a stage at a time, then thousands of steps; then a race against the CNN on 100 to 3,000 digits, where the CNN's locality wins when data is scarce.</span><em>Open →</em></a>
    </div>'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('landing page edited')
```

Run: `python .superpowers/check/landing_train.py index.html`. Expected: `landing page edited`.

Screenshot the ViT section at 1400 px, in light and dark, as in the previous plan's Task 6 (serve with `python -m http.server`, capture with Edge's `--screenshot`, and make a `data-theme="dark"` copy for dark). Check:
- the ViT section's badge reads "2 pages" and it shows two cards, Explorer and Training, side by side;
- the chip at the top reads "2 pages";
- the lede says both architectures are built, with U-Net, CoAtNet and ConvNeXt next.

- [ ] **Step 2: `CLAUDE.md`**

Make these replacements with the Edit tool.

```text
(CNN built; ViT explorer built; then the ViT training page, U-Net, CoAtNet, ConvNeXt, in that order)
```
with
```text
(CNN and ViT built, each with an explorer and a training page; then U-Net, CoAtNet, ConvNeXt, in that order)
```

```text
Tests: `node core/cnn.test.mjs`, `node core/data.test.mjs`, `node core/train.test.mjs`, `node core/vit.test.mjs`.
```
with
```text
Tests: `node core/cnn.test.mjs`, `node core/data.test.mjs`, `node core/train.test.mjs`, `node core/vit.test.mjs`, `node core/race.test.mjs`. `core/race.js` steps the CNN and the ViT on the same digits, for both the offline comparison and the live race.
```

```text
`trained.js` (trained-weights hand-off through local storage);
```
with
```text
`trained.js` (trained-weights hand-off through local storage, a key for the CNN and one for the ViT); `vitline.js` and `vitline.css` (the ViT pages' tensor drawing: views, lines of one or several rows, arcs, paint);
```

```text
and `train_vit.mjs`, which trains the ViT and writes `models/vit16.json`;
```
with
```text
`train_vit.mjs`, which trains the ViT and writes `models/vit16.json`, and `vit_vs_cnn.mjs`, which measures both models on 100 to 3,000 digits and writes `models/vit_vs_cnn.json`;
```

```text
- `cnn-2d/`, `cnn-3d/`, `cnn-train/`, `vit/` — one self-contained page each
```
with
```text
- `cnn-2d/`, `cnn-3d/`, `cnn-train/`, `vit/`, `vit-train/` — one self-contained page each
```

```text
New pages follow the same pattern (`vit-train/`, `unet/`, `coatnet/`, `convnext/`).
```
with
```text
New pages follow the same pattern (`unet/`, `coatnet/`, `convnext/`).
```

```text
- `models/` — trained weights: `vit16.json`, the ViT page's, made by `node tools/train_vit.mjs` (the command is in its `meta.made`).
```
with
```text
- `models/` — `vit16.json`, the ViT explorer's weights, made by `node tools/train_vit.mjs` (the command is in its `meta.made`); `vit_vs_cnn.json`, the measured comparison on the ViT training page, made by `node tools/vit_vs_cnn.mjs`.
```

```text
After attention (ViT page) green is graded:
```
with
```text
On the ViT training page a gradient's green runs downstream and, through attention, is graded by a column of the map (the queries that read a token), the mirror of the forward row. After attention (ViT page) green is graded:
```

- [ ] **Step 3: `HANDOFF.md`**

```text
its training page `vit-train/` is next._
```
with
```text
its training page `vit-train/` followed the same day: one step with the backward pass through attention, many steps, and the CNN-against-ViT comparison._
```

```text
    node core/vit.test.mjs
```
with
```text
    node core/vit.test.mjs
    node core/race.test.mjs
```

- [ ] **Step 4: `project_notes.md`, with the `project-notes` skill**

Invoke the `project-notes` skill and update "As built" only. Never touch Intent.
- **Header:** last updated 2026-10-28, the ViT training page built.
- **Overview:** five pages now; the ViT has an explorer and a training page.
- **Method:**
  - a ViT training page bullet: the three bands, the stages, the column-graded backward green, SGD with its slider range, the snapshots, the hand-off, the measured chart and the race;
  - `core/race.js`, `tools/vit_vs_cnn.mjs`, `ui/vitline.*`;
  - `back`, `nudge`, `gradTrace` and `BACK_STAGES` in `core/vit.js`.
- **Results**, dated 2026-10-28, laptop, Node 24.19.0, headless Edge:
  - Task 1's test lines;
  - the measurement table from Task 2;
  - the position-table finding: correlation of grid distance with cosine similarity +0.03 at the start, +0.19 for the shipped weights, +0.06 for Adam weights; neighbours −0.06 against far slots +0.08;
  - the shipped weights' 82.0% is seed 1's: seeds 2 and 3, under the same settings, reached 72.3% and 70.1% (single runs, 30,000 steps, 3,000 digits);
  - the page checks from Task 5;
  - `vit/` unchanged through the refactor;
  - the CNN DOM still identical.
- **Decisions**, with their why, from the spec: the three bands; the measured chart plus the live race; snapshots of attention and of the position table; 2D only; SGD.
- **Map:** the new files, the spec and this plan.
- **Status:** the ViT is complete for now (explorer and training page); next, U-Net.
- **Open questions:**
  - Replace the data-hunger question with its answer: with 100 digits the CNN leads by about 27 points in every seed; from 1,000 digits the two are level.
  - Keep "what each head attends to".
  - Add: a position table with grid structure would need far more data (60,000 MNIST digits, or augmentation); untested.

- [ ] **Step 5: Final checks**

Run:
```bash
node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs && node core/vit.test.mjs && node core/race.test.mjs
C=.superpowers/check
node $C/vitcheck.mjs . $C/scen_basic.js
node $C/vitcheck.mjs . $C/scen_train.js --page vit-train --width 1640 --height 1000
node $C/domcheck.mjs . $C/dom-final && diff -r $C/dom-before-t4 $C/dom-final && echo IDENTICAL
```
Expected:
- all five test files pass;
- both scenarios are as in Tasks 3 and 5, with no console errors;
- the DOM check prints `IDENTICAL`.

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "feat: landing page card for the ViT training page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add CLAUDE.md HANDOFF.md project_notes.md
git commit -m "docs: record the ViT training page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git status --short
```
Expected: a clean working tree. Do not push: the controller merges into `main` and pushes after the final review.

---

## Plan self-review

- **Spec coverage:**
  - Part 1 (phases, three rows, the shared builder, the column-graded backward green, readouts): Tasks 1, 3 and 5.
  - Part 2 (controls, the band following play, curves, snapshots, recent digits, the hand-off): Tasks 4 and 5.
  - Part 3 (`tools/vit_vs_cnn.mjs`, the chart, the race): Tasks 2 and 5.
  - Part 4 (`back`, `gradTrace`, `BACK_STAGES`, `ui/vitline.js`, `ui/trained.js`, the landing page, the notes corrections, the checks): Tasks 1 to 6.
- **Two deviations from the spec, by design:**
  - The ViT training page draws the LayerNorm outputs h₁ and h₂ as columns. The explorer's overview treats LayerNorm as an operator, but here they are the input columns of Wq, Wk, Wv and W₁, and part of the backward chain.
  - The race evaluates every 2,000 steps, not 500: each evaluation of both models over the 1,000 held-out digits takes about 0.6 s.
- **Placeholders:** none. Task 2's commit body has its expected figures, with an instruction for the case where they differ.
- **Names are consistent across tasks:** `back` keys = `ACTIVATIONS` minus `digit`; `gradTrace` namespaces `d`/`n`; `BACK_STAGES` names; `RACE`; `racer`; `saveTrainedViT`/`loadTrainedViT`; `VIT_TRAIN_TERMS`; `createViews`/`line`/`arcs`/`paint`.

