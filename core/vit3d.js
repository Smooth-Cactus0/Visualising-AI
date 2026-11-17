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
