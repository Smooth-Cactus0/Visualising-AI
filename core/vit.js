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

/** Every intermediate of one forward pass. position false leaves the position table out; order shuffles the patches.
 *  nudge { t, r, c, by } adds `by` to one cell of an activation as soon as it is computed (t a SHAPES activation key,
 *  or 'logits'); the tests use it to check each activation's gradient against finite differences. */
export function forward(w, img, { position = true, order = null, nudge = null } = {}) {
  const bump = (key, M) => { if (nudge && nudge.t === key) { if (Array.isArray(M[0])) M[nudge.r][nudge.c] += nudge.by; else M[nudge.c] += nudge.by; } return M; };
  const patches = patchify(img, order);
  const xp = bump('xp', patches.map((px) => w.Wp.map((k, d) => dot(px, k) + w.bp[d])));
  const z0 = bump('z0', xp.map((t, i) => t.map((v, d) => v + (position ? w.pos[i][d] : 0))));
  const n1 = z0.map((t) => layerNorm(t, w.ln1)), h1 = bump('h1', n1.map((n) => n.y));
  const q = bump('q', linear(h1, w.Wq, w.bq)), k = bump('k', linear(h1, w.Wk, w.bk)), v = bump('v', linear(h1, w.Wv, w.bv));
  const cols = (X, h) => X.map((r) => r.slice(h * DH, h * DH + DH));
  const scores = [], att = [], heads = [];
  for (let h = 0; h < H; h++) {
    const qh = cols(q, h), kh = cols(k, h), vh = cols(v, h);
    const s = qh.map((qi) => kh.map((kj) => dot(qi, kj) / Math.sqrt(DH)));
    const a = bump('att' + h, s.map(softmax));
    scores.push(s); att.push(a);
    heads.push(a.map((ai) => Array.from({ length: DH }, (_, c) => ai.reduce((t, aij, j) => t + aij * vh[j][c], 0))));
  }
  const o = bump('o', q.map((_, i) => heads.flatMap((hh) => hh[i])));
  const mixed = linear(o, w.Wo, w.bo);
  const z1 = bump('z1', z0.map((t, i) => t.map((x, d) => x + mixed[i][d])));
  const n2 = z1.map((t) => layerNorm(t, w.ln2)), h2 = bump('h2', n2.map((n) => n.y));
  const pre = linear(h2, w.W1, w.b1), hid = bump('hid', pre.map((r) => r.map(gelu)));
  const mlp = linear(hid, w.W2, w.b2);
  const z2 = bump('z2', z1.map((t, i) => t.map((x, d) => x + mlp[i][d])));
  const avg = bump('avg', Array.from({ length: D }, (_, d) => z2.reduce((t, r) => t + r[d], 0) / T));
  const nf = layerNorm(avg, w.lnf), avgNorm = bump('avgNorm', nf.y);
  const logits = bump('logits', w.bh.map((b, c) => avgNorm.reduce((t, x, d) => t + x * w.Wh[d][c], b)));
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

/** Loss and the gradient of every weight (same structure as the weights) for one digit, and `back`: the gradient of
 *  every drawn activation, keyed as SHAPES (`prob` holds ∂L/∂(scores) = p − y, drawn under the probabilities). */
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
  const dQ = zeros(T, D), dK = zeros(T, D), dV = zeros(T, D), sc = 1 / Math.sqrt(DH), dAtt = [];
  for (let h = 0; h < H; h++) {
    const A = S.att[h], off = h * DH;
    dAtt.push([]);
    for (let i = 0; i < T; i++) {
      const dA = new Array(T); dAtt[h].push(dA);
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
  const dH1 = dHq.map((r, i) => r.map((x, d) => x + dHk[i][d] + dHv[i][d]));
  const dZ0 = dZ1.map((r, i) => { const dx = lnBack(dH1[i], S.n1[i], w.ln1.g, g.ln1.g, g.ln1.b); return r.map((x, d) => x + dx[d]); });
  // position and patch embedding
  if (S.position) dZ0.forEach((r, i) => r.forEach((x, d) => { g.pos[i][d] += x; }));
  for (let p = 0; p < T; p++) for (let d = 0; d < D; d++) {
    const x = dZ0[p][d]; g.bp[d] += x;
    for (let q2 = 0; q2 < P * P; q2++) g.Wp[d][q2] += x * S.patches[p][q2];
  }
  const back = {
    prob: dLogits, avgNorm: dAvgNorm, avg: dAvg, z2: dZ2, hid: dHid, h2: dH2, z1: dZ1, o: dO, att0: dAtt[0], att1: dAtt[1],
    q: dQ, k: dK, v: dV, h1: dH1, z0: dZ0, xp: dZ0, // z₀ = patch tokens + position, so both get the same gradient
  };
  return { S, loss, grads: g, back };
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
    case 'prob': put('prob', 0, c, 1); break; // p − y: this class's probability (the label is not a drawn tensor)
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
