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
