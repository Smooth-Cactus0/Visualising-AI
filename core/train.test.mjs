import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { mulberry32, initWeights, crossEntropy, lossAndGrad, sgdStep, averageGrads, predict, accuracy, shuffledOrder } from './train.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

// ---- gradients against central finite differences, on a small random instance (8×8 input) ----
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

// ---- SGD on one real digit: the loss must fall well below its start ----
const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
{
  let w = initWeights({ seed: 1 });
  const x = image(data.train, 0), y = data.train.labels[0];
  const l0 = lossAndGrad(w, x, y).loss;
  for (let t = 0; t < 50; t++) w = sgdStep(w, lossAndGrad(w, x, y).grads, 0.1);
  const l1 = lossAndGrad(w, x, y).loss;
  assert(l1 < 0.5 * l0, `loss after 50 steps ${l1} vs start ${l0}`);
  assert(w.bank[0] !== initWeights({ seed: 1 }).bank[0] && Number.isFinite(w.W[0][0]), 'fresh arrays');
  console.log(`one-digit SGD: loss ${l0.toFixed(3)} → ${l1.toFixed(3)} after 50 steps`);
}
// ---- averaging two gradients ----
{
  const w = initWeights({ seed: 1 });
  const g1 = lossAndGrad(w, image(data.train, 0), data.train.labels[0]).grads, g2 = lossAndGrad(w, image(data.train, 1), data.train.labels[1]).grads;
  const g = averageGrads([g1, g2]);
  assert(Math.abs(g.W[2][3] - (g1.W[2][3] + g2.W[2][3]) / 2) < 1e-12 && Math.abs(g.bank[1][4] - (g1.bank[1][4] + g2.bank[1][4]) / 2) < 1e-12 && Math.abs(g.l2[5][2][8] - (g1.l2[5][2][8] + g2.l2[5][2][8]) / 2) < 1e-12 && Math.abs(g.B[9] - (g1.B[9] + g2.B[9]) / 2) < 1e-12, 'averaged gradients');
}
// ---- shuffled order is a permutation and depends on the seed ----
{
  const o = shuffledOrder(3000, 1);
  assert(o.length === 3000 && new Set(o).size === 3000 && Math.min(...o) === 0 && Math.max(...o) === 2999, 'permutation');
  assert(shuffledOrder(3000, 1).join() === o.join() && shuffledOrder(3000, 2).join() !== o.join(), 'seeded shuffle');
}
// ---- three passes over the training digits must beat 75% on the held-out digits ----
{
  const testIdx = Array.from({ length: data.test.n }, (_, i) => i);
  let w = initWeights({ seed: 1 });
  const before = accuracy(w, data.test, testIdx);
  const t0 = Date.now();
  const per = [];
  for (let pass = 0; pass < 3; pass++) { for (const i of shuffledOrder(data.train.n, 2 + pass)) w = sgdStep(w, lossAndGrad(w, image(data.train, i), data.train.labels[i]).grads, 0.1); per.push(accuracy(w, data.test, testIdx)); }
  const ms = Date.now() - t0;
  const after = per[per.length - 1];
  console.log(`three passes (3 × 3000 digits, lr 0.1, batch 1): held-out accuracy ${(before * 100).toFixed(1)}% → ${per.map((a) => (a * 100).toFixed(1)).join('% → ')}% in ${ms} ms`);
  assert(after > 0.75, 'held-out accuracy after three passes: ' + after);
  const p = predict(w, image(data.test, 0));
  assert(p.prob.length === 10 && p.argmax === p.prob.indexOf(Math.max(...p.prob)), 'predict');
}
console.log('core/train.js OK');
