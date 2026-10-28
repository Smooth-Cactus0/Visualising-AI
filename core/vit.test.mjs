import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { shuffledOrder } from './train.js';
import { N, P, T, D, H, DH, HID, NC, SHAPES, init, forward, tensor, paramCount, isWeights, crossEntropy, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, predict, accuracy, vitTrace, gradTrace, ACTIVATIONS, WEIGHT_KEYS, BACK_STAGES } from './vit.js';
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
console.log('core/vit.test.mjs: all checks passed');
