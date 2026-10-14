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

/** Random weights, He-uniform (±√(6/fan_in)), biases 0. layer1 'edges' uses the hand-picked bank instead. */
export function initWeights({ seed = 1, layer1 = 'random', C1 = 4, C2 = 8, NC = 10 } = {}) {
  const rnd = mulberry32(seed);
  const uni = (n, s) => Array.from({ length: n }, () => (rnd() * 2 - 1) * s);
  const bank = layer1 === 'edges' ? defaultBank.map((k) => k.slice()) : Array.from({ length: C1 }, () => uni(9, Math.sqrt(6 / 9)));
  const l2 = Array.from({ length: C2 }, () => Array.from({ length: C1 }, () => uni(9, Math.sqrt(6 / (9 * C1)))));
  const W = Array.from({ length: C2 }, () => uni(NC, Math.sqrt(6 / C2)));
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

/**
 * Where the gradient at one activation cell was computed from: the cells of the later gradient maps it is a sum
 * over, stage by stage up to the scores. It is the backward pass's mirror of the receptive field: a receptive field
 * grows towards the input, this grows towards the loss. probe { stage, r, c, ch } with stage conv1, pool1, conv2,
 * pool2, gap (r, c unused, ch the average) or scores (c the class). Returns { stage: { rect: [r0, r1, c0, c1],
 * chs } } as footprint() in core/cnn.js does; chs null means every channel. Stride 1, padding 1, 2×2 pools.
 * Whether a conv cell actually received anything (it was the pool's maximum, its ReLU was on) is a matter of the
 * values, not of the wiring; this gives the wiring.
 */
export function gradFootprint(probe, S) {
  const R = {}; if (!probe) return R;
  const { Q } = S, NC = S.prob.length, cl = (v, n) => Math.max(0, Math.min(n - 1, v));
  let st = probe.stage, chs = probe.ch == null ? null : new Set([probe.ch]);
  let rect = st === 'gap' ? [0, 0, 0, 0] : st === 'scores' ? [0, 0, probe.c, probe.c] : [probe.r, probe.r, probe.c, probe.c];
  R[st] = { rect, chs };
  const next = { conv1: 'pool1', pool1: 'conv2', conv2: 'pool2', pool2: 'gap', gap: 'scores' };
  while (next[st]) {
    const [r0, r1, c0, c1] = rect;
    if (st === 'conv1' || st === 'conv2') rect = [r0 >> 1, r1 >> 1, c0 >> 1, c1 >> 1]; // the pooled cell it fed, same channel
    else if (st === 'pool1') { rect = [cl(r0 - 1, Q), cl(r1 + 1, Q), cl(c0 - 1, Q), cl(c1 + 1, Q)]; chs = null; } // the 3×3 window of every conv-2 map
    else if (st === 'pool2') rect = [0, 0, 0, 0]; // its channel's average
    else { rect = [0, 0, 0, NC - 1]; chs = null; } // gap: every score
    st = next[st]; R[st] = { rect, chs };
  }
  return R;
}

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
