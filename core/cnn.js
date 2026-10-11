// Shared model core for the explorers. Pure functions, no DOM, no three.js.
// Both pages currently carry their own inline copy of this logic; wiring them
// to this module is task 1 in HANDOFF.md.

export const K = 3;

export function outSize(n, { stride = 1, pad = 1, dil = 1, k = K } = {}) {
  return Math.floor((n + 2 * pad - dil * (k - 1) - 1) / stride) + 1;
}

/** 2-D convolution of a single-channel n×n map with a k×k kernel (row-major). Zero padding. */
export function conv2d(src, n, ker, { stride = 1, pad = 1, dil = 1, k = K } = {}) {
  const O = outSize(n, { stride, pad, dil, k });
  const out = new Float32Array(O * O);
  const g = (r, c) => (r >= 0 && c >= 0 && r < n && c < n ? src[r * n + c] : 0);
  for (let i = 0; i < O; i++)
    for (let j = 0; j < O; j++) {
      let s = 0;
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++) s += g(i * stride + a * dil - pad, j * stride + b * dil - pad) * ker[a * k + b];
      out[i * O + j] = s;
    }
  return { out, O };
}

export const relu = (m) => m.map((v) => Math.max(0, v));

/** 2×2 / stride-2 pooling. mode: 'max' | 'avg'. Trailing row/col dropped when O is odd. */
export function pool2(m, O, mode = 'max') {
  const Q = Math.max(1, Math.floor(O / 2));
  const out = new Float32Array(Q * Q);
  for (let i = 0; i < Q; i++)
    for (let j = 0; j < Q; j++) {
      const v = [];
      for (const [r, c] of [[2 * i, 2 * j], [2 * i, 2 * j + 1], [2 * i + 1, 2 * j], [2 * i + 1, 2 * j + 1]])
        if (r < O && c < O) v.push(m[r * O + c]);
      out[i * Q + j] = mode === 'max' ? Math.max(...v) : v.reduce((a, b) => a + b, 0) / v.length;
    }
  return { out, Q };
}

export function softmax(logits) {
  const mx = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - mx));
  const Z = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / Z);
}

/**
 * Full forward pass of the toy network:
 *   input N×N×1 → conv1 (C1 kernels, stride/pad/dil) + ReLU → pool → conv2 (C2 kernels 3×3×C1, s1 p1) + ReLU → pool
 *   → global average → FC (C2×NC + NC) → softmax
 * params: { input, N, bank (C1 kernels of 9), l2 (C2 × C1 × 9), W (C2 × NC), B (NC), stride, pad, dil, poolMode }
 * Returns every intermediate tensor; channel maps are arrays of Float32Array.
 */
export function forward({ input, N, bank, l2, W, B, stride = 1, pad = 1, dil = 1, poolMode = 'max' }) {
  const S = { N, stride, pad, dil };
  const opts = { stride, pad, dil };
  S.c1 = bank.map((k) => conv2d(input, N, k, opts).out);
  S.O = outSize(N, opts);
  S.a1 = S.c1.map(relu);
  S.p1 = S.a1.map((m) => pool2(m, S.O, poolMode).out);
  S.Q = pool2(S.a1[0], S.O, poolMode).Q;
  // layer 2: one slice per input channel; partial sums kept for display
  S.parts = l2.map((ker) => ker.map((slice, f) => conv2d(S.p1[f], S.Q, slice, { stride: 1, pad: 1, dil: 1 }).out));
  S.c2 = S.parts.map((ps) => {
    const acc = new Float32Array(S.Q * S.Q);
    ps.forEach((p) => { for (let q = 0; q < acc.length; q++) acc[q] += p[q]; });
    return acc;
  });
  S.a2 = S.c2.map(relu);
  S.p2 = S.a2.map((m) => pool2(m, S.Q, poolMode).out);
  S.Q2 = pool2(S.a2[0], S.Q, poolMode).Q;
  S.gap = S.p2.map((m) => m.reduce((a, b) => a + b, 0) / m.length);
  const NC = B.length;
  S.logits = Array.from({ length: NC }, (_, c) => S.gap.reduce((t, g, f) => t + g * W[f][c], B[c]));
  S.prob = softmax(S.logits);
  return S;
}

/**
 * Receptive-field propagation. probe = { stage: 'conv1'|'pool1'|'conv2'|'pool2', r, c, ch }.
 * Returns { stage: { rect:[r0,r1,c0,c1], chs: Set|null } } for every stage back to 'input'
 * (chs null = all channels; 'padIn' is the rect in padded-input coordinates, unclamped).
 */
export function footprint(probe, S) {
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const { N, stride, pad, dil } = S;
  const R = {};
  if (!probe) return R;
  R[probe.stage] = { rect: [probe.r, probe.r, probe.c, probe.c], chs: new Set([probe.ch]) };
  const order = ['pool2', 'conv2', 'pool1', 'conv1'];
  for (let i = order.indexOf(probe.stage); i < order.length; i++) {
    const s = order[i]; const r = R[s].rect;
    if (s === 'pool2') R.conv2 = { rect: [2 * r[0], Math.min(2 * r[1] + 1, S.Q - 1), 2 * r[2], Math.min(2 * r[3] + 1, S.Q - 1)], chs: R.pool2.chs };
    else if (s === 'conv2') R.pool1 = { rect: [clamp(r[0] - 1, 0, S.Q - 1), clamp(r[1] + 1, 0, S.Q - 1), clamp(r[2] - 1, 0, S.Q - 1), clamp(r[3] + 1, 0, S.Q - 1)], chs: null };
    else if (s === 'pool1') R.conv1 = { rect: [2 * r[0], Math.min(2 * r[1] + 1, S.O - 1), 2 * r[2], Math.min(2 * r[3] + 1, S.O - 1)], chs: R.pool1.chs };
    else if (s === 'conv1') {
      const e = dil * (K - 1);
      const pr = [r[0] * stride, r[1] * stride + e, r[2] * stride, r[3] * stride + e];
      R.padIn = { rect: pr };
      R.input = { rect: pr.map((v) => clamp(v - pad, 0, N - 1)), chs: null };
    }
  }
  return R;
}

// ---- shared presets ----
export const kernelPresets = {
  'Vertical edge (Sobel x)': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  'Horizontal edge (Sobel y)': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  'Laplacian': [0, 1, 0, 1, -4, 1, 0, 1, 0],
  'Diagonal': [2, 0, -2, 0, 0, 0, -2, 0, 2],
  'Blur (box)': Array(9).fill(1 / 9),
  'Sharpen': [0, -1, 0, -1, 5, -1, 0, -1, 0],
  'Identity': [0, 0, 0, 0, 1, 0, 0, 0, 0],
};
export const defaultBank = [
  kernelPresets['Vertical edge (Sobel x)'].slice(),
  kernelPresets['Horizontal edge (Sobel y)'].slice(),
  kernelPresets['Laplacian'].slice(),
  [1.5, 0, -1.5, 0, 0, 0, -1.5, 0, 1.5],
];

/** Input presets for an N×N grid; each returns a fresh Float32Array. */
export function inputPresets(N) {
  const mk = (fn) => () => {
    const a = new Float32Array(N * N);
    const set = (r, c, v) => { if (r >= 0 && c >= 0 && r < N && c < N) a[r * N + c] = v; };
    fn(set); return a;
  };
  return {
    'Clear': mk(() => {}),
    '7': mk((set) => { for (let c = 3; c < 13; c++) set(3, c, 1); for (let r = 4; r < 14; r++) { const c = 12 - Math.round((r - 4) * 0.7); set(r, c, 1); set(r, c - 1, 0.6); } }),
    '0': mk((set) => { for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const d = Math.hypot((r - 7.5) / 5, (c - 7.5) / 3.6); const v = 1 - Math.abs(d - 1) * 3; if (v > 0) set(r, c, Math.min(1, v)); } }),
    'Cross': mk((set) => { for (let r = 0; r < N; r++) { set(r, 7, 1); set(r, 8, 1); set(7, r, 1); set(8, r, 1); } }),
    'Diagonal': mk((set) => { for (let r = 0; r < N; r++) { set(r, r, 1); if (r < N - 1) set(r, r + 1, 0.5); } }),
    'Square': mk((set) => { for (let r = 4; r < 12; r++) for (let c = 4; c < 12; c++) set(r, c, 1); }),
    'Noise': mk((set) => { for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) set(r, c, Math.random()); }),
  };
}
