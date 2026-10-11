import { conv2d, outSize, pool2, forward, footprint, defaultBank, inputPresets } from './cnn.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

assert(outSize(16, { stride: 1, pad: 1, dil: 1 }) === 16, 'same-size conv');
assert(outSize(16, { stride: 2, pad: 0, dil: 2 }) === 6, 'stride/dilation size');

const id = [0, 0, 0, 0, 1, 0, 0, 0, 0];
const x = inputPresets(16)['7']();
const { out } = conv2d(x, 16, id);
assert(out.every((v, i) => Math.abs(v - x[i]) < 1e-6), 'identity kernel reproduces input');

const { out: p, Q } = pool2(new Float32Array([1, 2, 3, 4]), 2, 'max');
assert(Q === 1 && p[0] === 4, 'max pool');

const rnd = (n) => Array.from({ length: n }, () => Math.random() * 2 - 1);
const l2 = Array.from({ length: 8 }, () => Array.from({ length: 4 }, () => rnd(9)));
const W = Array.from({ length: 8 }, () => rnd(10)), B = rnd(10);
const S = forward({ input: x, N: 16, bank: defaultBank, l2, W, B });
assert(Math.abs(S.prob.reduce((a, b) => a + b) - 1) < 1e-9, 'softmax sums to 1');
assert(S.O === 16 && S.Q === 8 && S.Q2 === 4, 'shapes 16 → 8 → 4');

const R = footprint({ stage: 'pool2', r: 1, c: 1, ch: 0 }, S);
assert(R.input.rect.join() === '1,10,1,10', 'receptive field of pool2 (1,1): ' + R.input.rect);
assert(R.pool1.chs === null && R.conv2.chs.has(0), 'channel sets through conv2');

// the 2D page's options (ReLU off, average pooling) and its probe forms (rect + ch, rect + chs)
const { out: pa } = pool2(new Float32Array([1, 2, 3, 4]), 2, 'avg');
assert(pa[0] === 2.5, 'avg pool');
const S0 = forward({ input: x, N: 16, bank: defaultBank, l2, W, B, activation: 'none', poolMode: 'avg' });
assert(S0.a1.every((m, k) => m === S0.c1[k]) && S0.a2.every((m, k) => m === S0.c2[k]), "activation 'none' passes conv outputs through");
assert(S0.c1[0].some((v) => v < 0) && S.a1[0].every((v) => v >= 0), 'ReLU clips negatives by default');
const Rr = footprint({ stage: 'conv1', rect: [2, 2, 3, 3], ch: 1 }, S);
assert(Rr.input.rect.join() === '1,3,2,4' && Rr.conv1.chs.size === 1 && Rr.conv1.chs.has(1), 'rect + ch probe');
const Rg = footprint({ stage: 'pool2', rect: [0, S.Q2 - 1, 0, S.Q2 - 1], chs: null }, S);
assert(Rg.input.rect.join() === '0,15,0,15' && Rg.pool2.chs === null && Rg.conv2.chs === null, 'whole pool-2 block (head probe) sees the whole input');

console.log('core/cnn.js OK');
