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
const S = forward({
  input: x, N: 16, bank: defaultBank,
  l2: Array.from({ length: 8 }, () => Array.from({ length: 4 }, () => rnd(9))),
  W: Array.from({ length: 8 }, () => rnd(10)), B: rnd(10),
});
assert(Math.abs(S.prob.reduce((a, b) => a + b) - 1) < 1e-9, 'softmax sums to 1');
assert(S.O === 16 && S.Q === 8 && S.Q2 === 4, 'shapes 16 → 8 → 4');

const R = footprint({ stage: 'pool2', r: 1, c: 1, ch: 0 }, S);
assert(R.input.rect.join() === '1,10,1,10', 'receptive field of pool2 (1,1): ' + R.input.rect);
assert(R.pool1.chs === null && R.conv2.chs.has(0), 'channel sets through conv2');

console.log('core/cnn.js OK');
