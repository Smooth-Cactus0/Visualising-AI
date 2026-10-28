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
