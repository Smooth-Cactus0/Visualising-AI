// core/race.js — the CNN and the ViT trained on the same small set of digits, for the data-hunger comparison. The
// offline measurement (tools/vit_vs_cnn.mjs) and the live race on vit-train/ both step these racers, so a live run at a
// measured seed reproduces that dot of the chart exactly. Pure, no DOM.
import { image } from './data.js';
import { initWeights, lossAndGrad as cnnGrad, sgdStep as cnnStep, accuracy as cnnAccuracy, shuffledOrder } from './train.js';
import { init, lossAndGrad as vitGrad, sgdStep as vitStep, accuracy as vitAccuracy } from './vit.js';

/** The settings shared by the measurement and the live race: training-set sizes, steps per run, learning rates. */
export const RACE = { sizes: [100, 300, 1000, 3000], steps: 30000, lr: { cnn: 0.1, vit: 0.003 } };

/** The first n/10 digits of each class in the set, in file order: a balanced training set of n digits. */
export function balancedSubset(set, n) {
  const per = n / 10, seen = new Array(10).fill(0), out = [];
  for (let i = 0; i < set.n && out.length < n; i++) { const l = set.labels[i]; if (seen[l] < per) { seen[l]++; out.push(i); } }
  return out;
}

/** One model ('cnn' or 'vit') training with SGD on the digits idx of a set, in seeded shuffled passes. */
export function racer(kind, set, idx, seed) {
  const lr = RACE.lr[kind];
  let w = kind === 'cnn' ? initWeights({ seed }) : init(seed), steps = 0, pass = 0, order = [], pos = 0;
  return {
    kind, seed,
    get steps() { return steps; },
    get weights() { return w; },
    /** One training digit: forward, backward, update. */
    step() {
      if (pos >= order.length) { pass++; order = shuffledOrder(idx.length, seed * 1000 + pass).map((k) => idx[k]); pos = 0; }
      const i = order[pos++], img = image(set, i), y = set.labels[i];
      w = kind === 'cnn' ? cnnStep(w, cnnGrad(w, img, y).grads, lr) : vitStep(w, vitGrad(w, img, y).grads, lr);
      steps++;
    },
    accuracy(testSet, testIdx) { return kind === 'cnn' ? cnnAccuracy(w, testSet, testIdx) : vitAccuracy(w, testSet, testIdx); },
  };
}
