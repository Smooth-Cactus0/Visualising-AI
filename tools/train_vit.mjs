// tools/train_vit.mjs — train the toy ViT of core/vit.js on data/mnist16.bin; with --out, write the weights the
// ViT page loads. Prints held-out accuracy after every pass over the 3,000 training digits.
//   node tools/train_vit.mjs --opt sgd --lr 0.1 --passes 10
//   node tools/train_vit.mjs --opt sgd --lr 0.003 --passes 10 --out models/vit16.json
// Options: --opt sgd|adam, --lr, --passes, --batch (default 1), --seed (default 1), --out (a path from the repo root).
import { readFileSync, writeFileSync } from 'node:fs';
import { parseMnist16, image } from '../core/data.js';
import { shuffledOrder } from '../core/train.js';
import { init, paramCount, lossAndGrad, sgdStep, adamInit, adamStep, averageGrads, accuracy } from '../core/vit.js';

const argv = process.argv.slice(2), arg = (name, dflt) => { const i = argv.indexOf('--' + name); return i < 0 ? dflt : argv[i + 1]; };
const opt = arg('opt', 'sgd'), lr = +arg('lr', 0.003), passes = +arg('passes', 10);
const batch = +arg('batch', 1), seed = +arg('seed', 1), out = arg('out', null);
if (!['sgd', 'adam'].includes(opt)) throw new Error('--opt is sgd or adam');

const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
const every = (set) => Array.from({ length: set.n }, (_, i) => i);
const pct = (x) => (100 * x).toFixed(1) + '%';
let w = init(seed), st = adamInit(w);
console.log(`ViT, ${paramCount(w)} weights; ${opt}, lr ${lr}, batch ${batch}, seed ${seed}`);
console.log(`pass 0: held-out ${pct(accuracy(w, data.test, every(data.test)))}`);
const t0 = Date.now();
for (let pass = 1; pass <= passes; pass++) {
  const order = shuffledOrder(data.train.n, seed * 1000 + pass);
  let lossSum = 0;
  for (let b = 0; b < order.length; b += batch) {
    const gs = order.slice(b, b + batch).map((i) => { const r = lossAndGrad(w, image(data.train, i), data.train.labels[i]); lossSum += r.loss; return r.grads; });
    const g = gs.length === 1 ? gs[0] : averageGrads(gs);
    if (opt === 'adam') ({ w, state: st } = adamStep(w, g, st, lr)); else w = sgdStep(w, g, lr);
  }
  console.log(`pass ${pass}: mean loss ${(lossSum / order.length).toFixed(3)}, held-out ${pct(accuracy(w, data.test, every(data.test)))} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

if (out) {
  const weights = JSON.parse(JSON.stringify(w), (k, v) => (typeof v === 'number' ? +v.toPrecision(6) : v));
  const meta = {
    seed, optimiser: opt, lr, batch, passes, digits: passes * data.train.n,
    trainAcc: +accuracy(weights, data.train, every(data.train)).toFixed(4), testAcc: +accuracy(weights, data.test, every(data.test)).toFixed(4),
    testAccNoPos: +accuracy(weights, data.test, every(data.test), { position: false }).toFixed(4),
    made: `node tools/train_vit.mjs ${argv.join(' ')}`, date: new Date().toLocaleDateString('en-CA'),
  };
  writeFileSync(new URL('../' + out, import.meta.url), JSON.stringify({ version: 1, arch: 'vit16', meta, weights }) + '\n');
  console.log(`wrote ${out}: held-out ${pct(meta.testAcc)}, training ${pct(meta.trainAcc)} (rounded weights)`);
}
