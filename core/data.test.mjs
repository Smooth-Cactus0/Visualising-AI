import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

const d = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
assert(d.N === 16 && d.train.n === 3000 && d.test.n === 1000, 'counts');
assert(d.train.labels.length === 3000 && d.train.pixels.length === 3000 * 256, 'train views');
assert(d.test.labels.length === 1000 && d.test.pixels.length === 1000 * 256, 'test views');
for (const set of [d.train, d.test]) {
  const per = new Array(10).fill(0); for (const l of set.labels) { assert(l >= 0 && l <= 9, 'label range'); per[l]++; }
  assert(per.every((v) => v === set.n / 10), 'balanced classes: ' + per);
}
const x = image(d.train, 0);
assert(x instanceof Float32Array && x.length === 256, 'image shape');
assert(x.every((v) => v >= 0 && v <= 1) && Math.max(...x) > 0.9 && x.filter((v) => v === 0).length > 100, 'image scale');
assert(image(d.train, 0) !== x, 'fresh array per call');
let threw = false; try { parseMnist16(new Uint8Array(32)); } catch { threw = true; } assert(threw, 'rejects a non-MN16 buffer');
console.log('core/data.js OK');
