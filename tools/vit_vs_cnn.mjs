// tools/vit_vs_cnn.mjs — measure the CNN and the ViT on small training sets, for the chart on vit-train/: held-out
// accuracy after RACE.steps SGD steps on balanced subsets of RACE.sizes digits, seeds 1 to 5, both models.
//   node tools/vit_vs_cnn.mjs            runs the 40 trainings in parallel and writes models/vit_vs_cnn.json
//   node tools/vit_vs_cnn.mjs --job cnn 100 1    one training; prints its held-out accuracy
import { readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { fileURLToPath } from 'node:url';
import { parseMnist16 } from '../core/data.js';
import { RACE, balancedSubset, racer } from '../core/race.js';

const SEEDS = [1, 2, 3, 4, 5];
const argv = process.argv.slice(2);

if (argv[0] === '--job') {
  const [kind, n, seed] = [argv[1], +argv[2], +argv[3]];
  const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
  const r = racer(kind, data.train, balancedSubset(data.train, n), seed);
  while (r.steps < RACE.steps) r.step();
  console.log(JSON.stringify({ kind, n, seed, acc: +r.accuracy(data.test, [...Array(data.test.n).keys()]).toFixed(4) }));
} else {
  const jobs = [];
  for (const kind of ['cnn', 'vit']) for (const n of RACE.sizes) for (const seed of SEEDS) jobs.push([kind, n, seed]);
  const me = fileURLToPath(import.meta.url), results = [], t0 = Date.now();
  const run = ([kind, n, seed]) => new Promise((done, fail) => {
    let out = '';
    const p = spawn(process.execPath, [me, '--job', kind, String(n), String(seed)]);
    p.stdout.on('data', (d) => { out += d; });
    p.on('close', (code) => (code === 0 ? done(JSON.parse(out)) : fail(new Error(`${kind} ${n} ${seed} exited ${code}`))));
  });
  const workers = Array.from({ length: Math.max(1, availableParallelism() - 1) }, async () => {
    while (jobs.length) { const r = await run(jobs.shift()); results.push(r); console.log(`${r.kind} ${r.n} digits, seed ${r.seed}: ${(100 * r.acc).toFixed(1)}%`); }
  });
  await Promise.all(workers);
  const table = (kind) => RACE.sizes.map((n) => SEEDS.map((seed) => results.find((r) => r.kind === kind && r.n === n && r.seed === seed).acc));
  const out = { version: 1, sizes: RACE.sizes, seeds: SEEDS, steps: RACE.steps, lr: RACE.lr, cnn: table('cnn'), vit: table('vit'),
    made: 'node tools/vit_vs_cnn.mjs', date: new Date().toLocaleDateString('en-CA') };
  writeFileSync(new URL('../models/vit_vs_cnn.json', import.meta.url), JSON.stringify(out) + '\n');
  console.log(`wrote models/vit_vs_cnn.json (${((Date.now() - t0) / 1000) | 0} s)`);
}
