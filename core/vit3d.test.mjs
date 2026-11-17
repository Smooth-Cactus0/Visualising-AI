import { readFileSync } from 'node:fs';
import { parseMnist16, image } from './data.js';
import { SHAPES, T, init, forward, tensor, vitTrace } from './vit.js';
import { VIT_BLOCKS, blockOf, cubeToCell, cellToCube, sliceValues, traceBoxes, attentionFan } from './vit3d.js';
const assert = (c, m) => { if (!c) throw new Error(m); };

// every drawn tensor has a block, with as many cubes as the tensor has cells
const drawn = new Set(VIT_BLOCKS.map((b) => b.key));
for (const key of Object.keys(SHAPES)) {
  const bk = key === 'att0' || key === 'att1' ? 'att' : key;
  assert(drawn.has(bk), `${key} has a block`);
}
for (const b of VIT_BLOCKS) {
  const cells = b.key === 'att' ? 2 * T * T : SHAPES[b.key][0] * SHAPES[b.key][1];
  assert(b.rows * b.cols * b.chans === cells, `${b.key}: ${b.rows}×${b.cols}×${b.chans} cubes for ${cells} cells`);
}

// cube → cell → cube round-trips for every cube of every block, and every cell is drawn exactly once
let n = 0; const seen = new Set();
for (const b of VIT_BLOCKS) for (let ch = 0; ch < b.chans; ch++) for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
  const p = cubeToCell(b.key, r, c, ch), q = cellToCube(p.t, p.r, p.c);
  assert(q.key === b.key && q.r === r && q.c === c && q.ch === ch, `${b.key} (${r}, ${c}, ${ch}) → ${JSON.stringify(p)} → ${JSON.stringify(q)}`);
  const id = `${p.t}:${p.r}:${p.c}`; assert(!seen.has(id), `${id} drawn twice`); seen.add(id); n++;
}
console.log(`vit3d: ${VIT_BLOCKS.length} blocks, ${n} cubes, every cube ↔ cell round-trips`);

// the layout: token t at its patch's place; attention weight (i, j) of head h in token i's group at token j's place
assert(JSON.stringify(cellToCube('z0', 6, 3)) === JSON.stringify({ key: 'z0', r: 1, c: 2, ch: 3 }), 'token 6 sits at row 1, column 2');
assert(JSON.stringify(cellToCube('att1', 6, 9)) === JSON.stringify({ key: 'att', r: 6, c: 9, ch: 1 }), 'att1 (6, 9): group of token 6 (rows 4–7, cols 8–11), place of token 9');
assert(JSON.stringify(cellToCube('att0', 5, 0)) === JSON.stringify({ key: 'att', r: 4, c: 4, ch: 0 }), 'att0 (5, 0)');
assert(JSON.stringify(cellToCube('Wp', 3, 6)) === JSON.stringify({ key: 'Wp', r: 1, c: 2, ch: 3 }), 'kernel 3, pixel 6');

// sliceValues reads the tensor through the mapping
const data = parseMnist16(readFileSync(new URL('../data/mnist16.bin', import.meta.url)));
const w = init(3), S = forward(w, image(data.test, 0)), value = (t, r, c) => tensor(t, S, w)[r][c];
const z = sliceValues('z0', 5, value);
assert(z.length === 16 && z[1 * 4 + 2] === Math.fround(S.z0[6][5]), 'z0 slice 5 at (1, 2) is token 6, number 5');
const a = sliceValues('att', 1, value);
assert(a[6 * 16 + 9] === Math.fround(S.att[1][6][9]), 'attention slice 1 at (6, 9) is head 2, weight (6, 9)');

// traceBoxes: a token before attention is one box through its depth; a lit matrix column is one box
const tq = vitTrace(S, { t: 'q', r: 5, c: 2 }), bq = traceBoxes(tq);
const box = (key) => bq.filter((x) => x.key === key);
assert(box('xp').length === 1 && JSON.stringify(box('xp')[0]) === JSON.stringify({ key: 'xp', r0: 1, r1: 1, c0: 1, c1: 1, ch0: 0, ch1: 7, s: 1 }), 'token 5 of xp: one box through its 8 slices');
assert(box('digit').length === 1 && box('digit')[0].r0 === 4 && box('digit')[0].r1 === 7 && box('digit')[0].c0 === 4 && box('digit')[0].c1 === 7, 'patch 5: one 4×4 box on the digit');
assert(box('Wq').length === 1 && box('Wq')[0].c0 === 2 && box('Wq')[0].c1 === 2 && box('Wq')[0].r0 === 0 && box('Wq')[0].r1 === 7, 'column 2 of Wq: one box');
// after attention: graded boxes, one per token and block, with the own token at 1
const tz = vitTrace(S, { t: 'z1', r: 6, c: 0 }), bz = traceBoxes(tz);
const z0boxes = bz.filter((x) => x.key === 'z0');
assert(z0boxes.length >= 2 && z0boxes.every((x) => x.ch0 === 0 && x.ch1 === 7), 'z0 after attention: whole-depth boxes');
assert(z0boxes.some((x) => x.r0 === 1 && x.c0 === 2 && x.s === 1), 'token 6 at full strength');
// covering: every lit cell lies in exactly one box, at its strength
for (const [t, arr] of Object.entries(tz)) {
  const C = SHAPES[t][1];
  arr.forEach((s, q) => {
    const cube = cellToCube(t, Math.floor(q / C), q % C);
    const hits = bz.filter((x) => x.key === cube.key && cube.r >= x.r0 && cube.r <= x.r1 && cube.c >= x.c0 && cube.c <= x.c1 && cube.ch >= x.ch0 && cube.ch <= x.ch1);
    if (s > 0) assert(hits.length === 1 && Math.abs(hits[0].s - s) < 1e-6, `${t}[${q}] covered once at ${s}`);
    else assert(hits.length === 0 || cube.key === 'att', `${t}[${q}] unlit and uncovered`); // att0 and att1 share a block, so each other's cells may be lit
  });
}
console.log(`traceBoxes: before attention ${bq.length} boxes, after attention ${bz.length}; every lit cell covered once`);

// the fan: after attention, links from token 6's group to the tokens it read, graded; nothing before attention
const fan = attentionFan(tz, { t: 'z1', r: 6, c: 0 });
assert(fan.length > 0 && fan.every((f) => f.i === 6 && f.j !== 6 && f.s >= 0.05 && f.s <= 0.9 + 1e-6), 'fan of token 6');
assert(Math.max(...fan.map((f) => f.s)) > 0.9 - 1e-6 && new Set(fan.map((f) => f.h)).size === 2, 'the most attended at 0.9, both heads');
const fanO = attentionFan(vitTrace(S, { t: 'o', r: 6, c: 5 }), { t: 'o', r: 6, c: 5 });
assert(fanO.length > 0 && fanO.every((f) => f.h === 1), "a head's own output fans from that head's slice");
assert(attentionFan(tq, { t: 'q', r: 5, c: 2 }).length === 0, 'no fan before attention');
console.log(`attentionFan: ${fan.length} links for z₁ token 6 (both heads), ${fanO.length} for head 2's output`);
console.log('core/vit3d.test.mjs: all checks passed');
