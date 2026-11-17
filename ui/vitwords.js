// ui/vitwords.js — what the ViT explorer pages say and load: the readout for a probed value, the sentences about where
// the weights came from and what their accuracy was measured on, and loading the digits and the weights (a hand-off from
// vit-train/, the shipped file, or random with a banner). Shared by vit/ and vit-3d/.
import { DH, tensor, isWeights } from '../core/vit.js';
import { parseMnist16 } from '../core/data.js';
import { fmt } from './grid.js';
import { loadTrainedViT, showTrainedViTBanner } from './trained.js';

export const pct = (x) => (100 * x).toFixed(1) + '%';
export const NAME = { xp: 'patch token', z0: 'z₀', h1: 'LayerNorm 1 output', q: 'query', k: 'key', v: 'value', o: 'heads joined', z1: 'z₁', h2: 'LayerNorm 2 output', hid: 'MLP hidden', z2: 'z₂',
  Wp: 'patch kernels', bp: 'patch bias', pos: 'position table', ln1: 'LayerNorm 1 (row 0 scale, row 1 shift)', ln2: 'LayerNorm 2 (row 0 scale, row 1 shift)', lnf: 'final LayerNorm (row 0 scale, row 1 shift)',
  Wq: 'Wq', bq: 'bq', Wk: 'Wk', bk: 'bk', Wv: 'Wv', bv: 'bv', Wo: 'Wo', bo: 'bo', W1: 'W₁', b1: 'b₁', W2: 'W₂', b2: 'b₂', Wh: 'W head', bh: 'head bias' };
const meanAtt = (S) => S.att[0].map((r, a) => r.map((x, b) => (x + S.att[1][a][b]) / 2));

/** The readout for a probe { t, r, c } of the forward pass S under the weights w. */
export function describe(p, S, w) {
  if (!p) return 'Hover any cell: green shows what it is computed from. After attention, its strength is the attention weight.';
  const val = tensor(p.t, S, w)[p.r][p.c], at = `<b>${NAME[p.t] || p.t}</b>`;
  if (p.t === 'digit') { const s = (p.r >> 2) * 4 + (p.c >> 2); return `<b>pixel (${p.r}, ${p.c})</b> = ${fmt(val)}, in patch ${s}, which becomes token ${s}.`; }
  if (p.t === 'att0' || p.t === 'att1') {
    const h = +p.t[3];
    return `<b>head ${h + 1}</b>: token ${p.r} gives token ${p.c} the weight <b>${fmt(val)}</b> (score ${fmt(S.scores[h][p.r][p.c])}); each row sums to 1. Green: query ${p.r} and key ${p.c}, and every other key faintly, since the softmax divides by all of them.`;
  }
  if (['xp', 'z0', 'h1', 'q', 'k', 'v'].includes(p.t)) return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. Before attention a token has seen only its own patch.`;
  if (['o', 'z1', 'h2', 'hid', 'z2'].includes(p.t)) {
    const one = p.t === 'o', A = one ? S.att[Math.floor(p.c / DH)] : meanAtt(S);
    const top = A[p.r].map((a, j) => [a, j]).filter(([, j]) => j !== p.r).sort((x, y) => y[0] - x[0]).slice(0, 3).map(([a, j]) => `token ${j} (${fmt(a)})`).join(', ');
    return `${at}, token ${p.r}, column ${p.c} = ${fmt(val)}. After attention: its own patch full green, every other patch by the attention weight it got${one ? ` in head ${Math.floor(p.c / DH) + 1}` : ', both heads averaged'}, scaled to the largest (attention, not importance). Most attended: ${top}.`;
  }
  if (p.t === 'avg') return `<b>average, column ${p.c}</b> = ${fmt(val)}: column ${p.c} of z₂ averaged over the 16 tokens, as the CNN averages each map.`;
  if (p.t === 'avgNorm') return `<b>normalised average, column ${p.c}</b> = ${fmt(val)}.`;
  if (p.t === 'prob') return `<b>class ${p.c}</b>: ${pct(val)}. It reads only the 8 normalised averages, through column ${p.c} of W head and its bias.`;
  return `${at} [${p.r}, ${p.c}] = ${fmt(val)}, learned. Green: what it reads and what it makes.`;
}

/** The page's two statements about the weights: `src` (HTML, beside the prediction) and `note` (a sentence). The
 *  accuracy was measured with position on and the patches in reading order; they say so when the page is set otherwise. */
export function weightWords({ S, trained, meta, position, order, data }) {
  const top = S.prob.indexOf(Math.max(...S.prob));
  const opt = meta && (meta.optimiser === 'adam' ? 'Adam' : 'SGD');
  const noPos = meta && Number.isFinite(meta.testAccNoPos) ? pct(meta.testAccNoPos) : null;
  let short = `trained weights: ${trained ? pct(meta.testAcc) : ''} right on held-out digits`, full = '';
  if (trained && !position) { short = `trained weights, trained with position${noPos ? `: ${noPos} right on held-out digits without it` : ''}`; full = ` The weights were trained with position, so with position off${noPos ? ` they get only ${noPos} right` : ' that figure does not apply'}.`; }
  if (trained && order) { short += ' (accuracy measured on patches in reading order, not shuffled)'; full += ' The accuracy was measured on patches in reading order, not on shuffled ones.'; }
  const n = (set) => set.n.toLocaleString('en-NZ');
  return {
    src: `prediction <b>${top}</b> (${pct(S.prob[top])}), ` + (trained ? short : 'random weights: arbitrary'),
    note: trained
      ? `These weights were trained ${meta.source === 'vit-train' ? `on the ViT training page (${meta.digits.toLocaleString('en-NZ')} digits seen, ${opt}, learning rate ${meta.lr})` : `offline by tools/train_vit.mjs on the ${n(data.train)} training digits (${meta.passes} passes, ${opt})`}; with position on and the patches in reading order they get ${pct(meta.testAcc)} of the ${n(data.test)} held-out digits right; the CNN pages' toy, trained on the same 3,000 digits for 30,000 steps, reaches 71 to 85% depending on the seed (5 runs, shown on the ViT training page).${full} On real photographs a ViT needs far more training data than a CNN to catch up: it has to learn from examples the locality a convolution has built in.`
      : 'These are random starting weights, before any training. Attention is learned: before training its maps are noise and the prediction is arbitrary.',
  };
}

function banner(anchor, text) { const d = document.createElement('div'); d.className = 'banner'; d.textContent = text; anchor.after(d); }
/**
 * Load the digits and the weights, with a banner after `anchor` when something is missing or the weights are a hand-off.
 * The page passes both URLs, so its checks can point them elsewhere. Returns null when the digits cannot be loaded,
 * else { data, byClass (held-out digit indices per class), wTrained (null when none), meta }.
 */
export async function loadDigitsAndWeights(anchor, digitsUrl, weightsUrl) {
  const [bin, js] = await Promise.all([
    fetch(digitsUrl).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null),
    fetch(weightsUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  if (!bin) { banner(anchor, 'Could not load the digits (data/mnist16.bin); this page needs to be served over http.'); return null; }
  const data = parseMnist16(bin);
  const byClass = Array.from({ length: 10 }, (_, c) => [...Array(data.test.n).keys()].filter((i) => data.test.labels[i] === c));
  const stored = loadTrainedViT(isWeights);
  let wTrained = null, meta = null;
  if (stored) { wTrained = stored.weights; meta = stored.meta; showTrainedViTBanner(anchor, stored); }
  else if (js && js.version === 1 && isWeights(js.weights)) { wTrained = js.weights; meta = js.meta; }
  else banner(anchor, 'Could not load the trained weights (models/vit16.json), so this page shows random starting weights.');
  return { data, byClass, wTrained, meta };
}
