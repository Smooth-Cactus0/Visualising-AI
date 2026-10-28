// ui/vitline.js — how the ViT pages draw tensors. A view is a canvas showing all or part of one tensor; a line lays views
// out as columns, token i always on row i, with keys and values above the token rows and labels under them; paint draws
// a view from its tensor. Shared by vit/ and vit-train/; styles in ui/vitline.css.
import { T, NC } from '../core/vit.js';
import { drawGrid, cellAt, cssVar } from './grid.js';

export const el = (cls, text) => { const d = document.createElement('div'); d.className = cls; if (text != null) d.textContent = text; return d; };
const pct = (x) => (100 * x).toFixed(1) + '%';

/** A view factory. shapeOf(key) gives a tensor's [rows, cols]; onProbe({ t, r, c } | null) is called as the pointer
 *  moves over a view's cells, with the tensor cell under it. */
export function createViews({ shapeOf, onProbe }) {
  const views = []; // { cv, key, rows, cols, cell, map, yl, own, bars, blocks }
  const kOf = (v) => (typeof v.key === 'function' ? v.key() : v.key);
  /** A canvas for tensor `key` (or a function giving it) at `cell` px a cell. o.rows, o.cols and o.map(r, c) → [row, col]
   *  in the tensor show a part of it; o.yl() gives the yellow ({ win, cell }); o.own() replaces the hover's green;
   *  o.bars draws probabilities as bars on a canvas of o.px = [width, height]; o.blocks draws the patch grid. */
  function view(key, cell, o = {}) {
    const [R, C] = shapeOf(typeof key === 'function' ? key() : key);
    const v = { key, cell, rows: o.rows ?? R, cols: o.cols ?? C, map: o.map || ((r, c) => [r, c]), yl: o.yl, own: o.own, bars: o.bars, blocks: o.blocks };
    const cv = document.createElement('canvas'); v.cv = cv;
    cv.width = o.px ? o.px[0] : v.cols * cell; cv.height = o.px ? o.px[1] : v.rows * cell;
    cv.style.width = cv.width + 'px'; cv.style.height = cv.height + 'px';
    const probeAt = (ev) => { const p = v.bars ? barAt(v, ev) : cellAt(cv, ev); onProbe(p ? (([r, c]) => ({ t: kOf(v), r, c }))(v.map(p[0], p[1])) : null); };
    cv.addEventListener('pointermove', probeAt);
    cv.addEventListener('pointerdown', probeAt);
    cv.addEventListener('pointerleave', (ev) => { if (ev.pointerType !== 'touch') onProbe(null); });
    views.push(v); return cv;
  }
  return { views, view, kOf };
}

/**
 * Lay out columns on a grid. `rows` is one list of items, or several lists that share their columns (the training
 * page's forward, backward and update rows), with `labels` naming them in a sticky first column. An item is
 * { op, id }: an operator; { v: [elements], up, lb, w, grid, sec, id }: a column, `up` above the token rows (keys and
 * values), `w` tinted as learned, `grid` its number of sub-columns, `sec` the section a click on it scrolls to; or null:
 * an empty place. Shorter columns sit level with the middle of the token rows.
 */
export function line(box, rows, cell, { labels = null } = {}) {
  const bands = Array.isArray(rows[0]) ? rows : [rows], off = labels ? 1 : 0;
  box.className = 'line'; box.innerHTML = ''; box.style.setProperty('--mid', Math.max(0, T * cell / 2 - 9) + 'px');
  bands.forEach((items, b) => {
    const gr = (k) => String(3 * b + k);
    if (labels) { const l = el('rl', labels[b]); l.style.gridColumn = '1'; l.style.gridRow = gr(2); box.append(l); }
    items.forEach((it, x) => {
      if (!it) return;
      const col = String(x + 1 + off);
      if (it.op != null) { const o = el('op', it.op); o.style.gridColumn = col; o.style.gridRow = gr(2); if (it.id) o.dataset.id = it.id; box.append(o); return; }
      const parts = [];
      if (it.up) { const u = el('up'); u.append(...it.up); u.style.gridRow = gr(1); parts.push(u); }
      const a = el('at' + (it.w ? ' w' : '')); if (it.grid) a.style.gridTemplateColumns = `repeat(${it.grid}, auto)`; a.append(...it.v); a.style.gridRow = gr(2); if (it.id) a.dataset.id = it.id; parts.push(a);
      const l = el('lb', it.lb || ''); l.style.gridRow = gr(3); parts.push(l);
      for (const p of parts) { p.style.gridColumn = col; if (it.sec) p.dataset.sec = it.sec; box.append(p); }
      a.style.marginTop = Math.max(0, (T * cell - a.offsetHeight) / 2) + 'px';
    });
  });
}

/** Dashed residual arcs under a line's labels, from a tensor's column to the + that adds it back. */
export function arcs(box, pairs) {
  box.querySelector('svg.arcs')?.remove();
  const b = box.getBoundingClientRect(); if (!b.width) return;
  const ids = pairs.flat().map((id) => box.querySelector(`[data-id="${id}"]`));
  if (ids.some((e) => !e)) return;
  const labRow = +ids[0].style.gridRow + 1; // the labels of the arcs' own row of the line
  const labs = [...box.querySelectorAll('.lb')].filter((l) => +l.style.gridRow === labRow).map((l) => l.getBoundingClientRect().bottom - b.top);
  const y = Math.max(...labs) + 6; let d = '';
  for (const [from, to] of pairs) {
    const A = box.querySelector(`[data-id="${from}"]`).getBoundingClientRect(), B = box.querySelector(`[data-id="${to}"]`).getBoundingClientRect();
    const xa = A.left + A.width / 2 - b.left, xb = B.left + B.width / 2 - b.left;
    d += `M${xa} ${y - 5}V${y}H${xb}V${B.bottom - b.top + 2}`;
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.classList.add('arcs');
  svg.setAttribute('width', b.width); svg.setAttribute('height', y + 4);
  svg.innerHTML = `<path d="${d}" fill="none" stroke="var(--ink2)" stroke-width="1" stroke-dasharray="3 2"/>`;
  box.append(svg);
}

/** Probabilities as bars, labelled when there is room; glow outlines the lit classes. */
export function drawBars(v, vals, glow) {
  const { cv } = v, ctx = cv.getContext('2d'), h = cv.height / NC, wd = cv.width, top = vals.indexOf(Math.max(...vals));
  const lab = h >= 14, x0 = lab ? 16 : 0, x1 = lab ? wd - 46 : wd;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, wd, cv.height);
  ctx.font = `${Math.min(12, h * 0.6)}px ${getComputedStyle(document.body).fontFamily}`; ctx.textBaseline = 'middle';
  for (let c = 0; c < NC; c++) {
    const y = c * h;
    ctx.fillStyle = cssVar('--line2'); ctx.fillRect(x0, y + 1, x1 - x0, h - 2);
    ctx.fillStyle = cssVar('--accent'); ctx.globalAlpha = c === top ? 1 : 0.6; ctx.fillRect(x0, y + 1, (x1 - x0) * vals[c], h - 2); ctx.globalAlpha = 1;
    if (glow && glow[c] > 0) { ctx.strokeStyle = cssVar('--hl2'); ctx.lineWidth = 2; ctx.strokeRect(x0 + 1, y + 1, x1 - x0 - 2, h - 2); }
    if (lab) { ctx.fillStyle = cssVar('--ink2'); ctx.textAlign = 'left'; ctx.fillText(String(c), 2, y + h / 2); ctx.textAlign = 'right'; ctx.fillText(pct(vals[c]), wd - 2, y + h / 2); }
  }
}
export function barAt(v, ev) { const b = v.cv.getBoundingClientRect(), r = Math.floor((ev.clientY - b.top) / (b.height / NC)); return r >= 0 && r < NC ? [r, 0] : null; }

/** Draw view v from its tensor M (rows of numbers, C columns). G: a green strength per tensor cell, or null; scale: the
 *  colour scale; y: the yellow { win, cell }; o.gray draws grey values (the digit), o.faint dims a view not computed
 *  yet, o.outline frames it in green (what the last step computed). */
export function paint(v, M, C, G, scale, y = {}, o = {}) {
  const vals = new Float32Array(v.rows * v.cols), glow = G ? new Float32Array(v.rows * v.cols) : null;
  for (let r = 0; r < v.rows; r++) for (let c = 0; c < v.cols; c++) {
    const [mr, mc] = v.map(r, c); vals[r * v.cols + c] = M[mr][mc]; if (glow) glow[r * v.cols + c] = G[mr * C + mc];
  }
  if (v.bars) drawBars(v, vals, glow);
  else drawGrid(v.cv, vals, v.cols, v.rows, { mode: o.gray ? null : 'div', scale, glow, lines: v.cell >= 7, blocks: v.blocks, win: y.win, cell: y.cell, outline: o.outline });
  v.cv.style.opacity = o.faint ? '0.22' : '';
}
