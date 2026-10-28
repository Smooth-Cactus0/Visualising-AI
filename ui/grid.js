// ui/grid.js — canvas grid drawing and hit-testing shared by the 2D, training and ViT pages.
// Colours come from the page's CSS tokens (--line, --ink, --hl, --hl2) and follow its light/dark theme.
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/** Short number: '0', '1.5', '-0.33'. */
export const fmt = (v) => { if (Math.abs(v) < 1e-9) return '0'; const s = v.toFixed(2).replace(/\.?0+$/, ''); return s === '-0' ? '0' : s; }; // no "-0" for tiny negatives
export const isDark = () => document.documentElement.dataset.theme === 'dark' || (matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light');
export const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/** Diverging colour for a signed value: blue negative, orange positive, the page's zero tone in between. */
export function divColor(v, scale) {
  const t = clamp(v / scale, -1, 1);
  const neg = [42, 111, 214], pos = [229, 100, 30];
  const zero = isDark() ? [42, 47, 56] : [243, 244, 241];
  const c = t < 0 ? neg : pos, a = Math.abs(t);
  return `rgb(${zero.map((z, i) => Math.round(z + (c[i] - z) * a)).join(',')})`;
}
/** Grey for a 0–1 value (ink on the page background). */
export function grayColor(v) { const g = isDark() ? 40 + v * 200 : 245 - v * 225; return `rgb(${g | 0},${g | 0},${g | 0})`; }
export const maxAbs = (m) => Math.max(0.01, ...Array.from(m, Math.abs));

const layouts = new WeakMap(); // canvas -> {cell, ox, oy, w, h}

/** Draw a w×h grid of values on a canvas, with optional labels and highlights; remembers the layout for cellAt.
 *  Options used by the ViT page only: glow (a 0–1 green strength per cell), win ([r0, r1, c0, c1], a yellow
 *  rectangle), blocks (dashed lines every n cells) and lines: false (no inner grid lines). */
export function drawGrid(cv, data, w, h, o = {}) {
  const ctx = cv.getContext('2d'); const Wd = cv.width, Hd = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, Wd, Hd);
  const cell = Math.floor(Math.min(Wd / w, Hd / h)); const ox = Math.floor((Wd - cell * w) / 2), oy = Math.floor((Hd - cell * h) / 2);
  layouts.set(cv, { cell, ox, oy, w, h });
  const scale = o.scale || 1;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = data[r * w + c];
    ctx.fillStyle = o.mode === 'div' ? divColor(v, scale) : grayColor(v);
    ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
  }
  ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
  if (o.lines === false) ctx.strokeRect(ox + .5, oy + .5, w * cell - 1, h * cell - 1); // small cells: the outline only
  else {
    for (let r = 0; r <= h; r++) { ctx.beginPath(); ctx.moveTo(ox, oy + r * cell + .5); ctx.lineTo(ox + w * cell, oy + r * cell + .5); ctx.stroke(); }
    for (let c = 0; c <= w; c++) { ctx.beginPath(); ctx.moveTo(ox + c * cell + .5, oy); ctx.lineTo(ox + c * cell + .5, oy + h * cell); ctx.stroke(); }
  }
  if (o.blocks) { // dashed lines every `blocks` cells: the patch grid
    ctx.strokeStyle = cssVar('--accent'); ctx.setLineDash([3, 2]); ctx.beginPath();
    for (let k = o.blocks; k < w; k += o.blocks) { ctx.moveTo(ox + k * cell + .5, oy); ctx.lineTo(ox + k * cell + .5, oy + h * cell); }
    for (let k = o.blocks; k < h; k += o.blocks) { ctx.moveTo(ox, oy + k * cell + .5); ctx.lineTo(ox + w * cell, oy + k * cell + .5); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  if (o.padBorder) { const pad = o.pad || 0; ctx.strokeStyle = 'rgba(128,128,128,.9)'; ctx.setLineDash([3, 3]);
    ctx.strokeRect(ox + pad * cell + .5, oy + pad * cell + .5, (w - 2 * pad) * cell, (h - 2 * pad) * cell); ctx.setLineDash([]); }
  if (o.labels && cell >= 22) {
    ctx.font = `${Math.max(9, cell * 0.36)}px ${getComputedStyle(document.body).fontFamily}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const v = data[r * w + c]; const strong = o.mode === 'div' ? Math.abs(v / scale) > .55 : (isDark() ? v < .5 : v > .5);
      ctx.fillStyle = strong ? '#fff' : cssVar('--ink');
      if (o.mode !== 'div' && !strong && !isDark()) ctx.fillStyle = '#333';
      ctx.fillText(fmt(v), ox + c * cell + cell / 2, oy + r * cell + cell / 2);
    }
  }
  const hl = cssVar('--hl'), hl2 = cssVar('--hl2');
  if (o.glow) { // graded green: a 0–1 strength per cell, the full-strength cells outlined as one region
    const g = o.glow, full = (r, c) => r >= 0 && c >= 0 && r < h && c < w && g[r * w + c] >= 0.999;
    ctx.fillStyle = hl2;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const s = g[r * w + c]; if (s > 0) { ctx.globalAlpha = 0.12 + 0.38 * Math.min(1, s); ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell); }
    }
    ctx.globalAlpha = 1; ctx.strokeStyle = hl2; ctx.lineWidth = cell >= 8 ? 2 : 1.5; const e = ctx.lineWidth / 2; ctx.beginPath();
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) if (full(r, c)) {
      const x = ox + c * cell, y = oy + r * cell;
      if (!full(r - 1, c)) { ctx.moveTo(x, y + e); ctx.lineTo(x + cell, y + e); }
      if (!full(r + 1, c)) { ctx.moveTo(x, y + cell - e); ctx.lineTo(x + cell, y + cell - e); }
      if (!full(r, c - 1)) { ctx.moveTo(x + e, y); ctx.lineTo(x + e, y + cell); }
      if (!full(r, c + 1)) { ctx.moveTo(x + cell - e, y); ctx.lineTo(x + cell - e, y + cell); }
    }
    ctx.stroke();
  }
  if (o.taps) { ctx.strokeStyle = hl; ctx.lineWidth = 3; o.taps.forEach(([r, c]) => ctx.strokeRect(ox + c * cell + 1.5, oy + r * cell + 1.5, cell - 3, cell - 3)); }
  if (o.cell) { ctx.strokeStyle = hl; ctx.lineWidth = 3; ctx.strokeRect(ox + o.cell[1] * cell + 1.5, oy + o.cell[0] * cell + 1.5, cell - 3, cell - 3); }
  if (o.win) { const [r0, r1, c0, c1] = o.win; ctx.strokeStyle = hl; ctx.lineWidth = cell >= 8 ? 3 : 2; const lw = ctx.lineWidth / 2; // yellow window
    ctx.strokeRect(ox + c0 * cell + lw, oy + r0 * cell + lw, (c1 - c0 + 1) * cell - 2 * lw, (r1 - r0 + 1) * cell - 2 * lw); }
  if (o.rect) { const [r0, r1, c0, c1] = o.rect; ctx.strokeStyle = hl2; ctx.lineWidth = Math.max(2, Math.min(3, cell / 6));
    const lw = ctx.lineWidth / 2; ctx.strokeRect(ox + c0 * cell + lw, oy + r0 * cell + lw, (c1 - c0 + 1) * cell - 2 * lw, (r1 - r0 + 1) * cell - 2 * lw);
    ctx.fillStyle = hl2; ctx.globalAlpha = .18; ctx.fillRect(ox + c0 * cell, oy + r0 * cell, (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell); ctx.globalAlpha = 1; }
  if (o.outline) { cv.style.outline = `3px solid ${hl2}`; cv.style.outlineOffset = '2px'; } else cv.style.outline = '';
}

/** Grid cell under a pointer event, as [row, col], or null outside the grid. */
export function cellAt(cv, ev) {
  const L = layouts.get(cv); if (!L) return null; const rect = cv.getBoundingClientRect(); const sx = rect.width / cv.width;
  const c = Math.floor((ev.clientX - rect.left - L.ox * sx) / (L.cell * sx)), r = Math.floor((ev.clientY - rect.top - L.oy * sx) / (L.cell * sx));
  return (r >= 0 && c >= 0 && r < L.h && c < L.w) ? [r, c] : null;
}
