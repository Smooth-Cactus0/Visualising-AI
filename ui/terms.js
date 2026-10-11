// ui/terms.js — dotted inline definitions (tooltip on hover, focus or tap) and the Terms section.
// Shared by every page. A page links ../ui/terms.css, has <div id="tip" role="tooltip"></div> and an empty
// .gloss element, and calls installTerms once its static text is in the DOM. Dynamic text (readouts, hints
// rewritten on every step) is left unlinked on purpose.

export const TERMS = [
  ['kernel', /\bkernel\b/i, 'Kernel (filter)', 'A small grid of weights, 3×3 here, that is slid over the input. Its values are learned parameters: training changes them. Each kernel produces one feature map.'],
  ['weights', /\bweights\b/i, 'Weights / parameters', 'The numbers a network learns: every kernel value and every FC entry. Nothing else on this page is learned — ReLU, pooling, global averaging and softmax are fixed arithmetic.'],
  ['feature map', /\bfeature map\b/i, 'Feature map', 'The output grid of one kernel: at each position, how strongly the kernel\'s pattern matches the input there. Positive = looks like the pattern, negative = looks like its opposite.'],
  ['channel', /\bchannels?\b/i, 'Channel', 'One of the stacked maps at a given stage. An RGB image has 3 channels; after a layer with 8 kernels there are 8. The "C" in H×W×C.'],
  ['stride', /\bstride\b/i, 'Stride', 'How many pixels the window moves between neighbouring outputs. Stride 2 halves the output size and makes each output cell cover twice the input width.'],
  ['padding', /\bpadding\b/i, 'Padding', 'A ring of zeros added around the input so the window can be centred on edge pixels. Padding 1 with a 3×3 kernel keeps the output the same size as the input.'],
  ['dilation', /\bdilation\b/i, 'Dilation', 'Spacing between the kernel taps. Dilation 2 reads every second pixel, so a 3×3 kernel spans a 5×5 patch with the same 9 weights.'],
  ['relu', /\bReLU\b/, 'ReLU', 'Rectified linear unit: max(0, x). Keeps positive responses, replaces negative ones with zero. This is the nonlinearity that stops stacked layers from collapsing into one.'],
  ['pool', /\b(max-?pool(ing)?|pooling)\b/i, 'Pooling', 'Shrinks a map by summarising each 2×2 block with one number (its max or mean). Halves height and width; no learned parameters. Keeps the strongest response, drops exactly where it was.'],
  ['gap', /\bglobal average(?: pooling)?\b/i, 'Global average pooling', 'Averages each channel over all positions, turning an H×W×C block into a vector of C numbers: "how much of each pattern, anywhere". Removes spatial layout entirely.'],
  ['fc', /\b(fully connected(?: \(FC\))?(?: layer)?|FC)\b/, 'Fully connected (FC) layer', 'Also called dense or linear. Every input value connects to every output value through its own weight: 8 inputs × 10 outputs = 80 weights plus 10 biases. Output = input · W + b. The weights are learned parameters, like kernels; on this page they are random.'],
  ['bias', /\bbias(es)?\b/i, 'Bias', 'A constant added to each output of a layer: one per output, learned. It shifts a score up or down regardless of the input.'],
  ['score', /\bclass scores?\b/i, 'Score (logit)', 'The raw output of the FC layer for one class, before softmax. Can be any real number; higher means the network favours that class.'],
  ['softmax', /\bsoftmax\b/i, 'Softmax', 'Turns scores into probabilities: p_k = exp(s_k) / Σ exp(s_j). All outputs are positive and sum to 1; the largest score gets the largest share, and gaps between scores are exaggerated.'],
  ['rf', /\breceptive field\b/i, 'Receptive field', 'The patch of the input that can influence one value in a later map. Grows with every conv and every pool; the green boxes trace it.'],
  ['training', /\btraining\b/i, 'Training', 'Showing the network many labelled images and nudging every weight so the correct class score rises. This page skips it; the weights are random or hand-picked, but the forward pass is identical.'],
];
const termById = Object.fromEntries(TERMS.map((t) => [t[0], t]));

/** Wrap the first mention of each term inside every element matching `selectors` in a <dfn>.
 *  Terms are taken in order of appearance in the text, so a term is never skipped because another one
 *  listed earlier in TERMS happens to occur later in the same paragraph. */
export function linkTerms(selectors) {
  document.querySelectorAll(selectors).forEach((el) => {
    const seen = new Set();
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.parentNode.closest('dfn,.n')) return;
      let text = node.nodeValue;
      for (;;) {
        let best = null;
        for (const [id, re] of TERMS) { if (seen.has(id)) continue; const m = re.exec(text); if (m && (!best || m.index < best.m.index)) best = { id, m }; }
        if (!best) break;
        const { id, m } = best; seen.add(id);
        const before = document.createTextNode(text.slice(0, m.index));
        const d = document.createElement('dfn'); d.dataset.t = id; d.tabIndex = 0; d.textContent = m[0];
        const after = document.createTextNode(text.slice(m.index + m[0].length));
        node.parentNode.insertBefore(before, node); node.parentNode.insertBefore(d, node); node.parentNode.replaceChild(after, node);
        node = after; text = after.nodeValue;
      }
    });
  });
}

/** Show a <dfn>'s definition in `tip` on hover and keyboard focus; a click or tap toggles it. */
export function bindTooltip(tip) {
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const show = (d) => {
    const t = termById[d.dataset.t]; tip.innerHTML = `<b>${t[2]}</b>${t[3]}`; tip.style.display = 'block';
    const r = d.getBoundingClientRect(); const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = clamp(r.left, 8, innerWidth - w - 8), y = r.bottom + 8; if (y + h > innerHeight - 8) y = r.top - h - 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };
  const dfnOf = (e) => e.target.closest && e.target.closest('dfn');
  document.addEventListener('pointerover', (e) => { const d = dfnOf(e); if (d) show(d); });
  document.addEventListener('pointerout', (e) => { if (dfnOf(e)) hide(); });
  document.addEventListener('focusin', (e) => { const d = dfnOf(e); if (d) show(d); });
  document.addEventListener('focusout', (e) => { if (dfnOf(e)) hide(); });
  document.addEventListener('click', (e) => {
    const d = dfnOf(e);
    if (d) { if (tip.style.display === 'block' && tip.dataset.for === d.dataset.t) { hide(); tip.dataset.for = ''; } else { show(d); tip.dataset.for = d.dataset.t; } }
    else { hide(); tip.dataset.for = ''; }
  });
}

/** Fill the Terms section with every definition. */
export function renderGlossary(el) {
  el.innerHTML = TERMS.map((t) => `<div><b>${t[2]}</b> — ${t[3]}</div>`).join('');
}

/** Link terms in the elements matching `selectors`, bind the tooltip element, fill the glossary element. */
export function installTerms({ selectors, tip, gloss }) {
  linkTerms(selectors);
  if (tip) bindTooltip(tip);
  if (gloss) renderGlossary(gloss);
}
