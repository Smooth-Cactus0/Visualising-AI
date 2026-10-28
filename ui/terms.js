// ui/terms.js — dotted inline definitions (tooltip on hover, focus or tap) and the Terms section.
// Shared by every page, each with its own list of terms (TERMS for the CNN pages, VIT_TERMS for the ViT page). A page links ../ui/terms.css, has <div id="tip" role="tooltip"></div> and an empty
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
  ['training', /\btraining\b/i, 'Training', 'Showing the network many labelled images and nudging every weight so the correct class score rises. The training page does it live; the explorer pages skip it, so their weights are random or hand-picked unless trained ones are loaded, but the forward pass is identical.'],
  ['loss', /\b(cross-entropy|loss)\b/i, 'Loss (cross-entropy)', 'One number for how wrong a prediction was: minus the log of the probability given to the true class. Zero when the network is certain and right, large when it is confident and wrong. Training pushes it down.'],
  ['gradient', /\bgradients?\b/i, 'Gradient', 'For each weight, how much the loss would change per unit change of that weight, sign included: the direction that makes things worse. Written ∂L/∂w. Training moves every weight a little the other way.'],
  ['backprop', /\bbackpropagation\b/i, 'Backpropagation', 'The bookkeeping that gives every weight its gradient. The error at the output is passed back through the layers one at a time; each layer turns the gradient at its output into the gradient at its input and at its own weights (the chain rule).'],
  ['lr', /\blearning rate\b/i, 'Learning rate', 'The step size: each weight moves by learning rate × its gradient. Too small and training crawls; too large and it overshoots, so the loss jumps around instead of falling.'],
  ['update', /\b(update step|SGD)\b/, 'Update (SGD step)', 'w ← w − learning rate × ∂L/∂w for every weight at once. Stochastic gradient descent: stochastic because each step uses one digit, or a small batch, rather than the whole dataset.'],
  ['batch', /\bbatch\b/i, 'Batch', 'How many digits are seen before one update. Their gradients are averaged, so a bigger batch gives steadier but fewer steps.'],
  ['pass', /\bpass(es)?\b/i, 'Pass (epoch)', 'One trip through the whole training set, in a shuffled order. Real training runs many.'],
  ['heldout', /\bheld-out\b/i, 'Held-out test set', 'Digits the network never trains on, kept only to measure accuracy. Accuracy on the training digits can be flattered by memorising them; held-out accuracy cannot.'],
  ['init', /\binitiali[sz]ation\b/i, 'Initialisation', 'The random starting values of the weights. Small random numbers, not zeros: identical weights would get identical gradients and never become different detectors.'],
];

/** The ViT page's terms. Longer phrases come before the words inside them ("patch embedding" before "patch"), so that
 *  at the same position the longer one is linked. */
export const VIT_TERMS = [
  ['patchemb', /\bpatch embedding\b/i, 'Patch embedding', 'The first layer of a ViT: each 4×4 patch of pixels becomes 8 numbers through 8 learned kernels. It is a convolution whose stride equals its size, so the windows never overlap.'],
  ['patch', /\bpatch(es)?\b/i, 'Patch', 'A square tile of the image, 4×4 pixels here. The 16×16 digit is cut into 16 of them, numbered in reading order.'],
  ['position', /\bposition (embedding|table)s?\b/i, 'Position embedding', 'A learned row of 8 numbers per patch slot, added to the token in that slot. Attention by itself treats the tokens as a set; this table is the only way the model can know where a patch sat.'],
  ['token', /\btokens?\b/i, 'Token', 'One vector per patch, 8 numbers here, carried through the network as one row. Every step except attention works on each token separately.'],
  ['head', /\b(attention )?heads?\b/i, 'Attention head', 'One complete attention pattern with its own queries, keys and values. This model has two, each 4 numbers wide; their outputs are joined back into 8 numbers and mixed by Wo.'],
  ['attention', /\b(self-)?attention\b/i, 'Attention', 'Each token builds its output as a weighted average of every token\'s value. The weights come from how well its query matches each key, through a softmax, so they are positive and sum to 1. They are computed from the image: where a token looks is not fixed, unlike a kernel\'s window.'],
  ['qkv', /\bquer(y|ies)\b/i, 'Query, key, value', 'Three learned projections of each token. The query is what the token looks for, the key what it offers to others, the value what it hands over. Score(i, j) = query i · key j / √4.'],
  ['layernorm', /\bLayerNorm\b/, 'LayerNorm', 'Rescales each token on its own to mean 0 and variance 1 across its 8 numbers, then applies a learned scale and shift per column. It keeps the numbers in range from step to step.'],
  ['residual', /\bresidual( connection)?s?\b/i, 'Residual connection', 'A block\'s input is added back to its output: z₁ = z₀ + attention, z₂ = z₁ + MLP. The block only has to learn a correction, and each token keeps its own information, which is why its own patch stays full green.'],
  ['mlp', /\bMLP\b/, 'MLP', 'Multilayer perceptron: two fully connected layers, 8 → 32 → 8, with GELU between them, applied to each token separately. In CNN terms, two 1×1 convolutions.'],
  ['gelu', /\bGELU\b/, 'GELU', 'Gaussian error linear unit, a smooth cousin of ReLU: close to 0 for large negative inputs, close to x for large positive ones, curved in between. Computed here in its tanh form.'],
  ['avg', /\baverage of the tokens\b/i, 'Average of the tokens', 'The 16 tokens averaged into one vector of 8 numbers, like the CNN\'s global average pooling, then turned into 10 class scores. The original ViT reads an extra class token instead; with a single block, only that token\'s row would reach the answer.'],
  ['kernel', /\bkernels?\b/i, 'Kernel (filter)', 'A small grid of learned weights, 4×4 here. In the patch embedding each of the 8 kernels reads one patch at a time and gives one of the token\'s 8 numbers.'],
  ['stride', /\bstride\b/i, 'Stride', 'How far the window jumps between outputs. With stride 4 and a 4×4 kernel the windows tile the image exactly: that is what cutting it into patches means.'],
  ['softmax', /\bsoftmax\b/i, 'Softmax', 'Turns scores into weights that are positive and sum to 1: exp(s) / Σ exp. Used twice here: across each row of an attention map, and over the 10 classes.'],
  ['rf', /\breceptive field\b/i, 'Receptive field', 'The part of the input a value can depend on. Before attention a token\'s receptive field is its own patch; after one attention step it is the whole image, which the green shades by attention weight.'],
  ['weights', /\bweights\b/i, 'Weights / parameters', 'The numbers a network learns: the patch kernels, the position table, every matrix and bias, the LayerNorm scales and shifts. Tinted wherever they sit. Softmax, GELU, the average and the residual additions are fixed arithmetic.'],
  ['heldout', /\bheld-out\b/i, 'Held-out test set', 'Digits the network never trained on, kept only to measure accuracy.'],
];

/** Wrap the first mention of each term inside every element matching `selectors` in a <dfn>.
 *  Terms are taken in order of appearance in the text, so a term is never skipped because another one
 *  listed earlier in TERMS happens to occur later in the same paragraph. */
export function linkTerms(selectors, terms = TERMS) {
  document.querySelectorAll(selectors).forEach((el) => {
    const seen = new Set();
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.parentNode.closest('dfn,.n')) return;
      let text = node.nodeValue;
      for (;;) {
        let best = null;
        for (const [id, re] of terms) { if (seen.has(id)) continue; const m = re.exec(text); if (m && (!best || m.index < best.m.index)) best = { id, m }; }
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
export function bindTooltip(tip, terms = TERMS) {
  const termById = Object.fromEntries(terms.map((t) => [t[0], t]));
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
export function renderGlossary(el, terms = TERMS) {
  el.innerHTML = terms.map((t) => `<div><b>${t[2]}</b> — ${t[3]}</div>`).join('');
}

/** Link terms in the elements matching `selectors`, bind the tooltip element, fill the glossary element.
 *  `terms` is the page's list: TERMS for the CNN pages (the default), VIT_TERMS for the ViT page. */
export function installTerms({ selectors, tip, gloss, terms = TERMS }) {
  linkTerms(selectors, terms);
  if (tip) bindTooltip(tip, terms);
  if (gloss) renderGlossary(gloss, terms);
}
