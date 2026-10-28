// ui/trained.js — hand-off of weights trained on the training page to the explorer pages, through local storage.
export const KEY = 'dl-explorer.weights.v1';

export function saveTrained(o) { localStorage.setItem(KEY, JSON.stringify(o)); }

/** The stored weights, or null when there are none or they are not in the expected shape. */
export function loadTrained() {
  try {
    const s = localStorage.getItem(KEY); if (!s) return null;
    const o = JSON.parse(s);
    return o && o.version === 1 && o.N === 16 && Array.isArray(o.bank) && Array.isArray(o.l2) && Array.isArray(o.W) && Array.isArray(o.B) && o.meta ? o : null;
  } catch { return null; }
}

export function clearTrained() { try { localStorage.removeItem(KEY); } catch { /* storage blocked: nothing to clear */ } }

/** A banner after `anchor` saying the page uses trained weights; its link removes them and reloads. */
export function showTrainedBanner(anchor, o) {
  const d = document.createElement('div'); d.className = 'banner';
  const acc = o.meta.testAcc == null ? '' : `, held-out accuracy ${Math.round(o.meta.testAcc * 100)}%`;
  d.innerHTML = `Using weights trained on the <a href="../cnn-train/">training page</a> (${o.meta.digits} digits${acc}). <a href="#" data-clear>Back to the hand-picked kernels</a>.`;
  d.querySelector('[data-clear]').onclick = (e) => { e.preventDefault(); clearTrained(); location.reload(); };
  anchor.after(d);
}

// The ViT's own hand-off, from vit-train/ to vit/: { version: 1, arch: 'vit16', weights, meta }, the shape of
// models/vit16.json, with meta.source 'vit-train'. The ViT pages pass core/vit.js's isWeights to check the weights,
// so this module stays free of the model code.
export const VIT_KEY = 'dl-explorer.vit-weights.v1';

export function saveTrainedViT(o) { localStorage.setItem(VIT_KEY, JSON.stringify(o)); }

/** The stored ViT weights, or null when there are none or isWeights rejects them. Also checks the meta figures the pages print (digits, testAcc, lr are finite numbers), so a malformed hand-off is ignored rather than breaking the page. */
export function loadTrainedViT(isWeights) {
  try {
    const s = localStorage.getItem(VIT_KEY); if (!s) return null;
    const o = JSON.parse(s);
    return o && o.version === 1 && o.arch === 'vit16' && o.meta
      && Number.isFinite(o.meta.digits) && Number.isFinite(o.meta.testAcc) && Number.isFinite(o.meta.lr)
      && isWeights(o.weights) ? o : null;
  } catch { return null; }
}

export function clearTrainedViT() { try { localStorage.removeItem(VIT_KEY); } catch { /* storage blocked: nothing to clear */ } }

/** A banner after `anchor` saying the ViT page uses weights from the training page; its link goes back to the shipped ones. */
export function showTrainedViTBanner(anchor, o) {
  const d = document.createElement('div'); d.className = 'banner';
  const acc = Number.isFinite(o.meta.testAcc) ? `, held-out accuracy ${Math.round(o.meta.testAcc * 100)}%` : '';
  d.innerHTML = `Using weights trained on the <a href="../vit-train/">ViT training page</a> (${o.meta.digits.toLocaleString('en-NZ')} digits${acc}). <a href="#" data-clear>Back to the shipped weights</a>.`;
  d.querySelector('[data-clear]').onclick = (e) => { e.preventDefault(); clearTrainedViT(); location.reload(); };
  anchor.after(d);
}
