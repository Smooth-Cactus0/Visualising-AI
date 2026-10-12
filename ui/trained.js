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
