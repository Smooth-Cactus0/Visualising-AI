// core/data.js — reader for data/mnist16.bin (format in data/README.md). Pure, no DOM.

/** Parse an MN16 buffer. Returns views into it (no copy): { N, train: { n, labels, pixels }, test: { ... } }. */
export function parseMnist16(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.length < 16 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'MN16') throw new Error('not an MN16 file');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const nTrain = dv.getUint16(4, true), nTest = dv.getUint16(6, true), N = u8[8];
  const px = N * N;
  let off = 16;
  const set = (n) => {
    const labels = u8.subarray(off, off + n); off += n;
    const pixels = u8.subarray(off, off + n * px); off += n * px;
    return { n, labels, pixels };
  };
  const train = set(nTrain), test = set(nTest);
  if (off !== u8.length) throw new Error(`MN16 size mismatch: expected ${off} bytes, got ${u8.length}`);
  return { N, train, test };
}

/** Digit i of a set as a fresh Float32Array scaled to 0–1. */
export function image(set, i) {
  const px = set.pixels.length / set.n;
  const out = new Float32Array(px);
  const base = i * px;
  for (let q = 0; q < px; q++) out[q] = set.pixels[base + q] / 255;
  return out;
}
