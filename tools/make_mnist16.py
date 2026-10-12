#!/usr/bin/env python3
"""Make data/mnist16.bin: MNIST digits area-resampled to 16x16, 300 training and 100 test digits per class.

Standard library only. The four MNIST files are downloaded once into tools/.cache/ (git-ignored).
Usage: python tools/make_mnist16.py [--cache tools/.cache] [--out data/mnist16.bin]
Format of the output (little-endian): bytes 0-3 'MN16', 4-5 nTrain (uint16), 6-7 nTest (uint16), 8 side (16),
9-15 zero; then nTrain labels (1 byte each), nTrain x side*side pixels (row-major, 0 background .. 255 full ink),
nTest labels, nTest x side*side pixels.
"""
import argparse, gzip, os, random, struct, urllib.request

MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {
    'train_img': 'train-images-idx3-ubyte.gz', 'train_lab': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz', 'test_lab': 't10k-labels-idx1-ubyte.gz',
}
SIDE, PER_CLASS = 16, {'train': 300, 'test': 100}
SEED = 1


def fetch(cache, name):
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        print('downloading', name)
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def read_idx(path):
    """Return (count, rows, cols, data bytes) for an IDX image file, or (count, 1, 1, data) for labels."""
    d = gzip.open(path).read()
    magic, n = struct.unpack('>II', d[:8])
    if magic & 0xff == 3:
        r, c = struct.unpack('>II', d[8:16])
        return n, r, c, d[16:]
    return n, 1, 1, d[8:]


def resample(img, src, side=SIDE):
    """Box-filter a src x src byte image down to side x side; each output cell averages the input area it covers."""
    s = src / side
    out = []
    for i in range(side):
        for j in range(side):
            y0, y1, x0, x1 = i * s, (i + 1) * s, j * s, (j + 1) * s
            acc = 0.0
            for y in range(int(y0), min(src, int(y1) + 1)):
                wy = min(y1, y + 1) - max(y0, y)
                if wy <= 0:
                    continue
                for x in range(int(x0), min(src, int(x1) + 1)):
                    wx = min(x1, x + 1) - max(x0, x)
                    if wx > 0:
                        acc += wy * wx * img[y * src + x]
            out.append(max(0, min(255, int(round(acc / (s * s))))))
    return bytes(out)


def pick(labels, per_class, rng):
    """per_class indices of each digit 0-9, chosen at random, then shuffled so the file is not sorted by class."""
    by_class = {d: [] for d in range(10)}
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    chosen = []
    for d in range(10):
        chosen += rng.sample(by_class[d], per_class)
    rng.shuffle(chosen)
    return chosen


def build(cache, out):
    rng = random.Random(SEED)
    sets = {}
    for split in ('train', 'test'):
        n, r, c, px = read_idx(fetch(cache, FILES[split + '_img']))
        nl, _, _, lab = read_idx(fetch(cache, FILES[split + '_lab']))
        assert n == nl and r == c, 'unexpected MNIST layout'
        idx = pick(lab, PER_CLASS[split], rng)
        labels = bytes(lab[i] for i in idx)
        pixels = b''.join(resample(px[i * r * c:(i + 1) * r * c], r) for i in idx)
        sets[split] = (labels, pixels)
        print(split, len(idx), 'digits')
    header = b'MN16' + struct.pack('<HHB', len(sets['train'][0]), len(sets['test'][0]), SIDE) + bytes(7)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'wb') as f:
        f.write(header + sets['train'][0] + sets['train'][1] + sets['test'][0] + sets['test'][1])
    print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default=os.path.join('tools', '.cache'))
    ap.add_argument('--out', default=os.path.join('data', 'mnist16.bin'))
    a = ap.parse_args()
    build(a.cache, a.out)
