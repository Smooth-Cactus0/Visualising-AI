# CNN training page: design

Date: 2026-10-12. Status: approved in conversation, awaiting Alexy's review of this document.

## Goal

A third page, `cnn-train/`, that shows how training works on the same toy CNN the 2D and 3D pages explore. Real MNIST digits stream in at the network's 16×16; the page shows one training step in full (forward, loss, the error flowing back, the gradient of every weight, the nudge) and, below it, the arc of many steps (loss and accuracy curves, the layer-1 kernels turning from noise into edge detectors, accuracy on digits never trained on). Weights trained here can be loaded into the 2D and 3D pages.

Intent it serves (Alexy, 2026-10-12): "make it interactive, to exemplify how training actually works by showing different digits in succession like the MNIST dataset and seeing how the weights are updated". Both the step and the arc matter: "the step is important but seeing weights being actualised over passes is necessary too".

## Decisions taken in the brainstorm

- Both a step mode and a play mode, on one page, as two stacked bands ("One step", "Many steps"), plus a Terms section. Chosen over a mode switch (hides one while the other runs) and over two columns (the forward pipeline would wrap; narrow windows stack anyway).
- The network stays at 16×16 with 4 → 8 kernels and an 8×10 FC head, at stride 1, padding 1, dilation 1, max pooling, ReLU. MNIST digits are area-resampled from 28×28. Why: the trained weights then plug into the existing pages. Cost, stated on the page: this toy reaches about 80 to 85% on held-out digits after a few passes (measured 2026-10-12: 82% after three passes, 83% after ten, 86% from the edge-detector start), far below a real LeNet. The brainstorm's guess of 85 to 92% was too high.
- Training is live in the browser with plain SGD, not a replayed offline run. Why: learning rate and batch size stay interactive, and the gradient arithmetic is real for the digit on screen.
- The weights are drawn as triplets per layer: weight, gradient, updated weight, with the update written out on hover. An arrow overlay on the weight grid is kept in reserve as a possible later toggle.
- Initialisation is random for all three layers so the kernels must learn edges; a second button starts layer 1 from the four hand-picked edge detectors for comparison.

## Files

| Path | Role |
|---|---|
| `tools/make_mnist16.py` | Standard-library Python. Downloads the four MNIST files (mirror `https://ossci-datasets.s3.amazonaws.com/mnist/`, cached under `tools/.cache/`, git-ignored), parses the IDX format, picks 300 training and 100 test digits per class with `random.Random(1)`, area-resamples each 28×28 digit to 16×16 (box filter with fractional overlap), scales to bytes 0–255, writes `data/mnist16.bin`. |
| `data/mnist16.bin` | The digits, about 1 MB. Format below. Committed. |
| `data/README.md` | Provenance (MNIST, LeCun, Cortes and Burges), licence (CC BY-SA 3.0), the mirror used, the selection rule, the regeneration command. |
| `core/data.js` | `parseMnist16(buffer)`: returns `{ N, train, test }`, each set `{ n, labels: Uint8Array, pixels: Uint8Array }`, and `image(set, i)` returning a fresh `Float32Array(N*N)` scaled to 0–1. Pure. |
| `core/train.js` | The training maths, pure functions, no DOM. See "Core maths". |
| `core/train.test.mjs` | Gradient checks and a short training run. See "Checks". |
| `ui/grid.js` | The canvas grid drawing and cell hit-testing now inline in the 2D page: `drawGrid`, `cellAt`, `divColor`, `grayColor`, `maxAbs`, `fmt`, `isDark`, `cssVar`. The 2D page imports it with no change in behaviour (the `padBorder` option takes the padding as `pad` instead of reading a closure). |
| `cnn-train/index.html` | The page. |
| `ui/terms.js` | Gains the training terms (list below). |
| `cnn-2d/index.html`, `cnn-3d/index.html` | Load trained weights from local storage when present; banner; wording switch. |
| `index.html` | Third link. |
| `.gitignore` | Adds `tools/.cache/`. |

### `data/mnist16.bin` format

Little-endian. Bytes 0–3: ASCII `MN16`. Bytes 4–5: number of training digits (uint16). Bytes 6–7: number of test digits (uint16). Byte 8: side length (16). Bytes 9–15: zero. Then: training labels (one byte each), training pixels (side² bytes per digit, row-major, 0 = background, 255 = full ink), test labels, test pixels. With 3,000 and 1,000 digits the file is 1,028,016 bytes. The page fetches it once with `fetch` and keeps it in memory.

## Core maths (`core/train.js`)

All functions are pure; weights and gradients use the same shapes as `core/cnn.js`: `bank` (4 × 9), `l2` (8 × 4 × 9), `W` (8 × 10), `B` (10). Maps are `Float32Array`; the forward pass is `forward` from `core/cnn.js` with `activation: 'relu'`, `poolMode: 'max'`, stride 1, pad 1, dil 1.

- `initWeights({ seed, layer1: 'random' | 'edges' })` → `{ bank, l2, W, B }`. Seeded PRNG (mulberry32). He-uniform, ±√(6/fan_in): layer 1 ±0.82, layer 2 ±0.41, FC ±0.87; biases 0. With `layer1: 'edges'` the bank is `defaultBank` from `core/cnn.js`. (±1/√fan_in was tried first: one pass reached 29% held-out and ten passes 80%; He-uniform reaches 82% by pass three.)
- `lossAndGrad(weights, input, label, { poolMode = 'max' })` → `{ S, loss, grads, back }` where `S` is the forward state, `loss = −log p[label]` (cross-entropy on the softmax), `grads = { bank, l2, W, B }` with the shapes of the weights, and `back = { scores, gap, pool2, conv2, pool1, conv1 }` the gradient of the loss with respect to each activation, kept for drawing (`conv1` and `conv2` are the gradients at the pre-ReLU conv outputs).
  - scores: `∂L/∂s = p − y`.
  - FC: `∂L/∂W[f][c] = gap[f] · (p[c] − y[c])`, `∂L/∂B[c] = p[c] − y[c]`, `∂L/∂gap[f] = Σ_c W[f][c] · (p[c] − y[c])`.
  - Global average: `∂L/∂pool2_f[q] = ∂L/∂gap[f] / Q2²`.
  - Max pool: the gradient of each 2×2 block goes to the position of its maximum (first on ties), zero elsewhere; average pool: a quarter to each. Determined from the activation map the pool read.
  - ReLU: multiply by 1 where the pre-activation was > 0, else 0.
  - Conv (stride 1, pad 1, dil 1), for a map `x` of side `n`, kernel `k`, output gradient `g`: `∂L/∂k[a][b] = Σ_{i,j} g[i][j] · x[i+a−1][j+b−1]` (x taken as 0 outside), and `∂L/∂x[r][c] = Σ_{a,b} g[r−a+1][c−b+1] · k[a][b]` (g taken as 0 outside). Layer 2 sums the input gradient over the 8 kernels into each of the 4 pool-1 maps; layer 1 needs no input gradient.
- `averageGrads(list)` → element-wise mean, for batches.
- `sgdStep(weights, grads, lr)` → new weights `w − lr · g`, fresh arrays, inputs untouched.
- `predict(weights, input)` → `{ prob, argmax }`; `accuracy(weights, set, indices)` → fraction correct.
- `shuffledOrder(n, seed)` → a permutation, for one pass over the training set.

## The page

Theme, fonts, colour tokens, dotted terms and layout rules follow the 2D page. Canvases are sized in CSS as well as pixels. Reduced motion disables the update animation. All text that mentions a prediction says the weights are being trained on this page.

### Header

Title "CNN explorer, training", a lede stating what the page shows and the honest ceiling (about 80 to 85% on held-out digits for this toy, after a few passes), and the legend: green = receptive field and, on this page, the two sources of a hovered weight's gradient. The yellow forward window of the other pages is not used here.

### Band 1: One step

A phase strip `digit → forward → backward → update` marks which phases have run for the current digit. Buttons: **Step** (runs the next phase), **Next digit** (finishes the current digit, loads the next one in the pass order and runs its forward phase). Keyboard: space = Step, enter = Next digit.

Forward row (visible after the forward phase): the digit (16×16, grey, with its label), conv 1 + ReLU (4 maps), pool 1 (4), conv 2 + ReLU (8), pool 2 (8), the global-average vector, the FC heatmap, the class bars with the true class ringed and the top class filled; readout `loss = −log p[true] = …`. Hovering a map cell traces its receptive field through the forward row, as on the 2D page; hovering a class bar shows `∂L/∂score[c] = p[c] − y[c] = …` (once the backward phase has run, else just `p[c]`).

Backward row (visible after the backward phase), same positions, diverging colours: `∂L/∂conv1` (4 maps), `∂L/∂pool1`, `∂L/∂conv2` (8), `∂L/∂pool2`, `∂L/∂gap`, and `∂L/∂scores` as signed bars with `p − y` written out. Caption: the error starts at the bars, ReLU and max-pool gate it, conv spreads it. Hovering a backward cell shows its value and which map it belongs to.

Weights band (triplets, visible after the backward phase; before it only the weight grids): per layer, weight · gradient · updated weight. Layer 1: four 3×3 triplets, large. Layer 2: 8 kernels × 4 slices, small. FC: three 8×10 grids plus three 1×10 bias strips. The "updated" grid shows `w − lr·∂L/∂w` as a preview; the update phase makes it the weight (a 300 ms colour interpolation unless reduced motion), clears the gradient grids and advances the counters. Hovering any weight cell writes the arithmetic in the readout and lights the two sources in green:

- FC weight (f, c): `∂L/∂W[f,c] = gap[f] × (p[c] − y[c]) = … × … = …` then `W′ = W − lr·∂L/∂W = … − …×… = …`; sources: global-average entry f and class bar c.
- FC bias c: `∂L/∂b[c] = p[c] − y[c]`.
- Layer-2 kernel k, slice s, cell (a, b): `∂L/∂w = Σ over 64 positions of ∂L/∂conv2_k[i,j] × pool1_s[i+a−1, j+b−1] = …`, update line; sources: backward map conv2_k and forward map pool1_s.
- Layer-1 kernel k, cell (a, b): the same with 256 positions, the input as the source map.

With batch size above 1, the band shows the last digit of the batch and the averaged gradient, and the readouts say "averaged over N digits".

### Band 2: Many steps

Controls: **Play / Pause**; speed 10, 50 or 200 digits per second; learning-rate slider (log scale 0.001 to 0.5, default 0.1, value shown; 0.05 learned too slowly in the measurements); batch 1, 8 or 32; **Reset** (new random weights; the seed starts at 1 and each Reset adds 1; counters and curves cleared; seed shown); **Start layer 1 from the edge detectors** (the same reset with the current seed, but layer 1 set to the hand-picked bank); counters: digits seen, passes completed (one pass = the 3,000 training digits in a seeded shuffled order; a new order each pass).

Play runs Next-digit steps in a `requestAnimationFrame` loop within a time budget per frame, draws the step band at most ten times a second, and draws the curves at the same rate. With batch N, each update consumes N digits.

Curves (canvas, theme-aware, axes with a few ticks): loss per digit as faint points with an exponential moving average line (α = 0.02); accuracy with two lines, running training accuracy over the last 200 digits and test accuracy on the 1,000 held-out digits measured every 200 digits and at reset (1,000 forward passes each time, which is why not every digit). Both plotted against digits seen; the x-axis grows.

Kernels over time: the four layer-1 kernels snapshotted when the counter reaches 0, 100, 300, 1,000, 3,000 and 10,000 digits, drawn as a strip of 3×3 grids with "now" at the end.

Recent digits: the last 20 digits as 16×16 thumbnails with a tick or a cross for the prediction made when they were seen.

Buttons: **Use these weights in the 2D / 3D explorers** and **Download JSON** (file `cnn16-trained.json`).

### Terms

Added to the shared array, so they get dotted definitions here and entries in every Terms section: loss (cross-entropy), gradient, backpropagation, learning rate, update (SGD step), batch, pass, held-out test set, initialisation.

## Hand-off to the explorer pages

Local-storage key `dl-explorer.weights.v1` holds `{ version: 1, N: 16, bank, l2, W, B, meta: { digits, passes, testAcc, lr, date } }`. On load, the 2D and 3D pages use these weights if the key is present, show a banner under the title ("Using weights trained on the training page: N digits, test accuracy X%. Back to the hand-picked kernels.") whose link removes the key and reloads, and replace their "weights are random" wording with "weights trained on the training page". Reroll buttons and kernel editing behave as before. The JSON download is the same object; committing one under `models/` is a separate, later decision.

## Checks

- `node core/train.test.mjs`: (1) for a tiny random instance, every gradient (`bank`, `l2`, `W`, `B`) matches central finite differences with relative error below 1e-4 (maps copied to Float64 for the check; the step 1e-3); (2) 50 SGD steps on one digit at lr 0.1 bring its loss below half the starting loss; (3) `parseMnist16` reads `data/mnist16.bin` with the expected counts and label range; (4) three passes over the 3,000 training digits at lr 0.1, batch 1, seed 1, reach above 75% on the 1,000 held-out digits (measured: 82%). The run time of (4) in Node is reported in the notes.
- Browser (headless Edge recipe): the page loads without console errors; a scripted Step sequence produces the expected phase strip and readouts; a scripted play of 300 digits lowers the smoothed loss; screenshots of both bands at 1400 and 1000 px widths.
- The 2D page's move to `ui/grid.js`: byte-identical rendered DOM before and after with seeded `Math.random`, as in step 1.
- The explorer pages with and without the stored weights: banner present or absent, wording switched, DOM identical to today when the key is absent.

## Left for later, on purpose

The arrow overlay on weight grids; MNIST digits as input presets on the explorer pages; a confusion heatmap; drawing every digit of a batch; momentum or Adam; 28×28.
