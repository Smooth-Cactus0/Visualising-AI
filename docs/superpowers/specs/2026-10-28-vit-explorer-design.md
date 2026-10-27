# Vision Transformer explorer: design

Date: 2026-10-28. Status: approved in conversation, section by section, and reviewed by Alexy; amended the same day to average the tokens instead of reading a class token (see the decisions).

## Goal

A page, `vit/`, that shows what the boxes in a ViT diagram compute, on the same 16×16 digits as the CNN pages: patches becoming tokens, position, one attention head written out, the MLP and residuals, the tokens averaged into a prediction. It delivers what the landing page's ViT section promised: patch embedding as a convolution whose stride equals its size, position added to each token, one attention head written out (queries against keys, softmax, a weighted sum of values), and the attention map as a receptive field that is global in one step.

Intent it serves (`project_notes.md`): "develop a good intuition on what different model architectures are, their strength and limitations".

## Decisions taken in the brainstorm

- **Sequence: explorer first, training page second.** `vit/` now; `vit-train/` later, reusing the CNN training page's three rows (forward, backward, update) and its many-steps band; 3D later still. Chosen over one combined page (the backward pass through softmax and Q·Kᵀ would land in the first iteration and compete with the attention walkthrough) and over an explorer only (it loses the position-table and data-hunger stories that only training can show).
- **The explorer runs on trained weights by default.** Unlike the CNN explorer, where hand-picked edge detectors carry the meaning, a random attention map is close to noise. So `core/vit.js` gets its backward pass now, and the shipped weights are trained offline by the same code.
- **Green is graded after attention.** Before attention, green is on or off as on the CNN pages. After attention, every token is upstream of every token, so on/off green would light the whole digit. Instead: the hovered token's own patch is full green (the residual carries it straight through), and every other token is green in proportion to its attention weight. The page says this is attention weight, not importance.
- **Two heads of 4 numbers each.** Both 16×16 maps are drawn; one is written out at a time with a switch. Their outputs are joined and multiplied by Wo, so the "multi-head" box of the diagrams becomes visible. If the toy learns two similar maps, the page says so.
- **Layout: the whole line as a pinned overview, stage sections below.** The overview is the forward pass on one line (it becomes the training page's forward row); the sections zoom into one part at a time at a readable size. Hovering in either lights both. Chosen over stage sections only (no single picture shows the token rows running straight through and crossing only at attention) and over the line only (too dense to write attention out). Wireframes: the brainstorm companion's `layout-v2.html` (local, under `.superpowers/`, not committed).
- **Attention drawn as a matrix product.** Token i's row runs straight through: its query, its row of the attention map, its output. Token j's key and value stand as column j above the map (Kᵀ and Vᵀ), so each row of the map weights the columns directly above it.
- **The tokens are averaged into the prediction; there is no class token.** Found while planning the backward pass: with one encoder block and a class token, only the class token's row reaches the prediction, so the other 16 rows after attention are computed but read by nothing, and their rows of the attention maps are never trained. Averaging the 16 tokens (as many current ViTs do) makes every row trained and every hover meaningful, and the average is the CNN's global average, which ties the two pages together. The page says the original ViT reads a class token instead, and why this toy does not. Chosen over keeping the class token with the unused rows drawn faint (most of the maps would be by-products) and over two encoder blocks (the middle of the page doubles, and the last block's rows are unused again).

## Grammar carried over from the CNN pages

- **Channel k is a line → token i is a line.** The 16 tokens are 16 rows and each keeps its height along the page; token i is patch i, in reading order. LayerNorm, the Q/K/V projections and the MLP act on one row at a time (in CNN terms, 1×1 convolutions); attention is the one step where rows cross.
- **Weights sit where they act**, between the tensor they read and the tensor they make, tinted as learned: the 8 patch kernels between the digit and the tokens; the position table; Wq, Wk, Wv stacked in one column; Wo; W₁ and W₂; W_head. Each matrix has its bias as a row beneath it, as the CNN training page draws the FC biases. LayerNorm scale and shift are thin tinted strips at their LayerNorm.
- **Yellow = the forward window**: the patch kernel's window jumping 4 pixels at a time; in attention, the query row and key column of the score being computed.
- **Green = computed from**, graded after attention as above.
- **At the head, only the direct link**: a class lights its column of W_head and the averaged vector; an entry of the average lights its column in every token of z₂, as the CNN's global-average entry lights its map. Neither traces back to the digit.

## The model

16×16 digit → 16 patches of 4×4 → patch embedding (16 → 8: 8 kernels of 4×4 at stride 4, plus a bias) → plus the learned position table (16×8) = z₀ → LayerNorm → Q, K, V (8×8 each, with biases), split into 2 heads of 4 → per head A = softmax(Q·Kᵀ / 2) (2 = √4) and A·V → heads joined → Wo (8×8) → plus z₀ = z₁ → LayerNorm → W₁ (8×32) → GELU → W₂ (32×8) → plus z₁ = z₂ → average of the 16 tokens → LayerNorm → W_head (8×10) → softmax over 10 classes.

Every matrix has a bias. 1,242 weights, three times the CNN's 414. The three LayerNorms keep their learned scale and shift (48 numbers), as real ViTs do. One encoder block; real ViTs stack 12, and the page says so.

## Code

- **`core/vit.js`**, pure functions in the style of `core/train.js`, no DOM:
  - `forward(weights, image, { position = true, order })` returns every intermediate the page draws: the image as the model sees it, patches, patch tokens, z₀, the LayerNorm outputs, Q, K, V, scores and attention per head, the joined head outputs, z₁, the MLP hidden layer before and after GELU, z₂, the average, its LayerNorm, scores and probabilities. `order` is a permutation of the 16 patches, for the shuffle demo; `position: false` leaves the position table out.
  - `lossAndGrad`, `init(seed)` (seeded), the update steps (SGD and Adam; which one trains the shipped weights is measured, see below), `predict`, `accuracy`, and `tensor(key, fwd, weights)` with the shapes in `SHAPES`, so the page and the tests name every drawn tensor the same way.
  - `vitTrace(fwd, probe)`: for a hovered value (`{ t, r, c }`, `t` a key of `SHAPES`), a strength from 0 to 1 for every upstream cell, which is the green. Before attention, 1 for the cells it is computed from (its patch, the kernel and bias that made its column, its position entry) and 0 elsewhere. After attention, 1 for the hovered token's own row and patch; every other token j gets A[i, j] scaled so the most-attended other token reaches 0.9, leaving full strength (and an outline) to the hovered token's own; a single head's output follows that head's map; values after the heads are joined follow the mean of the two maps. An attention weight A[i, j] lights query i and key j in full and every other key faintly, since the softmax divides by all of them. A weight lights the column it reads and the column it makes; the head follows the direct-link rules above.
- **`tools/train_vit.mjs`** (Node) trains on `data/mnist16.bin` through `core/data.js` and `core/vit.js` and writes **`models/vit16.json`** (about 20 KB): the weights, rounded to 6 significant digits, plus seed, optimiser, learning rate, batch, passes, training and held-out accuracy of the rounded weights, and date. Rerunning it reproduces the file.
- **Optimiser.** Plain SGD first, so that the later training page keeps the CNN's update row w′ = w − lr·∂L/∂w. If SGD falls clearly short of Adam (more than 3 points of held-out accuracy after 10 passes), the shipped weights use Adam and the training page's update row is decided when that page is built. Held-out accuracy per pass for both is measured on the same seed and the same 3,000 digits as the CNN, and recorded next to the CNN's 80 to 85%.
- **`ui/terms.js`**: `installTerms` takes an optional `terms` list, defaulting to today's `TERMS` so the CNN pages do not change. A new `VIT_TERMS` list covers patch embedding, patch, position embedding, token, attention head, attention, query/key/value, LayerNorm, residual connection, MLP, GELU and the average of the tokens, plus entries reworded for this page: kernel (4×4 here), stride, softmax, receptive field, weights, held-out. The tooltip and Terms-section mechanism stays shared.
- **`ui/grid.js`** draws and hit-tests the grids, as on the 2D and training pages. It gains four opt-in options the CNN pages do not use: `glow` (a strength per cell, the graded green, with an outline around the full-strength cells), `win` (a yellow rectangle), `blocks` (dashed lines every n cells, the patch grid) and `lines: false` (no inner grid lines, for small cells).
- **`vit/index.html`** holds the page and its own module, like the other pages. The overview and the sections are built by one small builder, so a section is the overview's columns at a larger cell size. If the script grows well past the training page's size (about 700 lines), the drawing code moves to a module beside it.

## The page

- **Top**: breadcrumb to the landing page, a short lede, the colour key.
- **Pinned overview**: the controls (a digit picker with real held-out MNIST digits, one per class, plus "another"; trained weights need real digits rather than hand-drawn presets; weights trained | random; position on | off; "Shuffle patches" and "Restore order"), then the whole forward pass on one line, token i on row i, weights tinted where they act, residuals as arcs beneath, then the hover readout. It stays on screen while the sections scroll, and scrolls sideways if wider than the window. Clicking a part of it scrolls to its section. Below 700 px it is not pinned.
- **Sections**:
  1. *Patches become tokens.* The digit with its patch grid, the 8 kernels with their biases, the tokens. The yellow window jumps 4 pixels at a time over the 16 positions, with play and step as on the CNN 2D page; clicking a kernel picks the column written out. The readout writes one token value out: 16 products plus the bias. Text: the CNN's kernel, jumping instead of sliding.
  2. *Position.* The tokens, the position table, z₀. The shuffle demo: with position off, shuffling the patches leaves the prediction unchanged; with it on, the prediction moves. The readout compares the two orders. Text: attention by itself does not know where a patch came from.
  3. *One attention head, written out.* A head 1 | head 2 switch; LayerNorm output → that head's columns of Wq, Wk, Wv → Q; Kᵀ and Vᵀ above the map; the map; the output; beside it, the digit shaded by the chosen token's attention. Yellow walks along one query row: query row i against key column j gives score (i, j), then the softmax over the row, then the weighted sum of the value columns; the readout writes the arithmetic at each step. Text: the attention map is a receptive field that is global in one step, and it costs the square of the number of tokens.
  4. *Mix the heads, residual, MLP.* Heads joined → Wo → plus z₀ → z₁ → LayerNorm → W₁ → GELU → W₂ → plus z₁ → z₂. Text: each of these works on one token at a time, unlike attention.
  5. *Average the tokens, then predict.* z₂ → average → LayerNorm → W_head → the 10 probabilities, with the wording on where the weights came from and the note on the class token.
  6. *Terms.*
- **Hover**, in the overview and the sections alike, follows `vitTrace`; the readout gives the real numbers (for attention, the actual weights, not the scaled green).
- Light and dark through the shared colour tokens; canvases sized in CSS as well as pixels; reduced motion makes the sweeps jump to their end and scrolling instant; no 3D in this iteration.

## Weights and wording

- **Trained** (default): loaded from `models/vit16.json`. Wherever a prediction appears, the page says "trained offline on 3,000 digits, N% on 1,000 held-out digits", with the numbers read from the file.
- **Random**: the seeded starting weights, with the line that attention is learned and before training it is noise.
- **If `models/vit16.json` fails to load**, the page falls back to random weights and says so in a banner, as the CNN pages do for stored weights. It never shows random weights as if they were trained.

## Landing page

The ViT section's "What the page will show" list becomes a card for `vit/`, and its badge becomes the page count ("1 page"), following the rule in CLAUDE.md; its chip at the top and the lede follow. The schematic, its caption and its label lose the class token: the tokens are averaged. The green key adds that after attention its strength is the attention weight. The "builds on" line stays. CLAUDE.md's layout section and `project_notes.md` gain the new files once the page is built.

## Checks

- **`node core/vit.test.mjs`**: shapes (16 patches, 16 tokens, two 16×16 maps, 10 probabilities, every `SHAPES` entry); every attention row sums to 1; LayerNorm output has mean 0 and variance 1 per token before its scale and shift; the patch embedding equals a 4×4 stride-4 convolution computed by a separate plain loop; with position off, shuffling the patches leaves the probabilities unchanged to 1e-12, and with position on they change; gradients against central finite differences for every weight, with and without position (smooth everywhere, so a maximum normalised error below 1e-5, stricter than the CNN's); SGD and Adam lower the loss on one digit; `vitTrace` is on/off and exactly the patch before attention, gives the own patch 1 and the other tokens 0.9 × their attention relative to the most-attended one after it, follows only its head's map for a single head's output and the mean map after the heads are joined, and keeps the head to its direct links; the same seed gives the same weights.
- **Training**: `tools/train_vit.mjs` prints held-out accuracy per pass for SGD and for Adam; the figures, the optimiser chosen and why go into the notes and into `models/vit16.json`.
- **Page, in headless Edge**: no console errors in light and dark at 1400, 1000 and 500 px; the overview pinned at the two wide sizes and not at 500 px, with no sideways overflow of the page; driven from the page, a hover in a section lights the same cells in the overview and the reverse; the patch sweep visits all 16 positions; the attention walk completes its row; shuffling gives identical probabilities with position off and different ones with position on; the trained | random switch changes the wording; a missing `models/vit16.json` shows the fallback banner; screenshots of each section and of the graded green over the digit, viewed and corrected.
- **Nothing else changes**: the three CNN pages, dumped before and after the `ui/terms.js` and `ui/grid.js` changes with the seeded `Math.random` shim, give identical DOMs; the landing page renders the new card in both themes and at phone width.

## Left for later

- `vit-train/`: forward, backward and update rows for the ViT; the position table going from noise to a 2D grid pattern over training; held-out accuracy against the CNN on the same digits (the data-hunger lesson); the hand-off of trained weights back to `vit/` through local storage under a ViT key of its own.
- A class-token variant with two encoder blocks, the original ViT's form.
- A 3D view: the tokens reshaped are a 4×4×8 map, a CNN feature map's shape, a natural bridge to CoAtNet.
- A patch-size switch (2×2 patches: 64 tokens, 64×64 maps) to show the quadratic cost; it needs a second set of trained weights.
