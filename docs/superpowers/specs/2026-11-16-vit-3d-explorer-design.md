# Vision Transformer 3D explorer: design

Date: 2026-11-16. Status: approved in conversation ("yes to all" on the four parts); awaiting Alexy's review of this document.

## Goal

A page, `vit-3d/`, that shows the toy ViT of `vit/` as blocks of cubes, as `cnn-3d/` does for the CNN. It shows the one thing the 2D pages cannot: where in the image each token sits and where it looks. It is the first of two builds; the second, a 2D | 3D switch on the training page's "One step" band, gets its own spec and reuses this scene.

Intent it serves (`project_notes.md`): "a bigger tool which allows us to switch between models and representation (side-by-side or depth, 2D or 3D)".

## Decisions taken in the brainstorm

- **Two builds, mirroring the CNN.** A separate `vit-3d/` explorer page first, then a 2D | 3D switch on `vit-train/`. Why: Alexy chose A. `vit/` already carries a pinned overview and five sections, and a WebGL view would have to fit in or replace them. The training view needs the explorer's scene anyway.
- **Geometry: tokens on the patch grid, and each attention map as a grid of little maps.**
  - The 16 tokens sit at their patches' places and form a 4×4 block, 8 deep: the shape of a CNN feature map.
  - Each head's attention map is a 16×16 slab arranged spatially: at token i's place sits token i's own 4×4 map of where it looks. So both indices of an attention weight are places in the image.
  - Why: Alexy chose C from mock-ups in the brainstorm companion. Links alone would leave the attention map without a body. Token slabs (the 2D grids stood up) would put nothing where it is in the image and add little over 2D.
- **The whole chain is drawn, weights included.** Why: Alexy chose A. It keeps "weights sit where they act", and the training view needs every block, so the scene is described once.
- **Interaction: hover, a patch sweep, a pinned probe, and the explorer's controls.** Why: Alexy chose A. A hover is lost when the camera orbits, so the main picture needs a pin. The score-by-score attention walk stays on the 2D page, where the arithmetic reads better.

## Part 1: the scene

One row of blocks in flow order, drawn by `ui/blocks3d.js`, one cube per number. Dimensions are rows × columns × slices (slices run into depth).

- **Digit:** 16×16×1, grey.
- **Embedding:**
  - the patch kernels, 4×4×8 (kernel d is slice d), with their bias as a strip beneath;
  - the patch tokens, 4×4×8;
  - the position table, 4×4×8 (each slot's row at its own place);
  - z₀, 4×4×8;
  - LayerNorm 1's scale and shift, a 2×1×8 strip;
  - its output, 4×4×8.
- **Attention:**
  - Wq, Wk, Wv: 8×8×1 slabs stacked in one column, each with its bias strip;
  - Q, K, V: 4×4×8 blocks stacked beside them, head 1 the front four slices, head 2 the back four;
  - the attention block, 16×16×2: one slab per head, front and back. The cube for weight (i, j) sits in token i's 4×4 group, at token j's place inside it;
  - the joined head outputs, 4×4×8.
- **Mix and MLP:** Wo (8×8×1 and bias), z₁, LayerNorm 2's strip and its output, W₁ (8×32×1 and bias), the hidden block (4×4×32), W₂ (32×8×1 and bias), z₂.
- **Head:** the average (1×1×8), the final LayerNorm's strip, the normalised average (1×1×8), W head (8×10×1 and bias), the 10 class cubes (10×1×1).
- **Weights** carry the accent frame, as on the CNN training page's 3D view.
- **Layout:** "side by side | depth" (remembered in local storage), as on `cnn-3d/`.
- **Camera presets:** whole · embedding · attention · MLP · head, each framed by fitting the camera to its blocks.

## Part 2: interaction

- **Hover a cube.** The cube maps to a tensor cell, the same probe the 2D pages use, so `vitTrace` gives the green unchanged.
  - Each lit tensor cell becomes a highlight box; boxes that share a strength along a token's column merge into one.
  - Strength sets the box's opacity, so after attention the hovered token's column is full green and every other token's column is graded by its attention weight.
  - For an after-attention probe, a fan of links runs from the hovered token's little map in the attention block to the tokens it read, their opacity the same strength.
  - The readout is the 2D explorer's.
- **Pin.** A click (without a drag) keeps the hovered probe lit while the camera moves; a click on empty space releases it. While a probe is pinned, hovering another cube shows that cube's probe, and leaving it returns to the pinned one.
- **Patch sweep.** Play and step over the 16 positions: a yellow window on the digit's 4×4 patch, a yellow box on the matching token column and on the kernels. Reduced motion jumps to the end.
- **Controls,** with the same state and wording rules as `vit/`:
  - a digit picker (held-out digits, one per class, plus "another");
  - weights trained | random;
  - position on | off;
  - "Shuffle patches" and "Restore order".
- **Weights and wording.**
  - Weights handed off from the training page load with the banner.
  - Wherever the prediction appears, the page says what the accuracy was measured on.
  - If `models/vit16.json` fails to load, the page runs on random weights and says so.

## Part 3: code

- **`core/vit3d.js`** (pure, tested in Node):
  - `VIT_BLOCKS`: for each drawn tensor, its key, its rows × columns × slices, whether it is a weight, and its group (embedding, attention, MLP, head);
  - `cubeToCell(key, r, c, ch)` and `cellToCube(key, row, col)`: the two-way mapping between a block's cube and a tensor cell.
    - Token tensors: token t sits at (t ÷ 4, t mod 4), and its column is the slice.
    - Attention: weight (i, j) of head h sits at row (i ÷ 4)·4 + (j ÷ 4), column (i mod 4)·4 + (j mod 4), slice h.
    - Matrices and biases are flat, one slice.
- **`ui/vitwords.js`** (new): what `vit/` carries inline today and both pages need:
  - loading the weights (the shipped file, or a hand-off, with the fallback);
  - the wording of what the accuracy was measured on;
  - the probe readout.

  `vit/index.html` imports it, and is checked unchanged.
- **`ui/blocks3d.js`:** two opt-in additions the CNN pages do not call:
  - highlight boxes that take a strength (opacity);
  - `fit(ids, …)`, a camera fit to a set of blocks.

  The CNN pages' scenes are checked unchanged. The camera-fit code in `cnn-train/` is left as it is; moving it onto `fit` is a later tidy-up.
- **`vit-3d/index.html`:** the page: breadcrumb, lede, colour key, controls, the view with its presets and layout switch, the readout, Terms (`VIT_TERMS`).
- **`index.html`:** a "3D explorer" card in the ViT section; its badge and chip become "3 pages".

## Part 4: checks

- **Node** (`core/vit3d.test.mjs`):
  - every `SHAPES` key drawn on the 2D line has a block, with matching cell counts;
  - `cubeToCell` and `cellToCube` round-trip for every cube of every block;
  - the attention mapping puts weight (i, j) in token i's group at token j's place.
- **Headless Edge,** with three.js served locally:
  - no console errors, and a live WebGL context;
  - a picked cube gives the same probe and the same readout text as `vit/` for the same cell;
  - a known probe gives the expected number of highlight boxes and links;
  - the sweep visits 16 positions;
  - a pinned probe survives a camera move, and a click on empty space releases it;
  - the layout switch and each camera preset keep their blocks inside the view (checked from projected corners);
  - frames rendered in the page and read back (the recipe of the earlier 3D work), for each preset in light and dark, viewed and corrected.
- **Unchanged:**
  - `vit/`: the same scenario values and byte-identical screenshots after the move to `ui/vitwords.js`;
  - the CNN 3D explorer and the CNN training page's 3D view: the same scene digest after the `ui/blocks3d.js` additions.

## Left for the second build

The training page's 2D | 3D switch: the same blocks in three rows (forward, backward, update), with the stages revealed, the green frames on the last stage and the faint blocks.
