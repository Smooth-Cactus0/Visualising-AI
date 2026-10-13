# Training page, 3D representation: design

Date: 2026-10-12. Status: approved in conversation, awaiting Alexy's review of this document. Builds on `2026-10-12-cnn-training-page-design.md`.

## Goal

The training page's One-step band gains a "2D | 3D" switch. In 3D, one WebGL view shows the forward activations as the 3D explorer draws them (blocks of cubes, channels into depth), the gradients as a mirrored row of blocks beneath, and the weights as cubes with their gradient and updated-weight cubes, in the same phases, with the same hover readouts. The play band, controls, curves and hand-off are shared between the two representations. This is Alexy's "switch between representation" on the same thing (Intent).

## Decisions taken in the brainstorm

- The 3D representation lives on the training page behind a toggle, not on a separate page and not inside the 3D explorer. Why: one training page with two representations matches the intent; the explorer is already full.
- The block renderer moves out of the 3D explorer into a shared module, `ui/blocks3d.js`, which both pages import; the explorer's behaviour does not change. Chosen over copying the code (a second copy to maintain) and over embedding the explorer in a frame (a message protocol, and the phases would not map onto it).
- Scene arrangement: rows. Forward blocks on top; each stage's gradient block directly beneath it; the weights beneath that as three rows, w, ∂L/∂w, w′. Only the side-by-side layout in the training view; the depth layout stays on the explorer.
- Hover keeps the probe kinds and readout text of the 2D representation; only the highlights change form (boxes and connectors instead of green rectangles).

## Files

| Path | Change |
|---|---|
| `ui/blocks3d.js` | New. The scene, blocks, highlights, connectors, labels, camera, picking and loop that are now inline in `cnn-3d/index.html`, as a small API (below). Depends on the global `THREE` (r128 UMD from cdnjs, loaded by the page). |
| `cnn-3d/index.html` | Imports the module; keeps its own block definitions, camera preset numbers, sweep, probes, readouts, input canvas and controls. No change in behaviour. |
| `cnn-train/index.html` | The toggle, a `#view3d` container with camera buttons and lateral bar, the 3D step-band renderer, the pick-to-probe mapping, the 3D highlights. The 2D code stays as it is. |
| `CLAUDE.md` | `ui/` line mentions `blocks3d.js`. |

## `ui/blocks3d.js`

```js
export function createBlockView(viewEl, options) → view
```

`options`: `{ explode = 0.7, layout = 'side' | 'depth', gapX = 5, gapZ = 7, rowGap = 22, lateral: HTMLInputElement | null, linkStages = false }`. The module creates the renderer (antialias, pixel ratio ≤ 2), scene, lights, camera (fov 38), the cube geometry and material, and the groups `net` (instanced cube meshes), `hl` and `aux` (overlays the page fills), `stage` (consecutive-stage links in the depth layout when `linkStages` is true). It appends the canvas to `viewEl`, observes its size, and owns the orbit camera (drag to orbit, right-drag or shift-drag to pan, wheel to zoom, horizontal wheel to travel along the flow axis, the lateral bar when given), the picking on pointer move and the render loop with camera easing. Colours: `colorFor(v, scale, mode)` as in the explorer (`gray` for 0–1, `div` for signed), dark theme by the page's `--scene` token and `isDark()` from `ui/grid.js`.

Block definition: `{ id, label, rows, cols, chans, mode: 'gray' | 'div', data: (ch) => arrayLike, scale, row = 0, x0?, hidden = false }`. `label` is two lines joined by a newline (title, shape). `x0` is set by the page or, when absent, laid out cumulatively within its row (`cols + gapX` per block). `row` places the block at `y = -row * rowGap` (side-by-side layout; the depth layout ignores rows and recedes along z as the explorer does). `hidden` blocks are not built.

`view` members:

- `setBlocks(defs)`: builds the instanced meshes (one per block), replacing the previous ones. Returns the block objects (`defs` with `x0`, `z0`, `mesh` added).
- `refresh()`: recomputes positions and colours from `data()` and `scale`, redraws stage links, keeps overlays. Call after data or `explode`/`layout` changes.
- `layout`, `explode`: properties; set then `refresh()`.
- `blockById(id)`, `voxelPos(b, r, c, ch) → Vector3`, `bounds(b, r0, r1, c0, c1, ch0, ch1) → Box3`, `bbox() → Box3` of everything, `centerOf(ids) → Vector3`, `flowRange() → [lo, hi]`, `flowAxis() → 'x' | 'z'`.
- `boxAt(b, r0, r1, c0, c1, ch, group, fillMat, edgeMat)`, `boxSpanZ(b, r0, r1, c0, c1, ch0, ch1, group, fillMat, edgeMat)`, `connect(boxA, boxB, lineMat, group)`: as in the explorer. `mats`: `{ hlFill, hlEdge, rfLine, stageLine, swFill, swEdge, swOut, swLine }` with the explorer's colours.
- `hl`, `aux`: THREE.Groups for overlays; `clear(group)`.
- `onPick(cb)`: `cb({ block, r, c, ch })` when the hovered cube changes, `cb(null)` when the pointer leaves the canvas or hovers nothing. Picking is suppressed while dragging.
- `setGoal({ theta, phi, dist, target: Vector3 })`: camera goal, eased over frames; `cam` (current) readable.
- `labelsFor(blocks)`: the module positions one label element per built block above it, hides it when behind the camera; the page styles `.lbl`.
- `start()`: begins the render loop; `resize()`; `dispose()`.

The explorer keeps: `defineBlocks` (its seven stages with data accessors from its forward state), `camPreset(name)` with its numbers (calling `setGoal`), `setLayout`, the sweep (drawing into `aux`), `footprint` and `updateHighlights` (drawing into `hl`), the readouts, `pick` semantics (mapping a picked cube to its probe), the 2D input canvas and all controls.

## The training page in 3D

- **Toggle.** A `.seg` pair "2D | 3D" beside the phase strip. 3D hides `#fwd`, `#bwd`, `#trips` and shows `#view3d` (full width, 560 px tall, with the camera buttons whole · forward · backward · weights, recentre, and the lateral bar). The readout stays under the view. The choice is stored in local storage (`dl-explorer.train.view`) and restored on load. Switching representation re-renders the current state; nothing in training state changes.
- **Blocks and rows** (side-by-side layout, `rowGap` 22). Row 0, forward: `in` (16×16×1, grey), `conv1` (16×16×4), `pool1` (8×8×4), `conv2` (8×8×8), `pool2` (4×4×8), `gap` (1×1×8), `scores` (10×1×1), labelled as on the explorer. Row 1, backward, same `x0` as the forward block above: `d_conv1`, `d_pool1`, `d_conv2`, `d_pool2`, `d_gap`, `d_scores`, labelled `∂L/∂conv 1` and so on; no block under the digit. Rows 2, 3, 4: the weights as w, ∂L/∂w, w′, each row holding `bank` (3×3×4, the four kernels as channels), `l2_1 … l2_8` (3×3×4 each, the four slices as channels), `W` (8 rows × 10 cols × 1), `B` (1×10×1), laid out cumulatively; the three rows share `x0`s. Scale per row group as in 2D (weights by the layer's largest |w|, gradients by the layer's largest |∂L/∂w|, w′ on the weight scale).
- **Phases.** Before forward: only `in` and the w row. After forward: row 0 and the w row. After backward: everything. After update: rows 3 and 4 hidden (the backward row stays, as in 2D); the w row fades from old to new colours over 300 ms unless reduced motion. Hidden blocks are rebuilt out, not left in place.
- **Pick to probe.** `in` → `{ kind: 'pixel', r, c }` (readout: the pixel value); `conv1 | pool1 | conv2 | pool2` → `{ kind: 'map', stage, r, c, ch }`; `gap` → `{ kind: 'gap', f: ch }`; `scores` → `{ kind: 'bar', c: r }`; `d_<stage>` → `{ kind: 'back', stage, r, c, ch }`; `d_gap` → gap; `d_scores` → bar; `bank_<row>` → `{ kind: 'w', layer: 'bank', k: ch, q: r * 3 + c }`; `l2_k_<row>` → `{ kind: 'w', layer: 'l2', k, s: ch, q: r * 3 + c }`; `W_<row>` → `{ kind: 'w', layer: 'W', f: r, c }`; `B_<row>` → `{ kind: 'w', layer: 'B', c }`. The readout function of the 2D representation is reused unchanged (it reads `probe`), with the `pixel` kind added.
- **Highlights.** Map probe: `footprint` rects drawn as boxes on the forward blocks (every channel when `chs` is null, as the explorer does) and a connector cone back to the digit. Back probe: a box on that cell. Gap and bar probes: the explorer's direct links (an average to its pool-2 map, a class to the averages), plus the FC column or row box on the `W` cubes. Weight probe: a box on the hovered cube, full-block boxes on its two sources (the gradient block in row 1 and the activation block in row 0; for `W` and `B`, the `gap` block and the `scores` cube) and connectors from the sources to the cube.
- **Camera presets.** whole: everything in frame from the front-right-above; forward: row 0; backward: rows 0 and 1; weights: rows 2 to 4. Recentre returns to the current preset. Numbers chosen from the projected label positions in headless runs (theta, phi, padding: whole 0.4, 1.25, 1.15; forward 0.45, 1.15, 1.35; backward 0.45, 1.2, 1.35; weights 0.3, 1.2, 1.5); blocks 7 units apart.
- **Play.** The 3D view holds still during play like the 2D band; pause re-renders it.
- **Theme and motion.** Scene background from the page's `--scene` token (added to the training page's tokens, light and dark). Reduced motion disables the fade and the camera easing.

## Checks

- Explorer before and after the extraction: from the same seeded weights, in five states (default, depth layout, layer-2 preset, layer-1 sweep at step 5, a conv-2 hover), a digest of the whole scene (every instance matrix and colour, every overlay's geometry and position, the camera) and the DOM (scripts stripped) are identical. Done this way, not with pixel screenshots, because software WebGL in headless Edge took minutes per screenshot and the eased camera made frame counts matter; the default-state screenshots that did complete differ by one pixel.
- Training page: the 2D representation's DOM check from the previous spec still passes unchanged. In 3D: scripted states (toggle to 3D; step ×2 with an FC-weight probe set; step ×3) dumped and screenshotted at 1400 px: the expected labels present (`conv 1 + ReLU`, `∂L/∂conv 1`, `w`, `∂L/∂w`, `w′` rows), the readout text as in 2D, no console errors; a dark-theme screenshot.
- `node core/*.test.mjs` unchanged and passing.

## Left for later, on purpose

The depth layout in the training view (built 2026-10-13, see `2026-10-13-cnn-training-live-depth-design.md`); animating the kernel sweep during a step; a 3D view of the play band's kernel history.
