# Training page: live step band during play, and the depth layout in 3D

Date: 2026-10-13. Status: built in one autonomous session from Alexy's request; the choices below were made without a brainstorm and are open to review. Builds on `2026-10-12-cnn-training-page-design.md` and `2026-10-12-cnn-training-3d-design.md`.

## Request

Alexy, 2026-10-13, after using the deployed 3D view: "It would be good to have the whole... depth. tab done as well and to be able to see the tab representation change when we run the training sequence when we play it. It would be good to see the different activation [plots] for each new pass."

Read as two features: (1) the explorer's "side by side | depth" layout choice in the training page's 3D view; (2) the One-step band, in 2D and 3D, follows the digits while Many steps plays instead of freezing. "Each new pass" is read as each digit's forward pass.

## Live band during play

- **What it shows.** Each digit is caught after its backward pass, before its update: the forward row, the gradient row, and w · ∂L/∂w · w′ with w′ the update about to be applied. This is the phase that shows everything at once.
- **How the loop gets there.** `advance()` replaces `trainOne()`: it finishes the digit on show (forward, backward, update as needed), loads the next and takes it to its backward pass. The training sequence is unchanged; only where a frame stops differs. Pressing Play takes the current digit straight to its backward pass; Pause leaves it there, so Step applies its update.
- **Cadence.** The band redraws at most ten times a second and only when a new digit is on show, so at 200 digits a second it shows about one in 20, and at the new 2 / s speed every digit for half a second.
- **Hover during play.** Kept: the hovered cell or weight stays probed, and its readout and highlight are recomputed for each new digit, so a weight's value and gradient can be watched changing. With nothing hovered, the readout says what the band is showing and at what rate.
- **3D cost.** Block data and scales are functions read when the view draws (`cur`, `shownWeights()`, `gradShown()`), so a new digit in the same phase is a refresh of colours and sizes, not a rebuild. A rebuild happens only when the set of rows or the layout changes; rebuilt meshes are disposed.

## Depth layout in the training 3D view

- **Rows become lanes.** Side by side, a row runs left to right and rows stack downwards. In depth, each row runs away from the viewer along z, as the explorer's depth layout does, and rows sit next to each other as lanes 26 units apart: forward, its gradients, then w, ∂L/∂w, w′. Stacking rows vertically was rejected: from the high camera the depth layout needs, the upper row hides the lower.
- **Alignment.** Each gradient block takes its forward block's place along the flow (`at`), so ∂L/∂conv 1 sits beside conv 1. Weight blocks use a smaller depth gap (3 instead of 7) so their lanes are not much longer than the forward lane; in depth the biases come before the FC weights, otherwise the two labels land on each other.
- **Stage links.** The forward and gradient lanes get the explorer's stage connectors; the weight lanes do not (their blocks are not a flow).
- **Camera.** Presets whole · forward · backward · weights as in side by side. Depth framing projects every block's corners and fits distance and target so the projected scene, plus room for the top labels, fills the view; angles (theta, phi, padding) whole 0.15, 0.9, 1.04; forward 0.55, 0.72, 1.06; backward 0.35, 0.8, 1.05; weights 0.15, 0.85, 1.05, chosen from in-page renders.
- **Switch.** "side by side | depth" at the top right of the view (bottom right below 640 px), remembered in local storage (`dl-explorer.train.layout`).

## Fixes found on the way

- Camera presets frame a complete step's layout (`view.place(defs)` without building), so the frame no longer depends on the phase. Before, switching to 3D before the backward pass fitted the camera to the rows present then, and the gradient rows appeared cut off.
- The "layer-2 kernels" label sits over the middle of the eight kernels instead of the first, where it overlapped "layer-1 kernels".
- `fmt` no longer prints "-0" for tiny negative values (shared by all pages).

## `ui/blocks3d.js` changes

`laneGap` option; block `at` (same flow position as another block, x in side by side, z in depth) and `gapZ` (depth gap after the block); per-row depth offsets; `linkStages` may list the rows to link; `place(defs)`; meshes disposed on rebuild; `colorFor(v, scale, mode, dark, out)` takes the theme and an output colour so a refresh allocates nothing per cube. With one row and none of the new properties, the explorer's scene is unchanged.

## Checks

- Explorer, before and after, six states (default, depth, layer-2 preset, sweep, conv-2 hover, depth with a pool-2 hover): serialised scene and DOM identical.
- Training page, side by side, before and after: every cube's matrix and colour and every overlay identical in six states; the camera differs only where 3D was switched on before the backward pass (now framed on the complete layout) and is identical where it was switched on after.
- Play, driven frame by frame: the band shows phase 2 of a new digit at each redraw, with 27 forward, 26 backward and 42 triplet canvases in 2D; after 476 digits the weights equal those of a plain loop over the same 476 digits with `core/train.js` (hash match); in 3D one rebuild at the start of play and one on a layout switch, otherwise refreshes; mesh count equals block count; at 2 / s every digit is shown.
- Renders: the WebGL frame read back in the same task as the render and composited with the HTML labels, so headless Edge gives real pictures in two seconds; depth presets, a hover in depth, dark theme, a narrow window, and play in both layouts.
- `node core/*.test.mjs` pass.
