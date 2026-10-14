# Training page: weights where they act, backward a layer at a time

Date: 2026-10-14. Status: first iteration, built in one session from Alexy's notes; expected to need a few rounds. Supersedes the step-band layout of `2026-10-12-cnn-training-page-design.md` (separate forward, backward and triplet rows) and the 3D rows of `2026-10-12-cnn-training-3d-design.md`.

## Request

Alexy, 2026-10-14, on the training page:

- "On the 2D part, we need to rework the plot when going step-by-step. We will need a horizontal scroll I believe to have the whole architecture on the same line. And then it should be a bit clearer where each weights are located."
- "It should also be clearer, by doing the green contouring, which weights is affected by the loss, and how this is propagating backwards. That might require a few iterations to get right but is important, as this is the foundation for all the future models."
- 3D, side by side: "not very clear which kernel is layer 1 and which are layer 2 ... not clear at a glance which is applied where ... it quickly gets messy with backward pass." Depth: not clear which derivative values belong to the main architecture and which to the kernels.
- "A restart button to just start anew and clear the plots."

## Principles (shared by 2D and 3D, and meant for the next architectures)

1. **Weights sit where they act.** The layout follows the data, left to right, and each layer's weights stand between the map they read and the map they make: layer-1 kernels between the digit and conv 1, layer-2 kernels between pool 1 and conv 2, FC weights and biases between the averages and the scores. Learned weights are marked: a tinted column in 2D, an accent-coloured frame in 3D.
2. **Three rows, one per question.** Forward: activations and the weights w. Backward: the gradient of each thing, directly under it. Update: w′ = w − lr·∂L/∂w, under each weight. The old triplet w · ∂L/∂w · w′ becomes this vertical column.
3. **Channel k is a line.** In 2D, kernel k, the maps it makes and, for layer 2, its average and its row of FC weights sit on one horizontal line.
4. **Backward a layer at a time.** Step goes digit → forward → back through FC → back through layer 2 → back through layer 1 → update. The gradient is computed in one go (`core/train.js`); the band reveals it from the loss towards the input. Each step frames in green what it computed and lights its backward arrows; what is not computed yet is drawn faint, in place, so nothing moves between steps.
5. **Green means "computed from".** Hovering an activation shows its receptive field upstream and the kernel that made its map. Hovering a gradient shows, downstream, the later gradients it was computed from up to the loss, the mirror of the receptive field (`gradFootprint` in `core/train.js`), and the weights the error came back through. Hovering a weight lights it in all three rows and outlines the two maps whose product is its gradient: what came forward into it, and what came back out of it.

## 2D

One line inside a horizontally scrolling frame, the band label column ("forward", "backward", "update") sticky at the left. Between columns, the forward operation (⋆, =, →, ·) and, in the backward row, an arrow ← that turns green on the step that sends the gradient through it. Sizes: digit 7 px cells; channel stacks 288 px tall (four 72 px slots for layer 1, eight 36 px slots for layer 2); layer-2 kernels as four 3×3 slices side by side; FC weights as eight rows of ten, with the biases as a ninth row; the softmax bars in rows of 24 px with p − y beneath them. Fits on one line from a window of about 1600 px; scrolls below that.

## 3D

The same three rows. Forward row: digit, layer-1 kernels (3×3×4, kernels into depth like conv 1's channels), conv 1, pool 1, the eight layer-2 kernels in a 2 × 4 grid (slices into depth like pool 1's channels), conv 2, pool 2, average, FC weights with the biases beneath, softmax. Gradients beneath, w′ beneath the weights. Accent frames around the weight groups in every row; green frames on what the last backward step computed. Camera presets: whole, forward, backward, update, framed for a complete step by projecting every block's corners (one rule for both layouts). Depth: rows become lanes; the scores stand beside the FC weights; a longer gap after the averages; overlapping labels are hidden, nearest first (`declutter` option of `ui/blocks3d.js`; the explorer does not use it).

`ui/blocks3d.js` gained `dx`, `dy` (offsets within a row), `ghost` blocks (hold a place in the layout before they exist, so blocks do not jump when the forward pass appears) and `declutter`. With none of them used, the explorer's scene is unchanged (digest identical in six states).

## Play and restart

Play takes each digit to the end of its backward pass, as before. "Restart" (new seed, everything cleared) sits next to Play; "Restart from edge detectors" replaces "Start layer 1 from edge detectors".

## Checks

- `node core/train.test.mjs`: `gradFootprint` wiring, and ∂L/∂pool1 at four cells recomputed from the conv-2 gradients inside its footprint only.
- 2D at every phase: 210 grids at all phases (nothing moves), green frames on 4, 5 and 3 cells for the three backward steps, faint cells 18 → 12 → 8 → 3 → 0; screenshots of the phases and of gradient, weight and activation hovers, light and dark.
- Play, frame by frame: after 396 digits the weights equal a plain `core/train.js` loop's (hash match), in 2D and 3D; the 3D scene rebuilds once when play starts; batch 8 collects and applies as before.
- Click-only run with a 60 fps loop: play trains and redraws, depth switches mid-play, Restart empties the history.

## Open for the next iterations

The green front marks cells; it does not yet animate the flow. Layer-2 kernels are drawn small (9 px cells). The depth view hides some labels at the far end rather than moving them.
