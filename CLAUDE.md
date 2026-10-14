# DL architecture explorers — project conventions

Purpose: interactive, visual explanations of how DL architectures compute (CNN built; then ViT, U-Net, CoAtNet, ConvNeXt, in that order). The audience is someone who uses these models daily and wants the concrete steps behind diagrams like "Conv 3×3 → ReLU → MaxPool → … → FC".

## Layout
- `core/` — pure JS model code and shared presets (ES modules, no DOM). Tests: `node core/cnn.test.mjs`.
- `ui/` — shared DOM helpers: `terms.js` (the `TERMS` array, tooltip, glossary) and `terms.css`; `grid.js` (canvas grids, hit-testing, colours); `trained.js` (trained-weights hand-off through local storage); `blocks3d.js` (the 3D block renderer: scene, instanced cubes, highlights, labels, orbit camera, picking). ES modules, imported by the pages.
- `data/` — the 16×16 MNIST subset (`mnist16.bin`, format and licence in its README); `tools/` — the standard-library Python script that makes it; `docs/superpowers/` — design specs and implementation plans.
- `index.html` — the landing page: the colour grammar, then one section per architecture (schematic, a plain definition, its pages as cards, or "what the page will show" while planned). When an architecture gets pages, turn its section's list into cards and its badge into the page count.
- `cnn-2d/`, `cnn-3d/`, `cnn-train/` — one self-contained page each, with a breadcrumb back to the landing page. New pages follow the same pattern (`vit/`, `unet/`, `coatnet/`, `convnext/`).
- `models/` — trained weights (JSON/ONNX) when added. Empty for now.
- No build step. Plain HTML + ES modules served statically (GitHub Pages workflow in `.github/workflows/pages.yml`). Local dev: `python3 -m http.server 8000` from the repo root; modules need http, not file://.
- External scripts only from cdnjs or jsDelivr, pinned to exact versions. three.js is r128 UMD from cdnjs; OrbitControls is not in that build, so `ui/blocks3d.js` has its own small orbit implementation, used by the 3D explorer and the training page's 3D view.

## Design rules (keep these)
- Visual and interactive over tables and text. The owner explicitly removed a table; don't reintroduce one.
- Linked highlighting across views is the core interaction. Yellow = forward kernel window; green = what a value is computed from: an activation's receptive field (upstream), a gradient's sources (downstream, up to the loss), a weight gradient's two maps, and on the training page the step of the backward pass just taken. Keep the two colours distinct and consistent across every page.
- Weights sit where they act: between the map they read and the map they make, marked as learned (tinted column in 2D, accent frame in 3D). Where training is shown, three rows: forward (activations and w), backward (each gradient under what it is the gradient of), update (w′ under each weight). This is the layout to reuse for the next architectures.
- Hovering a cell anywhere traces its receptive field through every upstream tensor. At the head (global average, classes) show only the direct link, not the full cone back to the input.
- Every technical term gets a dotted inline definition (tooltip) and an entry in a Terms section. The mechanism is shared: `ui/terms.js` (`TERMS` array, `installTerms()`) with `ui/terms.css`; every page imports it rather than carrying a copy.
- Weights: layer-1 kernels are hand-picked edge detectors; layer-2 and FC are random. Pages must say so wherever a prediction appears. The training page (`cnn-train/`) trains all three layers live; its weights can be loaded into the explorer pages, which then say so in a banner and in their wording.
- Theme-aware (light/dark via CSS tokens), responsive, reduced-motion respected. Canvas/WebGL elements must be sized with CSS as well as pixels (a 2× DPR overflow bug has already bitten once).

## Owner preferences
- Concise, reasoning-led replies. Compare independently reasoned answers and discuss discrepancies rather than just agreeing.
- NZ English spelling.
