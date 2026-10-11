# DL explorer (Visualise models): project notes
Last updated: 2026-10-12, by a working session: HANDOFF steps 1 (pages wired to `core/cnn.js`) and 2 (rough edges: 3D depth camera, terms on the 3D page, 3D pooling toggle, 2D padding default) done; published to GitHub Pages.

## Intent
_Alexy's words. Changed only when Alexy asks, directly or by pointing to a court point._
- **Purpose:** "visualize DL models, and later on maybe ML models, in an intuitive and interactive way"
- **What a good result looks like:** "A good result is based on the artifacts developed on the online claude session. Building on this, a good result is a bigger tool which allows us to switch between models and representation (side-by-side or depth, 2D or 3D) and develop a good intuition on what different model architectures are, their strength and limitations."
- **Constraints and must-nots:** "should look the same as the artifacts created in the online claude session"
- **Who it is for:** Not stated yet.

## As built
_Kept up to date by Claude._
### Overview
Interactive web pages, with no build step, that show what the boxes in a deep-learning architecture diagram actually compute. Two pages exist, both for the same toy CNN: a 2D page (convolution one output cell at a time, ReLU, pooling, channels, receptive field, through to a prediction) and a 3D page (the same network drawn as H×W×C blocks with three.js, with kernel sweeps and receptive-field cones). The repo's CLAUDE.md says: "Purpose: interactive, visual explanations of how DL architectures compute (CNN first; ViT, ConvNeXt, CoAtNet next). The audience is someone who uses these models daily and wants the concrete steps behind diagrams like 'Conv 3×3 → ReLU → MaxPool → … → FC'." Alexy, 2026-10-12: the project is to "visualise different models, starting with DL", ported locally "to add to it without being limited by artifacts size, versioning and such."

### Method
- Plain HTML plus ES modules, served statically. GitHub Pages deploys the repo root from `main` (`.github/workflows/pages.yml`). Local run: `python -m http.server 8000` from the repo root (modules need http, not file://); maths test: `node core/cnn.test.mjs`.
- Toy network: 16×16×1 input → conv1 (4 hand-picked 3×3 kernels: Sobel-x, Sobel-y, Laplacian, diagonal; stride, padding and dilation selectable) + ReLU → 2×2 pool (max or average, a toggle on both pages) → conv2 (8 random 3×3×4 kernels) + ReLU → 2×2 pool → global average (8 values) → FC 8×10 (random) → softmax over 10 classes. The 2D page can also switch the activation off. Nothing is trained; the pages say so wherever a prediction appears.
- `core/cnn.js` holds the forward pass, pooling, softmax, receptive-field propagation and the kernel and input presets, as pure functions with a test. Both pages are ES modules (`<script type="module">`) that import it; each keeps only display code plus two thin wrappers: the 2D page builds the zero-padded input it draws in stage 2, and both pages resolve the head probes (a global-average entry, a class, a score, an input pixel) before calling the core `footprint`. `forward` takes `activation: 'relu' | 'none'` (the 2D ReLU toggle) and `poolMode: 'max' | 'avg'`; `footprint` takes one cell `{stage, r, c, ch}` or a region `{stage, rect, ch | chs}`.
- Interaction grammar shared by every page: yellow = forward kernel window, green = receptive field traced backward; hovering any cell highlights its receptive field in every upstream tensor; technical terms get a dotted inline definition plus a Terms section on both pages. The 3D readout for a pooled cube writes out the 2×2 block it summarises, as the 2D page does.
- `ui/terms.js` and `ui/terms.css` hold the terms mechanism (the `TERMS` array, the dotted definitions with their tooltip, the Terms section); both pages import `installTerms` with their own text selectors. Terms are linked at their first mention per paragraph, in order of appearance; dynamic text (readouts, the sweep hint) is left unlinked.
- 3D depth layout: the default "whole network" camera looks from high up and nearly head-on from the input end, so the stages recede as a staircase and the input hides nothing; chosen from headless-Edge screenshots of nine candidate presets at two window widths.
- three.js r128 UMD from cdnjs; OrbitControls is not in that build, so the 3D page has its own small orbit implementation. External scripts only from cdnjs or jsDelivr, pinned.

### Results
- 2026-10-12: `node core/cnn.test.mjs` passes (laptop, Node 24.19.0, repo at commit 6a95715): output-size formula, identity kernel, max pool, softmax sums to 1, shapes 16 → 8 → 4, receptive field of pool2 cell (1,1) = input rows 1–10, cols 1–10.
- 2026-10-12: wiring check (laptop, Node 24.19.0 and headless Edge, commit a09b700). `node core/cnn.test.mjs` passes, now also covering `activation: 'none'`, average pooling and region probes. The pages' old inline maths (from commit 6a95715) and `core/cnn.js` agree bit for bit on every tensor of the forward pass over all reachable settings (2D: stride 1–3, padding 0–2, dilation 1–2, ReLU on/off, max/avg pooling; 3D: stride 1–2, padding 0–1, dilation 1–2), all seven input presets, fixed random layer-2 and FC weights, and on the receptive field of every cell of every stage and channel plus the head probes: 560 forward configurations, 21,384 probes, 27,572 comparisons, 0 differences (one-off script in the session scratchpad, not in the repo). Headless Edge rendered the old and new pages with the same seeded `Math.random`: identical DOMs once script blocks are stripped, no page console errors.
- 2026-10-12: step-2 check (laptop, headless Edge with SwiftShader WebGL, Node 24.19.0, commit after a09b700). In depth layout at 1400 px and 1000 px window widths the 3D page shows all seven stages, input to softmax, with nothing hidden or under the readout. With average pooling selected and a pool-1 cube probed, the readout writes out the 2×2 block (`avg( ... ) = ...`). Dotted terms: 30 on the 2D page (19 before the ordering fix in `linkTerms`), 4 on the 3D page (channels, receptive field, kernel, FC: its static text mentions few terms); glossary 16 entries on both. The 2D padding dropdown renders with 1 selected. No page console errors. `node core/cnn.test.mjs` passes.
- No accuracy figures exist: weights are hand-picked or random, no model has been trained.

### Decisions
- 2026-10-12: The repo lives at `C:\Users\LouisA\repos\dl-explorer`, not in the OneDrive project folder. Why: Windows Controlled Folder Access (IT policy, enforced since 2026-09-23) blocks git, node, python and every other shell tool from writing under OneDrive Documents; only Claude's Write/Edit tools can write there. The OneDrive folder keeps the original zip and a pointer README.
- 2026-10-12: First commit imports the artifact export unchanged, so the pages on `main` are exactly the published artifacts. Why: a clean baseline before the shared-module refactor.
- 2026-10-12: VS Code and Claude Code are opened on the repo, not on the OneDrive folder, from here on. Why: Alexy, after the port ("I'll continue on the proper folder afterwards"); notes and Claude memory live with the repo.
- 2026-10-12: The pages import `core/cnn.js` as ES modules instead of carrying inline copies of the maths. The 2D ReLU toggle and the head probes went into the core (an `activation` option, region probes for `footprint`) rather than staying as page-side copies of the propagation loop. Why: one copy of the maths, and the page wrappers stay a few lines each.
- 2026-10-12: The terms mechanism moved out of the 2D page into a new `ui/` folder (shared DOM code; `core/` stays DOM-free) instead of being copied into the 3D page. Why: the CLAUDE.md rule says reuse it, and step 1 had just removed the other duplicate.
- 2026-10-12: `linkTerms` links each term at its first occurrence, in order of appearance, instead of in `TERMS` order within the text left after the previous match. Why: the old order skipped terms ("receptive field" in the 3D lede came before "kernel" and was never linked); the 2D page gained 11 dotted terms from the change.
- 2026-10-12: The depth-layout default camera looks from the input end, high and nearly head-on, not from the output end looking back (the handoff offered both). Why: from the output end, perspective shrinks the far input and reverses the funnel the lede describes; from high up the funnel reads as a staircase and nothing occludes.
- 2026-10-12: The 2D padding dropdown marks 1 as selected. Why: it showed 0 while the model computed with 1.
- 2026-10-12: Public GitHub repo `Smooth-Cactus0/Visualising-AI`, deployed with GitHub Pages from `main`. The name is Alexy's "Visualising AI" with GitHub's own space-to-hyphen conversion. Why: Pages is free for public repos; a private repo would need a paid plan.
- Carried over from the artifact sessions (HANDOFF.md, "Decisions already taken"): the 3D second view is stages along depth with connections, not volumetric 3D convolution; head wiring links only direct neighbours (an average to its own pool-2 map, a class to the 8 averages); no shape/parameter table, the proportional ledger replaced it; padding is not drawn as cubes in 3D, the layer-1 window overhangs the input instead; layer-1 kernels are Sobel-x, Sobel-y, Laplacian and diagonal, layer 2 and the FC are random with reroll buttons.

### Map
- Repo (git, branch `main`): `C:\Users\LouisA\repos\dl-explorer`; remote `origin` = GitHub `Smooth-Cactus0/Visualising-AI` (public)
- Published site (GitHub Pages, deployed from `main` by the workflow on every push): https://smooth-cactus0.github.io/Visualising-AI/
- OneDrive project folder (original zip and a pointer README, no code): `C:\Users\LouisA\OneDrive - MWLR\Documents\Side projects\Visualise models`
- 2D page: `cnn-2d/index.html`; 3D page: `cnn-3d/index.html`; landing page: `index.html`
- Shared model core and its test: `core/cnn.js`, `core/cnn.test.mjs`
- Shared terms mechanism (dotted definitions, tooltip, glossary): `ui/terms.js`, `ui/terms.css`
- Handoff from the artifact sessions (what each page does, decisions, rough edges, ideas, next architectures): `HANDOFF.md`
- Conventions Claude Code reads (layout, colour grammar, no-tables rule, inline-terms rule): `CLAUDE.md`
- Pages deployment: `.github/workflows/pages.yml`
- Trained weights (planned, empty): `models/`

### Status
Imported 2026-10-12 (commit 6a95715); both pages wired to `core/cnn.js` the same day (commit a09b700, HANDOFF step 1), then the step-2 rough edges fixed (3D depth camera, terms on the 3D page, 3D pooling toggle, 2D padding default). Pushed to GitHub (`Smooth-Cactus0/Visualising-AI`, public) and deployed with GitHub Pages on 2026-10-12. Next, in HANDOFF.md order: (3) ideas: trained LeNet-style weights exported to `models/`, 28×28 input, shared input state between the pages; (4) ViT, ConvNeXt and CoAtNet pages reusing the same interaction grammar.

## Open questions and known gaps
- Intent does not say who the tool is for; the first court can capture it.
