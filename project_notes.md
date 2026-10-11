# DL explorer (Visualise models): project notes
Last updated: 2026-10-12, by a working session: Intent recorded in Alexy's words; work continues from the repo.

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
- Toy network: 16×16×1 input → conv1 (4 hand-picked 3×3 kernels: Sobel-x, Sobel-y, Laplacian, diagonal; stride, padding and dilation selectable) + ReLU → 2×2 pool (max or avg) → conv2 (8 random 3×3×4 kernels) + ReLU → 2×2 pool → global average (8 values) → FC 8×10 (random) → softmax over 10 classes. Nothing is trained; the pages say so wherever a prediction appears.
- `core/cnn.js` holds the forward pass, pooling, softmax, receptive-field propagation and the kernel and input presets, as pure functions with a test. The two pages do not import it yet: each still carries its own inline copy of the same logic.
- Interaction grammar shared by every page: yellow = forward kernel window, green = receptive field traced backward; hovering any cell highlights its receptive field in every upstream tensor; technical terms get a dotted inline definition plus a Terms section (2D page only so far).
- three.js r128 UMD from cdnjs; OrbitControls is not in that build, so the 3D page has its own small orbit implementation. External scripts only from cdnjs or jsDelivr, pinned.

### Results
- 2026-10-12: `node core/cnn.test.mjs` passes (laptop, Node 24.19.0, repo at commit 6a95715): output-size formula, identity kernel, max pool, softmax sums to 1, shapes 16 → 8 → 4, receptive field of pool2 cell (1,1) = input rows 1–10, cols 1–10.
- No accuracy figures exist: weights are hand-picked or random, no model has been trained.

### Decisions
- 2026-10-12: The repo lives at `C:\Users\LouisA\repos\dl-explorer`, not in the OneDrive project folder. Why: Windows Controlled Folder Access (IT policy, enforced since 2026-09-23) blocks git, node, python and every other shell tool from writing under OneDrive Documents; only Claude's Write/Edit tools can write there. The OneDrive folder keeps the original zip and a pointer README.
- 2026-10-12: First commit imports the artifact export unchanged, so the pages on `main` are exactly the published artifacts. Why: a clean baseline before the shared-module refactor.
- 2026-10-12: VS Code and Claude Code are opened on the repo, not on the OneDrive folder, from here on. Why: Alexy, after the port ("I'll continue on the proper folder afterwards"); notes and Claude memory live with the repo.
- Carried over from the artifact sessions (HANDOFF.md, "Decisions already taken"): the 3D second view is stages along depth with connections, not volumetric 3D convolution; head wiring links only direct neighbours (an average to its own pool-2 map, a class to the 8 averages); no shape/parameter table, the proportional ledger replaced it; padding is not drawn as cubes in 3D, the layer-1 window overhangs the input instead; layer-1 kernels are Sobel-x, Sobel-y, Laplacian and diagonal, layer 2 and the FC are random with reroll buttons.

### Map
- Repo (git, branch `main`, no remote yet): `C:\Users\LouisA\repos\dl-explorer`
- OneDrive project folder (original zip and a pointer README, no code): `C:\Users\LouisA\OneDrive - MWLR\Documents\Side projects\Visualise models`
- 2D page: `cnn-2d/index.html`; 3D page: `cnn-3d/index.html`; landing page: `index.html`
- Shared model core and its test: `core/cnn.js`, `core/cnn.test.mjs`
- Handoff from the artifact sessions (what each page does, decisions, rough edges, ideas, next architectures): `HANDOFF.md`
- Conventions Claude Code reads (layout, colour grammar, no-tables rule, inline-terms rule): `CLAUDE.md`
- Pages deployment: `.github/workflows/pages.yml`
- Trained weights (planned, empty): `models/`

### Status
Imported and committed on 2026-10-12 (commit 6a95715), not pushed, no GitHub remote yet. Next, in HANDOFF.md order: (1) wire both pages to `core/cnn.js` with no change in behaviour, checked in the browser and with the node test; (2) rough edges: in the 3D depth layout the input occludes deeper stages from the default angle, the 3D page lacks the Terms tooltips, 3D pooling is max only; (3) ideas: trained LeNet-style weights exported to `models/`, 28×28 input, shared input state between the pages; (4) ViT, ConvNeXt and CoAtNet pages reusing the same interaction grammar.

## Open questions and known gaps
- Intent does not say who the tool is for; the first court can capture it.
- No GitHub repo yet; the name, visibility and the Pages source ("GitHub Actions") are Alexy's call before the first push.
