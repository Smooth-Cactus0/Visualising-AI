# Handoff — state as of 2026-10-12

This repo began as two published artifacts in a claude.ai chat, exported here unchanged. Both pages work as single files. The shared-module split has been started but not wired in.

## What exists

**`cnn-2d/index.html`** — six stages on one page:
1. Drawable 16×16 input (presets: 7, 0, cross, diagonal, square, noise) and an editable 3×3 kernel with presets; stride / padding / dilation selects with the output-size formula evaluated live.
2. Convolution one position at a time: scrub/play through output cells; window ⊙ kernel = products, with the written-out sum.
3. ReLU (toggle) and 2×2 pooling (max/avg), three maps side by side.
4. Bank of 4 layer-1 kernels → 4 maps; one 3×3×4 layer-2 kernel shown as 4 slices → 4 partial sums → 1 map.
5. Full pipeline to a prediction: pool1 → conv2+ReLU (8 maps) → pool2 → global-average vector → 8×10 FC heatmap → softmax bars; parameter ledger (proportional bars) above it; hover a class to see its dot product written out.
6. Terms glossary; every first mention of a term in a section is a dotted tooltip (`TERMS`, `linkTerms()`).
Hover on any map cell anywhere traces its receptive field in green across every upstream map and the input, with a readout ("sees rows a–b, cols c–d: an n×m patch").

**`cnn-3d/index.html`** — the same network drawn with three.js (r128 UMD) as H×W×C blocks: InstancedMesh cubes, size = magnitude, colour = sign (blue −, orange +, grey input). Two layouts via toggle: *side by side* (stages along x, channels into z) and *depth* (stages receding along z, faint corner-to-corner links between consecutive stages). Channel-spacing slider (0 = solid block). Kernel sweep for layer 1 (3×3×1 window over the input) and layer 2 (3×3×4 window spanning all four pool-1 channels; per-channel partial sums written out), with yellow connector lines from window to output cube. Hover → receptive-field boxes on every upstream block plus a green connector cone; head probes show only the direct link (average ↔ its pool-2 map; class ↔ the 8 averages). Camera presets, recenter, lateral travel bar (synced with manual pan and horizontal wheel), custom orbit (drag / wheel / right-drag pan).

**`core/cnn.js`** — forward pass, pooling, softmax, receptive-field propagation, kernel and input presets, extracted from the pages. `core/cnn.test.mjs` passes. **Both pages import it** (ES modules, done 2026-10-12); each keeps only display code and thin wrappers (the padded-input canvas in 2D, the head probes in both).

## Decisions already taken (don't relitigate without reason)
- The 3D "second view" is stages-along-depth with connections, not volumetric 3D convolution.
- Head wiring: global average links only to its own pool-2 map; a class links only to the 8 averages. The full cone back to the input is omitted there because it would be the whole image.
- No shape/parameter table; the ledger replaced it.
- Padding is not drawn as cubes in 3D; the layer-1 window overhangs the input edge instead.
- Layer-1 kernels: Sobel-x, Sobel-y, Laplacian, diagonal. Layer 2 and FC random with reroll buttons.

## Next steps, in a sensible order
1. **Wire both pages to `core/cnn.js`.** _Done 2026-10-12, no change in behaviour (see project_notes.md, Results)._ Convert each inline `<script>` to `<script type="module">`, import `forward`, `footprint`, `kernelPresets`, `defaultBank`, `inputPresets`, delete the duplicated inline logic. Behaviour must not change; use the pages' interactions as the manual test and `node core/cnn.test.mjs` for the maths. Note the 2D page's `footprint` also returns `padIn` (for the padded-input canvas) and the 3D page's probe has extra kinds (`gap`, `scores`, `input`) handled before propagation — keep those wrappers in the pages.
2. **Known rough edges.** _Done 2026-10-12: higher, nearly head-on default camera for the depth layout; Terms tooltips and a Terms section on the 3D page through the shared `ui/terms.js`; a max/average pooling toggle in 3D; the 2D padding dropdown now shows its real default (1)._ In 3D depth layout the 16×16 input partly occludes deeper stages from the default angle (higher default camera, or a view from the output end looking back). The 3D page lacks the Terms tooltips. `poolMode` in 3D is max only.
3. **Ideas discussed, not built.** _Trained weights and shared state arrived 2026-10-12 in a different form: the training page `cnn-train/` trains the toy live in the browser on 16×16 MNIST (no PyTorch), and hands its weights to the 2D and 3D pages through local storage; 28×28 and the travelling layer-2 kernel block remain ideas._ Render the layer-2 kernel's weights as a small coloured 3×3×4 block travelling with the sweep window. Bump input to 28×28 so receptive-field growth reads better. Trained LeNet-style weights: train offline (PyTorch, MNIST), export to JSON under `models/`, load in-browser so maps and predictions become real; the forward pass in `core/cnn.js` is already the right shape. Shared input state between the 2D and 3D pages (URL hash or localStorage).
4. **Next architectures**, reusing the same interaction grammar (yellow forward window, green backward footprint, hover-linked highlights, inline terms):
   - ViT: patch embedding as a strided conv, tokens, one attention head shown as Q·Kᵀ → softmax → weighted sum, the attention map as "a receptive field that is global in one step".
   - ConvNeXt: 7×7 depthwise conv (one slice per channel, no cross-channel sum — contrast with the 3×3×C kernel here), pointwise 1×1 as the mixing step, inverted bottleneck, LayerNorm/GELU.
   - CoAtNet: the diagram that started this project — MBConv stages then relative-attention stages; show where the switch from local to global happens and what it costs.

## Run locally
    python3 -m http.server 8000     # http://localhost:8000/
    node core/cnn.test.mjs

## Publish
Push to `main` on GitHub with Pages set to "GitHub Actions" as the source; `.github/workflows/pages.yml` deploys the repo root.
