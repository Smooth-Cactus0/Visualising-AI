# Vision Transformer training page: design

Date: 2026-10-28. Status: Part 1 approved in conversation, Parts 2 to 4 approved in principle ("yes to all parts"); awaiting Alexy's review of this document.

## Goal

A page, `vit-train/`, that shows the toy ViT of `vit/` learning, as `cnn-train/` does for the CNN:
- one training step in full, with the backward pass through attention revealed a stage at a time;
- many steps, showing what changes over training;
- the ViT against the CNN on the same digits, which brings out the one limitation the explorer cannot show: a ViT needs more data than a CNN.

Intent it serves (`project_notes.md`): "develop a good intuition on what different model architectures are, their strength and limitations".

## What was measured before designing

These measurements were made on the laptop on 2026-10-28, with scratch scripts that are not in the repo.

**The position table does not learn a 2D grid at this scale.**
- Measure: the correlation between the grid distance of two patch slots and the cosine similarity of their position rows (a 2D grid would make it clearly negative).
- Results: +0.03 at the start (seed 1); +0.19 for the shipped weights (SGD, 10 passes), where neighbours are less alike than far slots (−0.06 against +0.08); +0.06 for Adam weights.
- Yet the model relies on position: its held-out accuracy falls from 82.0% to 33.5% with position off. It uses position as an identity per slot, not as a map.

**Data hunger shows as a gap at small data, with no crossover on average.**
- Setup: held-out accuracy on the 1,000 test digits after 30,000 SGD steps on the first N training digits (N/10 per class). The CNN starts from random He-uniform weights at lr 0.1; the ViT from `init(seed)` at lr 0.003. Seeds 1, 2 and 3.
- 100 digits: CNN 72.5, 74.7, 71.9 (mean 73.0); ViT 47.6, 45.3, 50.4 (mean 47.8).
- 300 digits: CNN 82.7, 66.4, 72.9 (mean 74.0); ViT 62.9, 64.1, 59.6 (mean 62.2).
- 1,000 digits: CNN 78.6, 73.8, 76.7 (mean 76.4); ViT 75.2, 76.1, 72.7 (mean 74.7).
- 3,000 digits: CNN 78.9, 79.1, 72.4 (mean 76.8); ViT 82.0, 72.3, 70.1 (mean 74.8).

So with 100 digits the CNN leads by about 25 points in every seed, and with 300 by about 12. From 1,000 digits up the two are level. Single runs vary widely: the CNN spread 16 points across seeds at 300 digits, the ViT 12 points at 3,000. The shipped explorer weights (seed 1, 82.0%) are a lucky run.

## Decisions taken in the brainstorm

- **The page has three bands: One step, Many steps, CNN against ViT.** Built in that order. Why: Alexy chose A. The one-step band applies the site's grammar to the newest mechanics (backward through attention). The race is the strengths-and-limitations lesson that no page shows yet. The data-hunger result could not be shown on the explorer.
- **The race is a measured chart plus a live race.** The chart: held-out accuracy against the number of training digits, one dot per seed for each model, and a line through the means, computed offline over several seeds. The live race: both models train side by side on the same digits, with a size switch defaulting to 100 digits, and its result lands on the chart as one more dot. Why: Alexy chose A. A single live run can reverse the lesson by chance (see the spreads above); the chart carries the robust answer and the race the feel.
- **"Many steps" shows the attention maps of one fixed digit over training, and the position table.** The position table is shown honestly as unstructured. Why: Alexy chose A. Attention is what a ViT learns that a CNN does not. The position table's lack of a grid is a real finding at this scale, and the page says so rather than promising the ViT paper's picture.
- **2D only in this iteration.** A 3D view of the ViT comes later, for the explorer and this page together.
- **Training uses SGD.** The update row stays w′ = w − lr·∂L/∂w, as on the CNN page and as the shipped weights were trained.

## Part 1: One step

- **Phases.** The phase strip: digit → forward → back: head → back: MLP → back: attention → back: embedding → update.
  - Step runs one phase; Next digit finishes the digit and brings the next.
  - The gradient is computed in one go when the backward pass starts, and revealed stage by stage from the loss towards the input:
    - **head:** ∂L/∂(scores), ∂L/∂W_head and its bias, ∂L/∂(normalised average), the final LayerNorm, ∂L/∂(average), ∂L/∂z₂;
    - **MLP:** W₂, b₂, hidden, W₁, b₁, LayerNorm 2, ∂L/∂z₁;
    - **attention:** Wo, bo, the joined heads, ∂L/∂A for each head, ∂L/∂Q, ∂L/∂K, ∂L/∂V, Wq, Wk, Wv and their biases;
    - **embedding:** LayerNorm 1, ∂L/∂z₀, the position table, the patch tokens, the patch kernels and bias.
  - Each stage frames in green what it computed. Cells not computed yet are drawn faint, in place, so nothing moves between phases.
- **Three rows on the ViT line**, in a sideways-scrolling frame, with the row labels sticky at the left, at the explorer overview's zoom (8 px cells for activations, 6 px for weights):
  - **forward:** activations, and the weights where they act;
  - **backward:** the gradient of each, directly under it;
  - **update:** w′ under each weight, blank under activations.
  - The attention block keeps its matrix layout in the backward row: ∂L/∂A stands in the map's place, with ∂L/∂Kᵀ and ∂L/∂Vᵀ as columns above it.
  - The residual arcs appear in the forward row only.
- **The column builder becomes shared.** The column builder of `vit/` (views, lines, arcs, bars) moves into a `ui/` module. Both pages draw from one copy, and the explorer is checked unchanged after the move. The builder learns to stack several rows that share their columns.
- **Hover, green = computed from:**
  - **An activation:** as on `vit/` (`vitTrace`): upstream, and after attention graded by a row of the map.
  - **A gradient:** its sources downstream, up to the loss (`gradTrace`, new in `core/vit.js`).
    - Before the attention block's backward step, the sources are per token: the same token's later gradients, and every token's after the average, since the average reads all 16.
    - Through attention the grading turns over. The gradient of token j's key and value collects from every query i that read it, in proportion to A[i][j]: a column of the map, where the forward used a row. Own token 1; other tokens 0.9 × A[i][j] / the largest A[k][j] with k ≠ j.
    - The query's gradient comes from its own row of ∂L/∂A.
  - **A weight:** lit in all three rows, together with the two things whose product is its gradient: the input column it reads in the forward row, and the gradient column it feeds in the backward row.
  - **Readouts** write the arithmetic out:
    - ∂L/∂W[a, c] = Σ over tokens of input[t, a] × gradient[t, c];
    - the softmax backward of a row, ∂L/∂s = A ⊙ (∂L/∂A − Σ A·∂L/∂A);
    - w′ = w − lr · ∂L/∂w.

## Part 2: Many steps

- **Controls:**
  - Play / Pause; Restart (a new seed, everything cleared);
  - speed of 2, 10, 50, 200 or 500 digits a second, default 200;
  - learning rate on a log slider from 0.0003 to 0.3, default 0.003;
  - batch 1, 8 or 32.

  The page says that at 0.1 and above this model fails to train: transformers are sensitive to the learning rate (measured: lr 0.1 reached 29.7% and 0.3 diverged).
- **While playing, the step band follows**, as on the CNN page. Each redraw shows the digit on show at the end of its backward pass, at most ten times a second. A hovered cell stays probed.
- **Curves:**
  - loss per digit, and its running average;
  - training accuracy over the last 200 digits;
  - held-out accuracy on the 1,000 test digits every 500 digits.
- **Over training**, at 0, 300, 1,000, 3,000, 10,000 and 30,000 digits:
  - **both heads' attention maps** for one fixed digit: the explorer's default, the first held-out 7;
  - **the position table as similarities:** for each of the 16 slots, a 4×4 picture of how alike its position row is to every other slot's, laid out on the patch grid, as in the ViT paper.

  The note says what was measured: no grid forms at this scale, yet removing position costs the model most of its accuracy.
- **Recent digits:** the last 20, each with the guess made at the time.
- **Hand-off:**
  - "Use these weights in the ViT explorer" stores the weights in local storage, under a ViT key of their own, next to the CNN's in `ui/trained.js`.
  - The stored weights carry their held-out accuracy, measured at hand-off with position on and patches in order, and with position off.
  - `vit/` loads them at start, says so in a banner with a link back to the shipped weights, and its wording reads the stored figures.
  - "Download JSON" saves the same object.

## Part 3: CNN against ViT

- **Offline: `tools/vit_vs_cnn.mjs` → `models/vit_vs_cnn.json`.**
  - Both models, at 100, 300, 1,000 and 3,000 training digits (balanced, the first N/10 of each class), with 5 seeds each, 30,000 SGD steps.
  - CNN: random He-uniform start, lr 0.1. ViT: `init(seed)`, lr 0.003. The same shuffled order for both models at a given seed.
  - The file records the held-out accuracy of every run, the settings and the date. The script runs the 40 trainings in parallel child processes, a few minutes on the laptop.
- **The chart.**
  - Held-out accuracy (y) against the number of training digits (x, log scale), one dot per seed for each model, and a line through the means.
  - CNN in the landing page's CNN colour, ViT in its token colour.
  - The text reads the gap and the spread from the file, not from hard-coded numbers.
- **The live race.**
  - A size switch: 100 (default), 300, 1,000 or 3,000 digits. Play, Pause, Restart; speed of 50 or 200 digits a second.
  - Both models train on the same shuffled stream of the chosen digits, each at its own learning rate, and stop at 30,000 steps, like the offline runs.
  - Held-out accuracy of both every 500 digits, as two curves on one chart.
  - When the race ends, its two results are added to the measured chart as hollow dots marked "this run".
  - The text says that one run is one dot, and that at 100 digits the gap held in every measured seed.

## Part 4: Code and checks

- **`core/vit.js`:**
  - `lossAndGrad` also returns `back`, the gradient of every drawn activation: scores, normalised average, average, z₂, hidden, z₁, joined heads, ∂L/∂A per head, Q, K, V, LayerNorm-1 output, z₀ and patch tokens. These are the ones the backward pass already computes, kept rather than discarded.
  - `gradTrace(S, back, probe)` is the backward green, with the rules of Part 1.
  - `BACK_STAGES` maps each phase to the tensors it reveals.
- **`ui/vitline.js` (new):** the column builder moved out of `vit/index.html`, generalised to several rows sharing columns. `vit/index.html` imports it.
- **`ui/trained.js`:** `saveTrainedViT`, `loadTrainedViT`, `clearTrainedViT`, under the key `dl-explorer.vit-weights.v1`. The CNN functions are unchanged.
- **`tools/vit_vs_cnn.mjs`, `models/vit_vs_cnn.json`.**
- **`vit-train/index.html`.**
- **`index.html`:** the ViT section gets its Training card and its badge becomes "2 pages".
- **`project_notes.md`:**
  - the shipped weights' 82.0% is noted as seed 1's, with seeds 2 and 3 at 72.3% and 70.1% under the same settings;
  - the data-hunger open question is answered;
  - the position-table finding is recorded.
- **Checks:**
  - **Node:**
    - `back`'s shapes;
    - each `back` tensor against central finite differences of the loss with respect to that activation, on a few cells each;
    - `gradTrace`'s rules: a column grading through attention, own token 1, before attention per token;
    - the existing tests still pass.
  - **Headless Edge:**
    - the phases reveal stage by stage, with the faint-cell counts falling to 0;
    - a hover in each row lights the matching cells in the others;
    - play, driven frame by frame, ends with weights equal to a plain `core/vit.js` loop over the same digits (hash);
    - the race and the measured chart render, and the race's dots land on the chart;
    - the hand-off writes the key, and `vit/` then shows the banner;
    - `vit/` is unchanged after the builder moves (scenario values and a screenshot comparison);
    - the CNN pages' DOM is identical after the `ui/trained.js` change;
    - no console errors at 1640, 1000 and 500 px, in light and dark.

## Left for later

- A 3D view of the ViT, for the explorer and this page.
- A class-token variant with two blocks; Adam as an optimiser on the page.
- Augmentation or more data, to see whether a ViT trained on the full 60,000 MNIST digits learns a grid-like position table.
