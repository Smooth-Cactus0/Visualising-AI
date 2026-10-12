# Training Page 3D Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "2D | 3D" switch on the training page's One-step band, where 3D shows the forward activations, the gradients beneath them and the weight triplets as blocks of cubes, with the same phases and hover readouts, built on a block renderer shared with the 3D explorer.

**Architecture:** The 3D explorer's scene code (cubes, highlights, connectors, labels, camera, picking, loop) moves verbatim into `ui/blocks3d.js` behind a small API; the explorer imports it with no change in behaviour, proven by pixel-identical headless screenshots. The training page adds a container, a toggle and a 3D renderer that declares its blocks in five rows and maps picked cubes to the probes the 2D readout already understands. Spec: `docs/superpowers/specs/2026-10-12-cnn-training-3d-design.md`.

**Tech Stack:** Plain HTML + ES modules, three.js r128 UMD from cdnjs (global `THREE`), headless Edge for checks, Node 24 for the unchanged core tests.

## Global Constraints

- No build step; `core/` DOM-free; `ui/` shared DOM helpers; external scripts only from cdnjs or jsDelivr, pinned (three.js r128 is already used).
- The explorer's behaviour does not change: same scene-graph order (lights, net, hl, aux, stage), same materials, same camera defaults and easing, same picking semantics.
- Colour grammar: green = receptive field and gradient sources; diverging colours for signed values; grey for the digit. Theme-aware; reduced motion respected (no fade, no camera easing).
- Write files with the Write tool; run multi-line scripts from files (`python -I <path>`), not from heredocs.
- Local commits per task on a feature branch; one push at the end after the checks, reported.
- NZ English.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ui/blocks3d.js` | `createBlockView(viewEl, options)`: scene, blocks, overlays, labels, camera, picking, loop; `colorFor` | 1 |
| `cnn-3d/index.html` | Module script rewired to the shared view; everything page-specific stays | 1 |
| `cnn-train/index.html` | Tokens and styles for the 3D view, toggle markup, `#band2d`/`#band3d`, three.js script tag, the 3D renderer, pick mapping, highlights, camera presets | 2 |
| `CLAUDE.md`, `project_notes.md`, the spec | Record the module and the result | 3 |

---

### Task 1: Extract the block renderer and rewire the explorer

**Files:**
- Create: `ui/blocks3d.js`
- Modify: `cnn-3d/index.html` (the whole module script)

**Interfaces:**
- Consumes: `isDark`, `cssVar` from `ui/grid.js`; global `THREE`.
- Produces: `createBlockView(viewEl, { explode = 0.7, layout = 'side', gapX = 5, gapZ = 7, rowGap = 22, lateral = null, linkStages = false, ease = 0.12 }) → view` with `view.setBlocks(defs) → blocks`, `view.refresh()`, `view.layout`, `view.explode`, `view.blocks`, `view.blockById(id)`, `view.voxelPos(b, r, c, ch)`, `view.bounds(b, r0, r1, c0, c1, ch0, ch1)`, `view.flowRange() → [lo, hi, box]`, `view.flowAxis()`, `view.centerOf(ids)`, `view.boxAt(...)`, `view.boxSpanZ(...)`, `view.connect(boxA, boxB, mat, group)`, `view.mats`, `view.hl`, `view.aux`, `view.onPick(cb)`, `view.setGoal({ theta, phi, dist, target })`, `view.cam`, `view.goal`, `view.camera`, `view.resize()`, `view.start()`. Block def: `{ id, label?, rows, cols, chans, mode, data(ch), scale: number | () => number, row = 0, x0?, hidden? }`.

- [ ] **Step 1: Create the module**

Create `ui/blocks3d.js`:

```js
// ui/blocks3d.js — blocks-of-cubes view shared by the 3D explorer and the training page's 3D representation.
// Owns the three.js scene, the instanced cube meshes, highlight boxes and connectors, labels, the orbit camera,
// picking and the render loop. Needs the global THREE (r128 UMD) loaded by the page, the page's --scene token
// for the background, and a .lbl style for labels. Everything page-specific (which blocks, camera presets,
// what a picked cube means) stays in the page.
import { isDark, cssVar } from './grid.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const C_NEG = new THREE.Color(0x2a6fd6), C_POS = new THREE.Color(0xe5641e);

/** Cube colour for a value: 'gray' maps 0–1 to ink on the scene tone, 'div' maps −scale..scale blue to orange. */
export function colorFor(v, scale, mode) {
  const c = new THREE.Color(); const zero = isDark() ? new THREE.Color(0x3a404a) : new THREE.Color(0xdfe2dc);
  if (mode === 'gray') { const t = clamp(v, 0, 1); return c.copy(zero).lerp(isDark() ? new THREE.Color(0xf0f0f0) : new THREE.Color(0x202020), t); }
  const t = clamp(v / scale, -1, 1); return c.copy(zero).lerp(t < 0 ? C_NEG : C_POS, Math.abs(t));
}

export function createBlockView(viewEl, { explode = 0.7, layout = 'side', gapX = 5, gapZ = 7, rowGap = 22, lateral = null, linkStages = false, ease = 0.12 } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1)); viewEl.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(-40, 60, 80); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0xffffff, 0.25); dl2.position.set(60, -20, -40); scene.add(dl2);
  const net = new THREE.Group(), hl = new THREE.Group(), aux = new THREE.Group(); scene.add(net, hl, aux);
  const cubeGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const mats = {
    stageLine: new THREE.LineBasicMaterial({ color: 0x8a93a3, transparent: true, opacity: 0.45 }),
    rfLine: new THREE.LineBasicMaterial({ color: 0x3fae4b, transparent: true, opacity: 0.8 }),
    swLine: new THREE.LineBasicMaterial({ color: 0xcc9a00, transparent: true, opacity: 0.9 }),
  };
  const stage = new THREE.Group(); scene.add(stage);
  Object.assign(mats, {
    hlFill: new THREE.MeshBasicMaterial({ color: 0x3fae4b, transparent: true, opacity: 0.22, depthWrite: false }),
    hlEdge: new THREE.LineBasicMaterial({ color: 0x2e8f3a }),
    swFill: new THREE.MeshBasicMaterial({ color: 0xffd94a, transparent: true, opacity: 0.25, depthWrite: false }),
    swEdge: new THREE.LineBasicMaterial({ color: 0xcc9a00 }),
    swOut: new THREE.MeshBasicMaterial({ color: 0xffd94a, transparent: true, opacity: 0.5, depthWrite: false }),
  });
  const view = { renderer, scene, camera, net, hl, aux, stage, mats, blocks: [], layout, explode };

  // ---- blocks ----
  const zOf = (ch) => -ch * (1 + view.explode);
  function layoutOffsets() { let z = 0; view.blocks.forEach((b) => { b.z0 = z; z -= b.chans * (1 + view.explode) + gapZ; }); }
  function voxelPos(b, r, c, ch) { const y = (b.rows - 1 - r) - (b.rows - 1) / 2 + (b.y0 || 0);
    if (view.layout === 'side') return new THREE.Vector3(b.x0 + c, y, zOf(ch));
    return new THREE.Vector3(c - (b.cols - 1) / 2, y, b.z0 + zOf(ch)); }
  // bounding box of a cell range across channels ch0..ch1 (indices may run outside the block, e.g. padding)
  function bounds(b, r0, r1, c0, c1, ch0, ch1) { const bx = new THREE.Box3(); [[r0, c0, ch0], [r1, c1, ch1], [r0, c1, ch0], [r1, c0, ch1]].forEach(([r, c, ch]) => { const p = voxelPos(b, r, c, ch); bx.expandByPoint(p.clone().addScalar(-0.5)); bx.expandByPoint(p.clone().addScalar(0.5)); }); return bx; }
  const flowAxis = () => (view.layout === 'side' ? 'x' : 'z');
  const flowSign = () => (view.layout === 'side' ? 1 : -1);
  function faceCorners(bx, towardsPositive) { // 4 corners of the face at the flow-axis extreme
    const a = flowAxis(); const v = (towardsPositive ? bx.max[a] : bx.min[a]); const pts = [];
    const others = ['x', 'y', 'z'].filter((k) => k !== a);
    for (const u of [bx.min[others[0]], bx.max[others[0]]]) for (const w of [bx.min[others[1]], bx.max[others[1]]]) { const p = new THREE.Vector3(); p[a] = v; p[others[0]] = u; p[others[1]] = w; pts.push(p); }
    return pts; }
  function connect(bxA, bxB, material, group) { const s = flowSign() > 0; const A = faceCorners(bxA, s), Bc = faceCorners(bxB, !s); const pts = []; for (let i = 0; i < 4; i++) pts.push(A[i], Bc[i]);
    const g = new THREE.BufferGeometry().setFromPoints(pts); group.add(new THREE.LineSegments(g, material)); }
  function updateStageLinks() { stage.clear(); if (!linkStages || view.layout !== 'depth') return;
    for (let i = 0; i + 1 < view.blocks.length; i++) { const A = view.blocks[i], Bb = view.blocks[i + 1]; connect(bounds(A, 0, A.rows - 1, 0, A.cols - 1, 0, A.chans - 1), bounds(Bb, 0, Bb.rows - 1, 0, Bb.cols - 1, 0, Bb.chans - 1), mats.stageLine, stage); } }
  function setBlocks(defs) {
    net.clear(); const cursors = {};
    view.blocks = defs.filter((d) => !d.hidden).map((d) => {
      const b = { ...d }; const row = b.row || 0;
      if (b.x0 == null) b.x0 = cursors[row] || 0;
      cursors[row] = b.x0 + b.cols + gapX; b.y0 = -row * rowGap;
      const mesh = new THREE.InstancedMesh(cubeGeo, mat, b.rows * b.cols * b.chans); mesh.userData.block = b; b.mesh = mesh; net.add(mesh);
      return b; });
    refresh(); return view.blocks;
  }
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
  function refresh() { // positions + colours
    layoutOffsets();
    view.blocks.forEach((b) => { let i = 0; const scale = typeof b.scale === 'function' ? b.scale() : b.scale;
      for (let ch = 0; ch < b.chans; ch++) { const d = b.data(ch);
        for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) { const v = d[r * b.cols + c];
          const t = b.mode === 'gray' ? clamp(v, 0, 1) : clamp(Math.abs(v) / scale, 0, 1);
          const sc = 0.16 + 0.84 * t; _s.set(sc, sc, sc); _m.compose(voxelPos(b, r, c, ch), _q, _s); b.mesh.setMatrixAt(i, _m);
          b.mesh.setColorAt(i, colorFor(v, scale, b.mode)); i++; } }
      b.mesh.instanceMatrix.needsUpdate = true; if (b.mesh.instanceColor) b.mesh.instanceColor.needsUpdate = true; });
    updateStageLinks();
  }
  const blockById = (id) => view.blocks.find((b) => b.id === id);

  // ---- overlay boxes ----
  function boxAt(b, r0, r1, c0, c1, ch, group, fillMat, edgeMat, padZ = 0.08) {
    const p0 = voxelPos(b, r0, c0, ch), p1 = voxelPos(b, r1, c1, ch);
    const w = c1 - c0 + 1, h = r1 - r0 + 1; const g = new THREE.BoxGeometry(w + 0.16, h + 0.16, 1 + 2 * padZ);
    const m = new THREE.Mesh(g, fillMat); m.position.set((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, p0.z); group.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat); e.position.copy(m.position); group.add(e); }
  function boxSpanZ(b, r0, r1, c0, c1, ch0, ch1, group, fillMat, edgeMat) { // one box spanning channels ch0..ch1
    const p0 = voxelPos(b, r0, c0, ch0), p1 = voxelPos(b, r1, c1, ch1);
    const g = new THREE.BoxGeometry(c1 - c0 + 1 + 0.16, r1 - r0 + 1 + 0.16, Math.abs(p1.z - p0.z) + 1 + 0.16);
    const m = new THREE.Mesh(g, fillMat); m.position.set((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, (p0.z + p1.z) / 2); group.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat); e.position.copy(m.position); group.add(e); }

  // ---- labels ----
  const labelEls = new Map();
  function syncLabels() {
    const rect = renderer.domElement.getBoundingClientRect();
    view.blocks.forEach((b) => { if (!b.label) return; let el = labelEls.get(b.id); if (!el) { el = document.createElement('div'); el.className = 'lbl'; viewEl.append(el); labelEls.set(b.id, el); }
      const [t, sh] = b.label.split('\n'); el.innerHTML = `<b>${t}</b>${sh}`;
      const bx = bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1); const anchor = new THREE.Vector3((bx.min.x + bx.max.x) / 2, bx.max.y + 0.9, view.layout === 'side' ? (bx.min.z + bx.max.z) / 2 : bx.max.z); anchor.project(camera);
      const vis = anchor.z < 1; el.style.display = vis ? '' : 'none';
      el.style.left = ((anchor.x + 1) / 2 * rect.width) + 'px'; el.style.top = ((1 - anchor.y) / 2 * rect.height) + 'px'; });
    for (const [id, el] of labelEls) { const b = blockById(id); if (!b || !b.label) { el.remove(); labelEls.delete(id); } }
  }

  // ---- orbit camera ----
  const cam = { theta: 0.55, phi: 1.1, dist: 80, target: new THREE.Vector3(40, 0, -4) };
  const camGoal = { ...cam, target: cam.target.clone() };
  function applyCam() { const t = cam.target; camera.position.set(t.x + cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta), t.y + cam.dist * Math.cos(cam.phi), t.z + cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta)); camera.lookAt(t); }
  function flowRange() { const a = flowAxis(); const bx = new THREE.Box3(); view.blocks.forEach((b) => bx.union(bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1))); return [bx.min[a], bx.max[a], bx]; }
  function centerOf(ids) { const bx = new THREE.Box3(); ids.forEach((id) => { const b = blockById(id); bx.union(bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1)); }); return bx.getCenter(new THREE.Vector3()); }
  function setGoal({ theta, phi, dist, target }) { if (theta != null) camGoal.theta = theta; if (phi != null) camGoal.phi = phi; if (dist != null) camGoal.dist = dist; if (target) camGoal.target.copy(target); }
  // lateral travel along the flow axis
  let latDrag = false;
  if (lateral) {
    lateral.addEventListener('pointerdown', () => latDrag = true); lateral.addEventListener('pointerup', () => latDrag = false);
    lateral.oninput = () => { const [lo, hi] = flowRange(); const a = flowAxis(); const t = +lateral.value / 1000; camGoal.target[a] = flowSign() > 0 ? lo + t * (hi - lo) : hi - t * (hi - lo); };
  }
  function syncLateral() { if (!lateral || latDrag) return; const [lo, hi] = flowRange(); const a = flowAxis(); const t = flowSign() > 0 ? (cam.target[a] - lo) / (hi - lo) : (hi - cam.target[a]) / (hi - lo); lateral.value = Math.round(clamp(t, 0, 1) * 1000); }
  let drag = null;
  const cv3 = renderer.domElement;
  cv3.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey, moved: false }; cv3.setPointerCapture(e.pointerId); });
  cv3.addEventListener('pointermove', (e) => {
    if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; if (Math.abs(dx) + Math.abs(dy) > 0) drag.moved = true;
      if (drag.btn === 2 || drag.shift) { const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1); const k = cam.dist * 0.0016;
        cam.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k); }
      else { cam.theta -= dx * 0.006; cam.phi = clamp(cam.phi - dy * 0.006, 0.15, Math.PI - 0.15); }
      Object.assign(camGoal, { theta: cam.theta, phi: cam.phi, dist: cam.dist }); camGoal.target.copy(cam.target); return; }
    pick(e); });
  cv3.addEventListener('pointerup', () => { drag = null; });
  cv3.addEventListener('pointerleave', () => { drag = null; if (lastKey !== null) { lastKey = null; if (pickCb) pickCb(null); } });
  cv3.addEventListener('contextmenu', (e) => e.preventDefault());
  cv3.addEventListener('wheel', (e) => { e.preventDefault();
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const a = flowAxis(); const [lo, hi] = flowRange(); const nv = clamp(cam.target[a] + flowSign() * e.deltaX * 0.05, lo, hi); cam.target[a] = nv; camGoal.target[a] = nv; return; }
    cam.dist = clamp(cam.dist * Math.exp(e.deltaY * 0.0012), 6, 300); camGoal.dist = cam.dist; }, { passive: false });

  // ---- picking ----
  let pickCb = null, lastKey = null;
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pick(e) { const r = cv3.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera); const hits = ray.intersectObjects(net.children);
    let hit = null; if (hits.length) { const h = hits[0]; const b = h.object.userData.block; const id = h.instanceId; const per = b.rows * b.cols; const ch = Math.floor(id / per), rem = id % per; hit = { block: b, r: Math.floor(rem / b.cols), c: rem % b.cols, ch }; }
    const key = hit ? `${hit.block.id}:${hit.r}:${hit.c}:${hit.ch}` : null;
    if (key !== lastKey) { lastKey = key; if (pickCb) pickCb(hit); } }

  // ---- loop ----
  function resize() { const w = viewEl.clientWidth, h = viewEl.clientHeight; if (!w || !h) return; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  addEventListener('resize', resize); new ResizeObserver(resize).observe(viewEl);
  let running = false;
  function loop() { if (!running) return; // ease camera toward goal
    cam.theta += (camGoal.theta - cam.theta) * ease; cam.phi += (camGoal.phi - cam.phi) * ease; cam.dist += (camGoal.dist - cam.dist) * ease; cam.target.lerp(camGoal.target, ease);
    applyCam(); renderer.setClearColor(new THREE.Color(cssVar('--scene'))); renderer.render(scene, camera); syncLabels(); syncLateral(); requestAnimationFrame(loop); }

  Object.assign(view, { setBlocks, refresh, blockById, voxelPos, bounds, flowRange, flowAxis, centerOf, boxAt, boxSpanZ, connect, setGoal, cam, goal: camGoal, resize,
    onPick: (cb) => { pickCb = cb; }, start: () => { running = true; applyCam(); loop(); }, clear: (group) => group.clear() });
  return view;
}
```

- [ ] **Step 2: Rewire the explorer**

Replace the whole module script of `cnn-3d/index.html` (from `<script type="module">` to `</script>`) with the following. Everything after `// ---------- three ----------` that was scene code is gone; the rest is unchanged except the lines marked `// (shared)`.

```js
import { K, forward as forwardCore, footprint as footprintCore, defaultBank, inputPresets } from '../core/cnn.js';
import { installTerms } from '../ui/terms.js';
import { loadTrained, showTrainedBanner } from '../ui/trained.js';
import { createBlockView } from '../ui/blocks3d.js';
const N=16, C1=4, C2=8, NC=10;
const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const fmt=v=>{ if(Math.abs(v)<1e-9) return '0'; return v.toFixed(2).replace(/\.?0+$/,''); };
const isDark=()=> document.documentElement.dataset.theme==='dark' || (matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme!=='light');
const cssVar=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------- model ----------
const input=new Float32Array(N*N);
const bank=defaultBank.map(k=>k.slice()); // layer-1 kernels: hand-picked edge detectors
let stride=1,pad=1,dil=1,poolMode='max';
let l2=null,W=null,B=null;
const rnd=(n,s=1)=>Array.from({length:n},()=>+((Math.random()*2-1)*s).toFixed(2));
function rerollL2(){ l2=Array.from({length:C2},()=>Array.from({length:C1},()=>rnd(9))); }
function rerollFC(){ W=Array.from({length:C2},()=>rnd(NC,0.6)); B=rnd(NC,0.3); }
let S=null;
function forward(){ return forwardCore({input,N,bank,l2,W,B,stride,pad,dil,poolMode}); } // ReLU after each conv; the maths lives in ../core/cnn.js
const maxAbs=arrs=>Math.max(0.01,...arrs.flatMap(a=>Array.from(a,Math.abs)));

// ---------- three (the scene lives in ../ui/blocks3d.js) ----------
const view=createBlockView($('view'),{explode:0.7,layout:'side',lateral:$('lateral'),linkStages:true}); // (shared)
const { hl:hlGroup, aux:swGroup, mats }=view; // (shared)
const hlMat=mats.hlFill, hlEdge=mats.hlEdge, rfLineMat=mats.rfLine, swMat=mats.swFill, swEdge=mats.swEdge, outMat=mats.swOut, swLineMat=mats.swLine; // (shared)
const voxelPos=(b,r,c,ch)=>view.voxelPos(b,r,c,ch), bounds=(...a)=>view.bounds(...a), connect=(...a)=>view.connect(...a), boxAt=(...a)=>view.boxAt(...a), boxSpanZ=(...a)=>view.boxSpanZ(...a), blockById=id=>view.blockById(id); // (shared)

// blocks: {id,label,rows,cols,chans,x0,mode,scale,data(ch)->array, mesh, meta}
let blocks=[]; const GAPX=5;
function defineBlocks(){
  const O=S.O,Q=S.Q,Q2=S.Q2; let x=0; const defs=[];
  const add=(id,label,rows,cols,chans,mode,data,scale)=>{ defs.push({id,label,rows,cols,chans,mode,data,scale,x0:x}); x+=cols+GAPX; };
  add('input','input',N,N,1,'gray',()=>input,1);
  add('conv1','conv 1 + ReLU',O,O,C1,'div',ch=>S.a1[ch],maxAbs(S.a1));
  add('pool1','pool 1',Q,Q,C1,'div',ch=>S.p1[ch],maxAbs(S.p1));
  add('conv2','conv 2 + ReLU',Q,Q,C2,'div',ch=>S.a2[ch],maxAbs(S.a2));
  add('pool2','pool 2',Q2,Q2,C2,'div',ch=>S.p2[ch],maxAbs(S.p2));
  add('gap','global avg',1,1,C2,'div',ch=>[S.gap[ch]],maxAbs([S.gap]));
  add('scores','softmax',NC,1,1,'div',()=>S.prob,1);
  defs.forEach(d=>{ d.label+=`\n${d.id==='scores'?NC+' classes':d.id==='gap'?'1×1×'+C2:`${d.rows}×${d.cols}×${d.chans}`}`; });
  return defs;
}
function buildNet(){ blocks=view.setBlocks(defineBlocks()); updateSweep(); updateHighlights(); } // (shared)
function refreshNet(){ view.refresh(); updateSweep(); updateHighlights(); } // (shared)

// receptive field (propagation in ../core/cnn.js)
let probe=null; // {stage,r,c,ch} | {stage:'gap',ch} | {stage:'scores',r} | {stage:'input',r,c,ch}
function footprint(){
  const R={}; if(!probe) return R;
  // head: show only the direct link (averages read pool 2; classes read the averages). Everything upstream of that is the whole image.
  if(probe.stage==='gap'){ R.pool2={rect:[0,S.Q2-1,0,S.Q2-1],chs:new Set([probe.ch])}; R.gap={rect:[0,0,0,0],chs:new Set([probe.ch])}; return R; }
  if(probe.stage==='scores'){ R.gap={rect:[0,0,0,0],chs:null}; return R; }
  if(probe.stage==='input') return R;
  return footprintCore(probe,S);
}
function poolMath(p){ // the 2×2 block a pooled cube summarises, written out as on the 2D page
  if(p.stage!=='pool1'&&p.stage!=='pool2') return '';
  const first=p.stage==='pool1'; const src=first?S.a1[p.ch]:S.a2[p.ch], n=first?S.O:S.Q, out=first?S.p1[p.ch]:S.p2[p.ch], q=first?S.Q:S.Q2;
  const v=[]; for(const [r,c] of [[2*p.r,2*p.c],[2*p.r,2*p.c+1],[2*p.r+1,2*p.c],[2*p.r+1,2*p.c+1]]) if(r<n&&c<n) v.push(src[r*n+c]);
  return `<span class="m">${poolMode} over the 2×2 block of ${first?'conv 1':'conv 2'} + ReLU: ${poolMode}( ${v.map(fmt).join(', ')} ) = ${fmt(out[p.r*q+p.c])}</span>`;
}
function updateHighlights(){
  hlGroup.clear(); const R=footprint(); const ro=$('readout');
  for(const [st,h] of Object.entries(R)){ const b=blockById(st); if(!b) continue; const [r0,r1,c0,c1]=h.rect;
    if(h.chs===null && b.chans>1 && view.explode<0.05) boxSpanZ(b,r0,r1,c0,c1,0,b.chans-1,hlGroup,hlMat,hlEdge); // (shared)
    else for(let ch=0;ch<b.chans;ch++){ if(h.chs===null||h.chs.has(ch)) boxAt(b,r0,r1,c0,c1,ch,hlGroup,hlMat,hlEdge); } }
  // connector cone from the input patch up to the probed cube
  if(probe){ const chain=[]; for(const st of ['input','conv1','pool1','conv2','pool2','gap']){ const h=R[st]; if(!h) continue; const b=blockById(st); const [r0,r1,c0,c1]=h.rect;
      const lit=[...Array(b.chans).keys()].filter(ch=>h.chs===null||h.chs.has(ch)); chain.push(bounds(b,r0,r1,c0,c1,Math.min(...lit),Math.max(...lit))); }
    if(probe.stage==='scores') chain.push(bounds(blockById('scores'),probe.r,probe.r,0,0,0,0));
    for(let i=0;i+1<chain.length;i++) connect(chain[i],chain[i+1],rfLineMat,hlGroup); }
  if(!probe){ ro.textContent='Hover a cube.'; return; }
  const b=blockById(probe.stage);
  if(probe.stage==='input'){ ro.innerHTML=`<b>input</b> pixel (${probe.r}, ${probe.c}) = ${fmt(input[probe.r*N+probe.c])}`; return; }
  if(probe.stage==='scores'){ ro.innerHTML=`<b>class ${probe.r}</b>: score ${fmt(S.logits[probe.r])} → softmax ${(S.prob[probe.r]*100).toFixed(1)}%. Reads only the 8 averages, each weighted by its FC entry.<span class="m">score = Σ_f gap[f]·W[f,${probe.r}] + b = ${S.gap.map((g,f)=>`${fmt(g)}×${fmt(W[f][probe.r])}`).join(' + ')} + ${fmt(B[probe.r])}</span>`; return; }
  if(probe.stage==='gap'){ ro.innerHTML=`<b>global average, channel ${probe.ch+1}</b> = mean of pool-2 map ${probe.ch+1} (${S.Q2*S.Q2} cubes) = ${fmt(S.gap[probe.ch])}. Reads only that map; through it, the whole image.`; return; }
  const v=b.data(probe.ch)[probe.r*b.cols+probe.c]; const inR=R.input&&R.input.rect;
  ro.innerHTML=`<b>${b.label.split('\n')[0]}</b>, channel ${probe.ch+1}, cell (${probe.r}, ${probe.c}) = ${fmt(v)}` + (inR?` — sees input rows ${inR[0]}–${inR[1]}, cols ${inR[2]}–${inR[3]}: a <b>${inR[1]-inR[0]+1}×${inR[3]-inR[2]+1}</b> patch` : '') + poolMath(probe) + (probe.stage==='conv2'?`<span class="m">partial sums over the 4 input channels: ${S.parts[probe.ch].map(p=>fmt(p[probe.r*S.Q+probe.c])).join(' + ')} = ${fmt(S.c2[probe.ch][probe.r*S.Q+probe.c])} → ReLU → ${fmt(v)}</span>`:'');
}

// sweep
let sweep='off', swK=0, swStep=0, swTimer=null;
function swTotal(){ return sweep==='l1'? S.O*S.O : sweep==='l2'? S.Q*S.Q : 0; }
function updateSweep(){
  swGroup.clear(); const sl=$('swStep'); sl.max=Math.max(0,swTotal()-1); swStep=clamp(swStep,0,Math.max(0,swTotal()-1)); sl.value=swStep;
  const hint=$('swHint');
  if(sweep==='off'){ hint.textContent='Layer 1: a 3×3×1 window slides over the input and writes one cube of one output channel. Layer 2: a 3×3×4 window slides over the whole pool-1 block, and its four channel partial sums add into one cube.'; return; }
  if(sweep==='l1'){ const O=S.O, i=Math.floor(swStep/O), j=swStep%O; const bi=blockById('input'), bo=blockById('conv1');
    const e=dil*(K-1); const r0=i*stride-pad, c0=j*stride-pad; // may extend beyond input (padding)
    const p0=voxelPos(bi,r0,c0,0), p1=voxelPos(bi,r0+e,c0+e,0);
    const g=new THREE.BoxGeometry(e+1+0.16,e+1+0.16,1.16); const m=new THREE.Mesh(g,swMat); m.position.set((p0.x+p1.x)/2,(p0.y+p1.y)/2,p0.z); swGroup.add(m);
    const ed=new THREE.LineSegments(new THREE.EdgesGeometry(g),swEdge); ed.position.copy(m.position); swGroup.add(ed);
    boxAt(bo,i,i,j,j,swK,swGroup,outMat,swEdge);
    connect(bounds(bi,r0,r0+e,c0,c0+e,0,0), bounds(bo,i,i,j,j,swK,swK), swLineMat, swGroup);
    hint.innerHTML=`Layer 1, kernel ${swK+1}, output (${i}, ${j}): 9 products summed → ${fmt(S.c1[swK][i*O+j])}, ReLU → ${fmt(S.a1[swK][i*O+j])}. Written to channel ${swK+1} of conv 1.`; }
  else { const Q=S.Q, i=Math.floor(swStep/Q), j=swStep%Q; const bi=blockById('pool1'), bo=blockById('conv2');
    boxSpanZ(bi,i-1,i+1,j-1,j+1,0,C1-1,swGroup,swMat,swEdge);
    boxAt(bo,i,i,j,j,swK,swGroup,outMat,swEdge);
    connect(bounds(bi,i-1,i+1,j-1,j+1,0,C1-1), bounds(bo,i,i,j,j,swK,swK), swLineMat, swGroup);
    const parts=S.parts[swK].map(p=>p[i*Q+j]);
    hint.innerHTML=`Layer 2, kernel ${swK+1}, output (${i}, ${j}): the 3×3×4 window covers all four pool-1 channels. Per-channel partial sums ${parts.map(fmt).join(' + ')} = ${fmt(S.c2[swK][i*Q+j])}, ReLU → ${fmt(S.a2[swK][i*Q+j])}. Written to channel ${swK+1} of conv 2.`; }
}
function setSweep(mode){ sweep=mode; swStep=0; ['swOff','swL1','swL2'].forEach(id=>$(id).classList.remove('sel')); $(mode==='off'?'swOff':mode==='l1'?'swL1':'swL2').classList.add('sel');
  const sel=$('swK'); sel.innerHTML=''; const n=mode==='l2'?C2:C1; for(let k=0;k<n;k++){ const o=document.createElement('option'); o.value=k; o.textContent=k+1; sel.append(o); } swK=0; stopSw(); updateSweep(); }
function stopSw(){ clearInterval(swTimer); swTimer=null; $('swPlay').textContent='Play'; }
$('swOff').onclick=()=>setSweep('off'); $('swL1').onclick=()=>setSweep('l1'); $('swL2').onclick=()=>setSweep('l2');
$('swK').onchange=e=>{ swK=+e.target.value; updateSweep(); };
$('swStep').oninput=e=>{ swStep=+e.target.value; updateSweep(); };
$('swPrev').onclick=()=>{ swStep=Math.max(0,swStep-1); updateSweep(); };
$('swNext').onclick=()=>{ swStep=(swStep+1)%Math.max(1,swTotal()); updateSweep(); };
$('swPlay').onclick=()=>{ if(sweep==='off') setSweep('l1'); if(swTimer){ stopSw(); return; } $('swPlay').textContent='Pause';
  swTimer=setInterval(()=>{ const t=swTotal(); swStep=(swStep+1)%t; updateSweep(); if(swStep===t-1) stopSw(); },140); };

// camera presets (the orbit camera itself is shared)
function camPreset(name){
  const [lo,hi,all]=view.flowRange(); const span=hi-lo;
  const side=view.layout==='side';
  if(name==='all'){ view.setGoal(side?{theta:0.55,phi:1.15,dist:span*1.05}:{theta:0.65,phi:0.7,dist:span*1.3}); view.goal.target.copy(all.getCenter(new THREE.Vector3())); if(!side) view.goal.target.y=-4.5; } // depth: from high and nearly head-on, so the stages recede as a staircase and the input hides nothing
  if(name==='l1'){ view.setGoal(side?{theta:0.7,phi:1.2,dist:46}:{theta:0.9,phi:1.2,dist:44}); view.goal.target.copy(view.centerOf(['input','conv1'])); }
  if(name==='l2'){ view.setGoal(side?{theta:0.8,phi:1.2,dist:34}:{theta:0.9,phi:1.2,dist:36}); view.goal.target.copy(view.centerOf(['pool1','conv2'])); }
  if(name==='head'){ view.setGoal(side?{theta:0.9,phi:1.25,dist:30}:{theta:0.9,phi:1.25,dist:30}); view.goal.target.copy(view.centerOf(['pool2','scores'])); }
  document.querySelectorAll('.cams button[data-cam]').forEach(x=>x.classList.toggle('sel',x.dataset.cam===name));
}
$('recenter').onclick=()=>camPreset('all');
function setLayout(l){ view.layout=l; $('laySide').classList.toggle('sel',l==='side'); $('layDepth').classList.toggle('sel',l==='depth'); refreshNet(); camPreset('all'); } // (shared)
$('laySide').onclick=()=>setLayout('side'); $('layDepth').onclick=()=>setLayout('depth');
document.querySelectorAll('.cams button[data-cam]').forEach(x=>x.onclick=()=>camPreset(x.dataset.cam));
view.onPick(hit=>{ let np=null; // (shared) a picked cube becomes this page's probe
  if(hit){ const b=hit.block; np = b.id==='gap'? {stage:'gap',ch:hit.ch} : b.id==='scores'? {stage:'scores',r:hit.r} : {stage:b.id,r:hit.r,c:hit.c,ch:hit.ch}; }
  if(JSON.stringify(np)!==JSON.stringify(probe)){ probe=np; updateHighlights(); } });

// ---------- 2D input canvas ----------
const cIn=$('cIn'); const ctx=cIn.getContext('2d');
function drawIn(){ const cell=cIn.width/N; for(let r=0;r<N;r++)for(let c=0;c<N;c++){ const v=input[r*N+c]; const g=isDark()?40+v*200:245-v*225; ctx.fillStyle=`rgb(${g|0},${g|0},${g|0})`; ctx.fillRect(c*cell,r*cell,cell,cell); }
  ctx.strokeStyle=cssVar('--line'); ctx.lineWidth=1; for(let k=0;k<=N;k++){ ctx.beginPath();ctx.moveTo(k*cell+.5,0);ctx.lineTo(k*cell+.5,cIn.height);ctx.stroke(); ctx.beginPath();ctx.moveTo(0,k*cell+.5);ctx.lineTo(cIn.width,k*cell+.5);ctx.stroke(); } }
function set(r,c,v){ if(r>=0&&c>=0&&r<N&&c<N) input[r*N+c]=v; }
const inPresets=Object.fromEntries(Object.entries(inputPresets(N)).map(([name,make])=>[name,()=>input.set(make())])); // presets live in ../core/cnn.js
Object.entries(inPresets).forEach(([name,fn])=>{ const b=document.createElement('button'); b.textContent=name; b.onclick=()=>{ fn(); recompute(); }; $('inPresets').append(b); });
let painting=false, erase=false;
function paintAt(ev){ const rect=cIn.getBoundingClientRect(); const cell=rect.width/N; const c=Math.floor((ev.clientX-rect.left)/cell), r=Math.floor((ev.clientY-rect.top)/cell); if(r<0||c<0||r>=N||c>=N) return;
  set(r,c,erase?0:1); set(r+1,c,erase?0:Math.max(input[(r+1)*N+c]||0,.4)); set(r,c+1,erase?0:Math.max(input[r*N+c+1]||0,.4)); recompute(); }
cIn.addEventListener('pointerdown',ev=>{ painting=true; erase=ev.button===2||ev.shiftKey; cIn.setPointerCapture(ev.pointerId); paintAt(ev); });
cIn.addEventListener('pointermove',ev=>{ if(painting) paintAt(ev); });
cIn.addEventListener('pointerup',()=>painting=false);
cIn.addEventListener('contextmenu',ev=>ev.preventDefault());

// ---------- wiring ----------
function recompute(rebuild=false){ const old=S; S=forward(); drawIn();
  const shapeChanged=!old||old.O!==S.O||old.Q!==S.Q||old.Q2!==S.Q2;
  if(rebuild||shapeChanged||!blocks.length){ buildNet(); camPreset((document.querySelector('.cams button[data-cam].sel')||{dataset:{cam:'all'}}).dataset.cam); } else refreshNet(); }
$('stride').onchange=e=>{ stride=+e.target.value; recompute(true); };
$('pad').onchange=e=>{ pad=+e.target.value; recompute(true); };
$('dil').onchange=e=>{ dil=+e.target.value; recompute(true); };
function setPool(mode){ poolMode=mode; $('poolMax').classList.toggle('sel',mode==='max'); $('poolAvg').classList.toggle('sel',mode==='avg'); recompute(); }
$('poolMax').onclick=()=>setPool('max'); $('poolAvg').onclick=()=>setPool('avg');
$('explode').oninput=e=>{ view.explode=+e.target.value; refreshNet(); }; // (shared)
$('l2Rand').onclick=()=>{ rerollL2(); recompute(); };
$('fcRand').onclick=()=>{ rerollFC(); recompute(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>recompute());

installTerms({selectors:'.lede, .hint:not(#swHint)', tip:$('tip'), gloss:$('gloss')});
const trained = loadTrained();
if (trained) { trained.bank.forEach((k, i) => { bank[i] = k.slice(); }); l2 = trained.l2; W = trained.W; B = trained.B; showTrainedBanner(document.querySelector('h1'), trained);
  $('wnote').textContent = `All weights were trained on the training page (${trained.meta.digits} digits): the prediction means something for digits, and the mechanics are the same either way.`; }
else { rerollL2(); rerollFC(); }
inPresets['7'](); setSweep('off'); view.resize(); recompute(true); view.start(); // (shared)
```

- [ ] **Step 3: Syntax**

`node --check ui/blocks3d.js` and the extracted explorer module script with `node --check`. Expected: no output.

- [ ] **Step 4: Pixel-identical check**

Stage `old/` from `git show HEAD:` (`cnn-3d/index.html`, `core/cnn.js`, `ui/terms.js`, `ui/terms.css`, `ui/trained.js`, `ui/grid.js`) and `new/` from the working tree (plus `ui/blocks3d.js`). Inject the seeded `Math.random` shim after `<head>` in both pages. For each of four states, append to the init line (`view.start();` new, `loop();` old) nothing (state A), `setLayout('depth');` (B), `camPreset('l2');` (C), `setSweep('l1'); swStep=5; updateSweep();` (D). Serve, screenshot each at 1400×900 with `--virtual-time-budget=8000`, and compare old and new with `cmp`. Also dump both DOMs, strip scripts, diff. Expected: all four screenshot pairs identical, DOM identical, no console errors. If a pair differs, view both and the byte count of the difference; accept only a difference that is invisible side by side, and record it in the notes.

- [ ] **Step 5: Commit**

```bash
git add ui/blocks3d.js cnn-3d/index.html
git commit -m "refactor: share the 3D block renderer in ui/blocks3d.js"
```

---
### Task 2: The training page's 3D representation

**Files:**
- Modify: `cnn-train/index.html` (tokens, styles, markup, the three.js script tag, the 3D code)

**Interfaces:**
- Consumes: `createBlockView` from Task 1; the page's existing `cur`, `probe`, `weights`, `shownWeights()`, `gradShown()`, `collected`, `lr`, `reduced`, `footprint`, `maxAbs`, `fmt`, `readout()`, `renderPhase()`, `renderPlay()`, `renderAll()`, `startAnim()`.
- Produces: `rep` ('2d' | '3d'), `setRep(r)`, `render3d()`, `defs3d()`, `camPreset3d(name)`, `onPick3d(hit)`, `highlights3d()`; probe kind `pixel`.

- [ ] **Step 1: The 3D code, saved to the scratchpad as `train3d.js`**

```js
// ---------- 3D representation (blocks of cubes; the renderer is shared in ../ui/blocks3d.js) ----------
const VIEW_KEY = 'dl-explorer.train.view', GX = 5;
let rep = '2d', view3d = null, built3d = '', curPreset = 'whole', fitted3d = false;
function ensureView3d() {
  if (view3d) return;
  view3d = createBlockView($('view3d'), { explode: 0.7, layout: 'side', rowGap: 22, lateral: $('lateral3d'), ease: reduced ? 1 : 0.12 });
  view3d.onPick(onPick3d); view3d.start();
}
function setRep(r) {
  rep = r; try { localStorage.setItem(VIEW_KEY, r); } catch { /* storage blocked: the choice just is not remembered */ }
  $('rep2d').classList.toggle('sel', r === '2d'); $('rep3d').classList.toggle('sel', r === '3d');
  $('band2d').hidden = r === '3d'; $('band3d').hidden = r === '2d';
  if (r === '3d') { ensureView3d(); view3d.resize(); built3d = ''; }
  probe = null; renderAll();
  if (r === '3d') camPreset3d(curPreset);
}
const flatW = (M) => { const o = new Float32Array(C2 * NC); for (let f = 0; f < C2; f++) for (let c = 0; c < NC; c++) o[f * NC + c] = M[f][c]; return o; };
function defs3d() { // the blocks for the current phase: forward row, gradient row beneath it, then w · ∂L/∂w · w′ rows
  const S = cur.S, ph = cur.phase, g = gradShown(), w = shownWeights; // w() is read at draw time, so the update fade shows
  const ab = (arrs) => Math.max(...arrs.map(maxAbs)); const shape = (rows, cols, chans) => `${rows}×${cols}×${chans}`; const D = [];
  const fwd = [['in', 'digit', N, N, 1, 'gray', () => cur.input, 1]];
  if (ph >= 1) fwd.push(['conv1', 'conv 1 + ReLU', S.O, S.O, C1, 'div', (ch) => S.a1[ch], ab(S.a1)], ['pool1', 'pool 1', S.Q, S.Q, C1, 'div', (ch) => S.p1[ch], ab(S.p1)],
    ['conv2', 'conv 2 + ReLU', S.Q, S.Q, C2, 'div', (ch) => S.a2[ch], ab(S.a2)], ['pool2', 'pool 2', S.Q2, S.Q2, C2, 'div', (ch) => S.p2[ch], ab(S.p2)],
    ['gap', 'global avg', 1, 1, C2, 'div', (ch) => [S.gap[ch]], maxAbs(S.gap)], ['scores', 'softmax', NC, 1, 1, 'div', () => S.prob, 1]);
  let x = 0; const xs = {};
  for (const [id, title, rows, cols, chans, mode, data, scale] of fwd) { xs[id] = x; D.push({ id, label: `${title}\n${id === 'scores' ? NC + ' classes' : id === 'gap' ? '1×1×' + C2 : shape(rows, cols, chans)}`, rows, cols, chans, mode, data, scale, row: 0, x0: x }); x += cols + GX; }
  if (ph >= 2) { const B = cur.back;
    for (const [id, title, rows, cols, chans, data, scale] of [['conv1', '∂L/∂conv 1', S.O, S.O, C1, (ch) => B.conv1[ch], ab(B.conv1)], ['pool1', '∂L/∂pool 1', S.Q, S.Q, C1, (ch) => B.pool1[ch], ab(B.pool1)],
      ['conv2', '∂L/∂conv 2', S.Q, S.Q, C2, (ch) => B.conv2[ch], ab(B.conv2)], ['pool2', '∂L/∂pool 2', S.Q2, S.Q2, C2, (ch) => B.pool2[ch], ab(B.pool2)],
      ['gap', '∂L/∂gap', 1, 1, C2, (ch) => [B.gap[ch]], maxAbs(B.gap)], ['scores', '∂L/∂scores = p − y', NC, 1, 1, () => B.scores, 1]])
      D.push({ id: 'd_' + id, label: `${title}\n${shape(rows, cols, chans)}`, rows, cols, chans, mode: 'div', data, scale, row: 1, x0: xs[id] }); }
  const sB = () => ab(w().bank), sL = () => Math.max(...w().l2.flat().map(maxAbs)), sW = () => maxAbs(flatW(w().W)), sb = () => maxAbs(w().B);
  const gB = g ? ab(g.bank) : 1, gL = g ? Math.max(...g.l2.flat().map(maxAbs)) : 1, gW = g ? maxAbs(flatW(g.W)) : 1, gb = g ? maxAbs(g.B) : 1;
  const nxt = (a, d) => a.map((v, q) => v - lr * d[q]);
  const kinds = [['w', 2, 'w']]; if (g) kinds.push(['∂L/∂w', 3, 'g'], ['w′ = w − lr·∂L/∂w', 4, 'n']);
  for (const [title, row, kind] of kinds) {
    const ver = (getW, getG) => (kind === 'g' ? getG : kind === 'n' ? (ch) => nxt(getW(ch), getG(ch)) : getW);
    const sc = (sw, sg) => (kind === 'g' ? sg : sw);
    let x = 0;
    D.push({ id: 'bank_' + kind, label: row === 2 ? 'layer-1 kernels\n3×3×4' : title + '\n ', rows: 3, cols: 3, chans: C1, mode: 'div', data: ver((ch) => w().bank[ch], (ch) => g.bank[ch]), scale: sc(sB, gB), row, x0: x }); x += 3 + GX;
    for (let k = 0; k < C2; k++) { D.push({ id: `l2_${k}_${kind}`, label: row === 2 && k === 0 ? 'layer-2 kernels\n8 × 3×3×4' : null, rows: 3, cols: 3, chans: C1, mode: 'div', data: ver((ch) => w().l2[k][ch], (ch) => g.l2[k][ch]), scale: sc(sL, gL), row, x0: x }); x += 3 + GX; }
    D.push({ id: 'W_' + kind, label: row === 2 ? 'FC weights\n8×10' : null, rows: C2, cols: NC, chans: 1, mode: 'div', data: ver(() => flatW(w().W), () => flatW(g.W)), scale: sc(sW, gW), row, x0: x }); x += NC + GX;
    D.push({ id: 'B_' + kind, label: row === 2 ? 'FC biases\n1×10' : null, rows: 1, cols: NC, chans: 1, mode: 'div', data: ver(() => w().B, () => g.B), scale: sc(sb, gb), row, x0: x });
  }
  return D;
}
function render3d() {
  const key = `${cur.i}:${cur.phase}:${collected.length}:${cur.applied}`;
  if (key !== built3d) { built3d = key; view3d.setBlocks(defs3d()); if (!fitted3d) { fitted3d = true; camPreset3d('whole'); } }
  else view3d.refresh();
  highlights3d();
}
function fit(ids, theta, phi, pad) { // frame the given blocks: distance from their extent and the camera's fields of view
  const bx = new THREE.Box3(); for (const id of ids) { const b = view3d.blockById(id); if (b) bx.union(view3d.bounds(b, 0, b.rows - 1, 0, b.cols - 1, 0, b.chans - 1)); }
  if (bx.isEmpty()) return;
  const size = bx.getSize(new THREE.Vector3()), centre = bx.getCenter(new THREE.Vector3());
  const vf = (38 / 2) * Math.PI / 180, hf = Math.atan(Math.tan(vf) * view3d.camera.aspect);
  const dist = Math.max(size.y / (2 * Math.tan(vf)), size.x / (2 * Math.tan(hf))) * pad + size.z;
  view3d.setGoal({ theta, phi, dist, target: centre });
}
function camPreset3d(name) {
  curPreset = name; const inRows = (rs) => view3d.blocks.filter((b) => rs.includes(b.row || 0)).map((b) => b.id);
  if (name === 'whole') fit(inRows([0, 1, 2, 3, 4]), 0.45, 1.25, 1.2);
  if (name === 'forward') fit(inRows([0]), 0.55, 1.15, 1.15);
  if (name === 'backward') fit(inRows([0, 1]), 0.55, 1.2, 1.15);
  if (name === 'weights') fit(inRows([2, 3, 4]), 0.45, 1.2, 1.15);
  document.querySelectorAll('#view3d .cams button[data-cam]').forEach((x) => x.classList.toggle('sel', x.dataset.cam === name));
}
function onPick3d(hit) { // a picked cube becomes one of the probes the readout already understands
  let np = null;
  if (hit) { const { block: b, r, c, ch } = hit; const id = b.id;
    if (id === 'in') np = { kind: 'pixel', r, c };
    else if (id === 'gap' || id === 'd_gap') np = { kind: 'gap', f: ch };
    else if (id === 'scores' || id === 'd_scores') np = { kind: 'bar', c: r };
    else if (id.startsWith('d_')) np = { kind: 'back', stage: id.slice(2), r, c, ch };
    else if (id.startsWith('bank_')) np = { kind: 'w', layer: 'bank', k: ch, q: r * 3 + c };
    else if (id.startsWith('l2_')) np = { kind: 'w', layer: 'l2', k: +id.split('_')[1], s: ch, q: r * 3 + c };
    else if (id.startsWith('W_')) np = { kind: 'w', layer: 'W', f: r, c };
    else if (id.startsWith('B_')) np = { kind: 'w', layer: 'B', c };
    else np = { kind: 'map', stage: id, r, c, ch }; }
  if (JSON.stringify(np) !== JSON.stringify(probe)) { probe = np; readout(); highlights3d(); }
}
function highlights3d() {
  const v = view3d; v.hl.clear(); const p = probe; if (!p || !cur.S) return;
  const B = (id) => v.blockById(id), M = v.mats;
  const box = (b, r0, r1, c0, c1, ch) => v.boxAt(b, r0, r1, c0, c1, ch, v.hl, M.hlFill, M.hlEdge);
  const whole = (b, ch) => { box(b, 0, b.rows - 1, 0, b.cols - 1, ch); return v.bounds(b, 0, b.rows - 1, 0, b.cols - 1, ch, ch); };
  const link = (a, b) => v.connect(a, b, M.rfLine, v.hl);
  if (p.kind === 'pixel') { const b = B('in'); if (b) box(b, p.r, p.r, p.c, p.c, 0); return; }
  if (p.kind === 'map') { // receptive field back through the forward row, as on the explorer
    const R = footprint({ stage: p.stage, rect: [p.r, p.r, p.c, p.c], ch: p.ch }, cur.S); const chain = [];
    for (const st of ['input', 'conv1', 'pool1', 'conv2', 'pool2']) { const h = R[st]; const b = B(st === 'input' ? 'in' : st); if (!h || !b) continue; const [r0, r1, c0, c1] = h.rect;
      const lit = [...Array(b.chans).keys()].filter((ch) => h.chs === null || h.chs.has(ch)); lit.forEach((ch) => box(b, r0, r1, c0, c1, ch)); chain.push(v.bounds(b, r0, r1, c0, c1, Math.min(...lit), Math.max(...lit))); }
    for (let i = 0; i + 1 < chain.length; i++) link(chain[i], chain[i + 1]); return; }
  if (p.kind === 'back') { const b = B('d_' + p.stage); if (b) box(b, p.r, p.r, p.c, p.c, p.ch); return; }
  if (p.kind === 'gap') { const pool2 = B('pool2'), gap = B('gap'), W = B('W_w'); let a = null, g2 = null;
    if (pool2) a = whole(pool2, p.f); if (gap) g2 = whole(gap, p.f); if (W) box(W, p.f, p.f, 0, NC - 1, 0); if (a && g2) link(a, g2); return; }
  if (p.kind === 'bar') { const gap = B('gap'), sc = B('scores'), W = B('W_w'); let a = null, s = null;
    if (gap) { for (let f = 0; f < C2; f++) box(gap, 0, 0, 0, 0, f); a = v.bounds(gap, 0, 0, 0, 0, 0, C2 - 1); }
    if (sc) { box(sc, p.c, p.c, 0, 0, 0); s = v.bounds(sc, p.c, p.c, 0, 0, 0, 0); }
    if (W) box(W, 0, C2 - 1, p.c, p.c, 0); if (a && s) link(a, s); return; }
  if (p.kind === 'w') { // the hovered weight in each of its rows, and the two blocks whose product it is
    const isFC = p.layer === 'W' || p.layer === 'B'; const a = isFC ? (p.layer === 'W' ? p.f : 0) : Math.floor(p.q / 3), c = isFC ? p.c : p.q % 3, ch = p.layer === 'bank' ? p.k : p.layer === 'l2' ? p.s : 0;
    const base = p.layer === 'l2' ? `l2_${p.k}` : p.layer; let target = null;
    for (const kind of ['w', 'g', 'n']) { const b = B(`${base}_${kind}`); if (!b) continue; box(b, a, a, c, c, ch); if (!target) target = v.bounds(b, a, a, c, c, ch, ch); }
    const srcs = [];
    if (p.layer === 'bank') { if (B('d_conv1')) srcs.push(whole(B('d_conv1'), p.k)); if (B('in')) srcs.push(whole(B('in'), 0)); }
    else if (p.layer === 'l2') { if (B('d_conv2')) srcs.push(whole(B('d_conv2'), p.k)); if (B('pool1')) srcs.push(whole(B('pool1'), p.s)); }
    else { const gap = B('gap'), sc = B('scores'); if (p.layer === 'W' && gap) { box(gap, 0, 0, 0, 0, p.f); srcs.push(v.bounds(gap, 0, 0, 0, 0, p.f, p.f)); } if (sc) { box(sc, p.c, p.c, 0, 0, 0); srcs.push(v.bounds(sc, p.c, p.c, 0, 0, 0, 0)); } }
    if (target) for (const s of srcs) link(s, target);
  }
}
$('rep2d').onclick = () => setRep('2d'); $('rep3d').onclick = () => setRep('3d');
$('recenter3d').onclick = () => camPreset3d(curPreset);
document.querySelectorAll('#view3d .cams button[data-cam]').forEach((x) => { x.onclick = () => camPreset3d(x.dataset.cam); });

```

- [ ] **Step 2: The edit script, `edit_train_3d.py` in the scratchpad**

```python
import pathlib, sys
sys.stdout.reconfigure(encoding='utf-8')
S = pathlib.Path(sys.argv[1])  # scratchpad, holds train3d.js
p = pathlib.Path(r'C:\Users\LouisA\repos\dl-explorer\cnn-train\index.html'); t = p.read_text(encoding='utf-8')
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:60]); t = t.replace(old, new)
# tokens and styles
rep("  --hl:#ffd94a; --hl2:#3fae4b;\n  --mono:", "  --hl:#ffd94a; --hl2:#3fae4b; --scene:#eef0ea;\n  --mono:")
rep("--accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a;", "--accent:#7fa7ea; --accent-soft:#263650; --zero:#2a2f38; --hl:#d9b32d; --hl2:#5ad66a; --scene:#1b1f26;", 2)
rep("@media (prefers-reduced-motion: reduce){ *{transition:none!important} }\n</style>", """#view3d{position:relative;background:var(--scene);border:1px solid var(--line);border-radius:10px;height:560px;overflow:hidden;touch-action:none;margin-top:6px}
#view3d canvas{display:block;width:100%!important;height:100%!important;border-radius:0}
.lbl{position:absolute;transform:translate(-50%,-100%);font-size:12.5px;color:var(--ink2);text-align:center;pointer-events:none;white-space:nowrap;line-height:1.3}
.lbl b{display:block;color:var(--ink);font-weight:600}
.cams{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap;z-index:1}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--panel)}
.seg button{border:0;border-radius:0}
.seg button+button{border-left:1px solid var(--line)}
button.sel{background:var(--accent-soft);border-color:var(--accent)}
.lat{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:12.5px;color:var(--ink2)}
.lat input{flex:1}
@media (prefers-reduced-motion: reduce){ *{transition:none!important} }
</style>""")
# markup: toggle, the 2D band wrapped, the 3D band
rep('    <span class="phase" id="phase">', '    <span class="seg"><button id="rep2d" class="sel">2D</button><button id="rep3d">3D</button></span>\n    <span class="phase" id="phase">')
rep('  <div class="cap" style="margin-top:12px">Forward: what came in</div>', '  <div id="band2d">\n  <div class="cap" style="margin-top:12px">Forward: what came in</div>')
rep('  <div class="trips" id="trips"></div>\n', """  <div class="trips" id="trips"></div>
  </div>
  <div id="band3d" hidden>
    <div id="view3d"><div class="cams"><button data-cam="whole" class="sel">whole</button><button data-cam="forward">forward</button><button data-cam="backward">backward</button><button data-cam="weights">weights</button><button id="recenter3d" title="reset orientation and zoom">⟲ recentre</button></div></div>
    <div class="lat"><span>start</span><input type="range" id="lateral3d" min="0" max="1000" value="500" title="scroll along the network"><span>end</span></div>
  </div>
""")
rep('<script type="module">\n', '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n<script type="module">\n')
# script
rep("import { fmt, maxAbs, drawGrid, cellAt, cssVar } from '../ui/grid.js';\n", "import { fmt, maxAbs, drawGrid, cellAt, cssVar } from '../ui/grid.js';\nimport { createBlockView } from '../ui/blocks3d.js';\n")
rep("function renderAll() { renderPhase(); renderForward(); renderBackward(); renderTrips(); readout(); renderPlay(); }",
    "function renderAll() { renderPhase(); if (rep === '3d') render3d(); else { renderForward(); renderBackward(); renderTrips(); } readout(); renderPlay(); }")
rep("function startAnim() { anim = { t0: performance.now(), t: 0 }; const tick = () => { if (!anim) return; const t = (performance.now() - anim.t0) / 300; if (t >= 1) { anim = null; renderTrips(); return; } anim.t = t; renderTrips(); requestAnimationFrame(tick); }; requestAnimationFrame(tick); }",
    "function paintWeights() { if (rep === '3d') view3d.refresh(); else renderTrips(); }\nfunction startAnim() { anim = { t0: performance.now(), t: 0 }; const tick = () => { if (!anim) return; const t = (performance.now() - anim.t0) / 300; if (t >= 1) { anim = null; paintWeights(); return; } anim.t = t; paintWeights(); requestAnimationFrame(tick); }; requestAnimationFrame(tick); }")
rep("  if (p.kind === 'map') { const name = p.stage.replace(/([0-9])/, ' $1');",
    "  if (p.kind === 'pixel') { ro.innerHTML = `<b>input</b> pixel (${p.r}, ${p.c}) = ${fmt(cur.input[p.r * N + p.c])}.`; return; }\n  if (p.kind === 'map') { const name = p.stage.replace(/([0-9])/, ' $1');")
rep("// ---------- controls ----------", (S / 'train3d.js').read_text(encoding='utf-8') + "// ---------- controls ----------")
rep("  $('status').remove();\n  reset();\n", "  $('status').remove();\n  reset();\n  let stored = '2d'; try { stored = localStorage.getItem(VIEW_KEY) || '2d'; } catch { /* storage blocked */ }\n  if (stored === '3d') setRep('3d');\n")
p.write_text(t, encoding='utf-8', newline='\n'); print('cnn-train: 3D representation added')
```

Run: `python -I <scratchpad>/edit_train_3d.py <scratchpad>`. Then extract the module script and `node --check` it.

- [ ] **Step 3: Check headless**

Stage the training page with `core/`, `ui/` (including `blocks3d.js`) and `data/`. Variants appended after `reset();` in the init: (a) `setRep('3d');` (b) `setRep('3d'); step(); step(); probe = { kind: 'w', layer: 'W', f: 2, c: 3 }; readout(); highlights3d();` (c) `setRep('3d'); step(); step(); step();` (d) variant (b) with `data-theme="dark"` on `<html>`. For each: dump the DOM (no console errors; `#band3d` not hidden and `#band2d` hidden; `.lbl` count: (a) 5 labels (digit and the four weight-row titles), (b) 7 + 6 + 4 + 2 = 19, (c) 7 + 6 + 4 = 17; `#readout` in (b) contains `W[row 3, class 3]` and `∂L/∂W = gap[3] × (p[3] − y[3])`) and take a 1400×900 screenshot; look at (b), (c) and (d) and at the `forward`, `backward` and `weights` presets of (b) (append `camPreset3d('weights');` and so on). Adjust the preset numbers in `camPreset3d` until every preset frames its rows with nothing cut off, and record the final numbers.

- [ ] **Step 4: Commit**

```bash
git add cnn-train/index.html
git commit -m "feat(cnn-train): 3D representation of the step band"
```

---

### Task 3: Whole-site check, documentation, deploy

**Files:**
- Modify: `CLAUDE.md`, `project_notes.md`, `docs/superpowers/specs/2026-10-12-cnn-training-3d-design.md` (only if a measured value or behaviour differs: the phase-3 rows, the preset numbers)

- [ ] **Step 1: Everything from one server**

Serve the repo root; dump the four pages with console logging (no page errors); `node core/cnn.test.mjs && node core/data.test.mjs && node core/train.test.mjs` pass. The training page's 2D checks from the previous plan (step ×3 and the FC-weight probe) still give the same readouts.

- [ ] **Step 2: Docs**

`CLAUDE.md`: the `ui/` line gains `blocks3d.js` (the 3D block renderer: scene, cubes, highlights, labels, orbit camera, picking) and the three.js sentence says both the 3D explorer and the training page use it. `project_notes.md` (project-notes skill, Update): Method (the 2D | 3D switch, the five rows, the shared renderer), Results (the explorer's pixel check outcome, the 3D checks), Decisions (the brainstorm's: toggle on the training page, shared module over a copy, rows, side-by-side only; plus any made while building), Map (`ui/blocks3d.js`, the new spec and plan), Status. Spec: fix the phase-3 line to "rows 3 and 4 hidden" (the backward row stays, as in 2D) and record the preset numbers.

- [ ] **Step 3: Commit, merge, deploy**

```bash
git add CLAUDE.md project_notes.md docs/superpowers/specs/2026-10-12-cnn-training-3d-design.md
git commit -m "docs: record the training page's 3D representation"
git switch main && git merge --ff-only feat/train-3d && git branch -d feat/train-3d
git push origin main
```

Watch the Pages run; fetch the live training page with a cache-busting query and confirm it contains `id="view3d"` and `ui/blocks3d.js`. Report the push and the URL.

## Plan self-review

- Spec coverage: module API (Task 1), explorer unchanged with the pixel check (Task 1), toggle, rows, phases, pick mapping, highlights, presets, play behaviour, theme and motion (Task 2), checks (1, 2, 3), docs (3).
- Placeholders: none. The camera preset numbers are explicitly tuned in Task 2 Step 3 from screenshots, as the spec says.
- Names: `createBlockView`, `setBlocks`, `refresh`, `blockById`, `voxelPos`, `bounds`, `boxAt`, `boxSpanZ`, `connect`, `mats.hlFill/hlEdge/rfLine`, `hl`, `onPick`, `setGoal`, `goal`, `camera`, `resize`, `start` (Task 1 → 2); block ids `in`, `conv1`, `pool1`, `conv2`, `pool2`, `gap`, `scores`, `d_*`, `bank_<kind>`, `l2_<k>_<kind>`, `W_<kind>`, `B_<kind>` used consistently in `defs3d`, `onPick3d` and `highlights3d`.
