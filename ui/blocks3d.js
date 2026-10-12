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
