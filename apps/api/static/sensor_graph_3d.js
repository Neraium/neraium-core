/**
 * NeraiumSensorGraph3D — live sensor mesh: nodes = sensors, edges = relationships.
 * Visual language: stable = smooth teal flow; stress = jitter + amber shift; critical = fragmentation + red.
 * Rendering is isolated from app.js; data mapping happens in app.js before calling update().
 */
(function initNeraiumSensorGraph3D(global) {
  const THREE = global.THREE;
  if (!THREE) {
    global.NeraiumSensorGraph3D = {
      init() {},
      update() {},
      dispose() {},
    };
    return;
  }

  let renderer;
  let scene;
  let camera;
  let graphRoot;
  let clock;
  let animationId = 0;
  let container;
  let raycaster;
  const pointerNdc = new THREE.Vector2();
  let nodeMeshes = [];
  let labelsList = [];
  let edgeSegments = [];
  let flowParticles = [];
  let lastLabelsKey = "";
  let spherical = { theta: 0.55, phi: 1.05, radius: 17 };
  let isDragging = false;
  let dragLast = { x: 0, y: 0 };
  let distortion = 0;
  let riskLevel = "UNKNOWN";
  let hasTelemetry = false;
  let resizeObserver;

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function stressFromScalars(drift, instability) {
    const d = typeof drift === "number" && Number.isFinite(drift) ? Math.max(0, Math.min(1, drift)) : 0;
    const ins = typeof instability === "number" && Number.isFinite(instability) ? Math.max(0, Math.min(1, instability)) : 0;
    return Math.max(0, Math.min(1, 0.55 * d + 0.45 * ins));
  }

  function paletteForRisk(risk, stress) {
    const r = String(risk || "UNKNOWN").toUpperCase();
    let calm = new THREE.Color(0x2dd4bf);
    let hot = new THREE.Color(0x38bdf8);
    if (r === "HIGH" || stress > 0.72) {
      calm = new THREE.Color(0xfbbf24);
      hot = new THREE.Color(0xef4444);
    } else if (r === "MEDIUM" || stress > 0.38) {
      calm = new THREE.Color(0x5eead4);
      hot = new THREE.Color(0xf59e0b);
    }
    return { calm, hot };
  }

  function fibonacciSphere(n, radius) {
    const pts = [];
    const g = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i += 1) {
      const y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0;
      const yr = Math.sqrt(Math.max(0, 1 - y * y));
      const th = g * i;
      pts.push(new THREE.Vector3(Math.cos(th) * yr * radius, y * radius, Math.sin(th) * yr * radius));
    }
    return pts;
  }

  function clearGraph() {
    if (!graphRoot) return;
    while (graphRoot.children.length) {
      const o = graphRoot.children[0];
      graphRoot.remove(o);
      if (o.geometry) o.geometry.dispose();
      const mat = o.material;
      if (mat) {
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose && m.dispose());
        else mat.dispose && mat.dispose();
      }
    }
    nodeMeshes = [];
    edgeSegments = [];
    flowParticles = [];
  }

  function makeEdgeTube(a, b, color, stress) {
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
    mid.normalize().multiplyScalar(a.length() * 0.92);
    const curve = new THREE.QuadraticBezierCurve3(a.clone(), mid, b.clone());
    const points = curve.getPoints(24);
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: lerp(0.55, 0.22, stress),
    });
    const line = new THREE.Line(geom, mat);
    line.userData = { curve, segments: points.length, a: a.clone(), b: b.clone(), stress };
    return line;
  }

  function addFlowParticle(curve, color, phase) {
    const g = new THREE.SphereGeometry(0.07, 10, 10);
    const m = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.85,
      transparent: true,
      opacity: 0.92,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.userData = { curve, t: phase, speed: 0.35 + Math.random() * 0.2 };
    graphRoot.add(mesh);
    flowParticles.push(mesh);
  }

  function buildIdle() {
    clearGraph();
    hasTelemetry = false;
    const n = 20;
    const pts = fibonacciSphere(n, 4.2);
    labelsList = Array.from({ length: n }, (_, i) => `signal_${i + 1}`);
    const { calm } = paletteForRisk("LOW", 0.08);
    for (let i = 0; i < n; i += 1) {
      const geo = new THREE.SphereGeometry(0.2, 20, 20);
      const mat = new THREE.MeshStandardMaterial({
        color: calm,
        emissive: calm,
        emissiveIntensity: 0.25,
        metalness: 0.2,
        roughness: 0.45,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pts[i]);
      mesh.userData = { index: i, base: pts[i].clone() };
      graphRoot.add(mesh);
      nodeMeshes.push(mesh);
    }
    for (let i = 0; i < n; i += 1) {
      const a = pts[i];
      const b = pts[(i + 1) % n];
      const line = makeEdgeTube(a, b, calm, 0.12);
      graphRoot.add(line);
      edgeSegments.push(line);
      const curve = line.userData.curve;
      addFlowParticle(curve, calm, i / n);
    }
  }

  function refreshNodeMaterials(stress, risk, labels, values) {
    const { calm, hot } = paletteForRisk(risk, stress);
    for (let i = 0; i < nodeMeshes.length; i += 1) {
      const mesh = nodeMeshes[i];
      const t = typeof values[i] === "number" && Number.isFinite(values[i]) ? values[i] : 0.5;
      const col = calm.clone().lerp(hot, t * 0.85);
      mesh.userData.activity = t;
      mesh.userData.label = labels[i] || mesh.userData.label;
      if (mesh.material) {
        mesh.material.color.copy(col);
        mesh.material.emissive.copy(col);
        const sel = mesh.userData.selected;
        mesh.material.emissiveIntensity = sel ? 0.95 : 0.2 + t * 0.55;
      }
      const r = lerp(0.16, 0.32, t);
      mesh.userData.baseScale = r / 0.2;
      mesh.scale.setScalar(mesh.userData.baseScale * (mesh.userData.selected ? 1.14 : 1));
    }
  }

  function buildFromData(payload) {
    const labels = Array.isArray(payload.labels) ? payload.labels.map((s) => String(s || "").trim()).filter(Boolean) : [];
    const values = Array.isArray(payload.values) ? payload.values : [];
    const key = labels.join("\0");
    const stress = stressFromScalars(payload.drift, payload.instability);
    if (key === lastLabelsKey && nodeMeshes.length === labels.length && labels.length > 0) {
      refreshNodeMaterials(stress, payload.risk, labels, values);
      return;
    }
    lastLabelsKey = key;
    clearGraph();
    hasTelemetry = true;
    if (labels.length < 2) {
      buildIdle();
      return;
    }
    const n = labels.length;
    labelsList = labels.slice();
    const pts = fibonacciSphere(n, 4.4);
    const { calm, hot } = paletteForRisk(payload.risk, stress);
    for (let i = 0; i < n; i += 1) {
      const t = typeof values[i] === "number" && Number.isFinite(values[i]) ? values[i] : 0.5;
      const col = calm.clone().lerp(hot, t * 0.85);
      const geo = new THREE.SphereGeometry(0.2, 20, 20);
      const mat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.2 + t * 0.55,
        metalness: 0.25,
        roughness: 0.42,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pts[i]);
      const r = lerp(0.16, 0.32, t);
      mesh.scale.setScalar(r / 0.2);
      mesh.userData = {
        index: i,
        base: pts[i].clone(),
        label: labels[i],
        activity: t,
        baseScale: r / 0.2,
        selected: false,
      };
      graphRoot.add(mesh);
      nodeMeshes.push(mesh);
    }
    const r = String(payload.risk || "UNKNOWN").toUpperCase();
    const criticalBreak = r === "HIGH" && stress > 0.58;
    for (let i = 0; i < n; i += 1) {
      if (criticalBreak && i % 4 === 2) continue;
      const a = pts[i];
      const b = pts[(i + 1) % n];
      const ccol = calm.clone().lerp(hot, stress);
      const line = makeEdgeTube(a, b, ccol, stress);
      graphRoot.add(line);
      edgeSegments.push(line);
      addFlowParticle(line.userData.curve, ccol, i / n);
    }
    if (n >= 6) {
      for (let i = 0; i < n; i += 1) {
        if (criticalBreak && i % 5 === 1) continue;
        const a = pts[i];
        const b = pts[(i + 2) % n];
        const ccol = calm.clone().lerp(hot, stress * 0.75);
        const line = makeEdgeTube(a, b, ccol, stress * 0.85);
        line.scale.set(0.92, 0.92, 0.92);
        graphRoot.add(line);
        edgeSegments.push(line);
        addFlowParticle(line.userData.curve, ccol, (i + 0.5) / n);
      }
    }
  }

  function updateCamera() {
    spherical.phi = Math.max(0.18, Math.min(Math.PI - 0.18, spherical.phi));
    const rad = spherical.radius;
    camera.position.x = rad * Math.sin(spherical.phi) * Math.cos(spherical.theta);
    camera.position.y = rad * Math.cos(spherical.phi);
    camera.position.z = rad * Math.sin(spherical.phi) * Math.sin(spherical.theta);
    camera.lookAt(0, 0, 0);
  }

  function onPointerDown(ev) {
    isDragging = true;
    dragLast = { x: ev.clientX, y: ev.clientY };
  }

  function onPointerUp() {
    isDragging = false;
  }

  function onPointerMove(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointerNdc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    if (isDragging) {
      const dx = ev.clientX - dragLast.x;
      const dy = ev.clientY - dragLast.y;
      dragLast = { x: ev.clientX, y: ev.clientY };
      spherical.theta += dx * 0.0055;
      spherical.phi += dy * 0.0045;
    }
  }

  function onWheel(ev) {
    ev.preventDefault();
    spherical.radius = Math.max(8, Math.min(32, spherical.radius + ev.deltaY * 0.012));
  }

  function pickNode() {
    raycaster.setFromCamera(pointerNdc, camera);
    const hits = raycaster.intersectObjects(nodeMeshes, false);
    return hits.length ? hits[0].object : null;
  }

  function onClick() {
    const m = pickNode();
    if (!m || !m.userData) return;
    const idx = m.userData.index;
    for (let i = 0; i < nodeMeshes.length; i += 1) {
      const mesh = nodeMeshes[i];
      const sel = i === idx;
      mesh.userData.selected = sel;
      const base = typeof mesh.userData.baseScale === "number" ? mesh.userData.baseScale : 1;
      mesh.scale.setScalar(base * (sel ? 1.14 : 1));
      if (mesh.material && mesh.material.emissiveIntensity !== undefined) {
        mesh.material.emissiveIntensity = sel ? 0.95 : 0.2 + (mesh.userData.activity || 0.5) * 0.55;
      }
    }
  }

  function updateTooltip(ev, mesh) {
    const tip = document.getElementById("dashboardSystemMap3dTooltip");
    if (!tip) return;
    if (!mesh || !mesh.userData) {
      tip.classList.add("hidden");
      tip.textContent = "";
      return;
    }
    const label = mesh.userData.label || labelsList[mesh.userData.index] || "sensor";
    const act =
      typeof mesh.userData.activity === "number" ? ` · activity ${(mesh.userData.activity * 100).toFixed(0)}%` : "";
    tip.textContent = `${label}${act}`;
    tip.classList.remove("hidden");
    tip.style.left = `${ev.clientX + 12}px`;
    tip.style.top = `${ev.clientY + 12}px`;
  }

  function animate() {
    animationId = global.requestAnimationFrame(animate);
    if (!clock || !renderer || !scene || !camera) return;
    const t = clock.getElapsedTime();
    const stress = distortion;
    const jitter = hasTelemetry ? stress * 0.42 : 0.06;
    const breathe = 0.04 + 0.03 * Math.sin(t * 0.7);
    graphRoot.rotation.y = t * (hasTelemetry ? 0.05 : 0.11);

    for (let i = 0; i < nodeMeshes.length; i += 1) {
      const mesh = nodeMeshes[i];
      const base = mesh.userData.base;
      if (!base) continue;
      const jx = jitter * Math.sin(t * 2.2 + i * 0.7);
      const jy = jitter * Math.cos(t * 1.9 + i * 0.5);
      const jz = jitter * Math.sin(t * 2.05 + i * 0.35);
      mesh.position.set(base.x + jx, base.y + jy, base.z + jz);
      const pulse = breathe * (0.5 + (mesh.userData.activity || 0.5));
      if (mesh.material && mesh.material.emissiveIntensity !== undefined && !mesh.userData.selected) {
        mesh.material.emissiveIntensity = 0.18 + pulse + stress * 0.25;
      }
    }

    for (let i = 0; i < edgeSegments.length; i += 1) {
      const line = edgeSegments[i];
      const ud = line.userData;
      if (!ud || !ud.curve || !line.geometry.attributes.position) continue;
      const pos = line.geometry.attributes.position;
      const arr = pos.array;
      const segs = ud.segments || 24;
      const amp = stress * 0.55 + (riskLevel === "HIGH" ? 0.25 : 0);
      for (let j = 0; j < segs; j += 1) {
        const u = segs > 1 ? j / (segs - 1) : 0;
        const orig = ud.curve.getPoint(u);
        const jx = amp * Math.sin(t * 3 + j * 0.4 + i);
        const jy = amp * Math.cos(t * 2.7 + j * 0.35 + i * 0.2);
        const jz = amp * Math.sin(t * 2.9 + j * 0.3);
        arr[j * 3] = orig.x + jx;
        arr[j * 3 + 1] = orig.y + jy;
        arr[j * 3 + 2] = orig.z + jz;
      }
      pos.needsUpdate = true;
      if (line.material) {
        line.material.opacity = lerp(0.52, 0.12, Math.min(1, stress + (riskLevel === "HIGH" ? 0.35 : 0)));
      }
    }

    for (let i = 0; i < flowParticles.length; i += 1) {
      const fp = flowParticles[i];
      const c = fp.userData.curve;
      if (!c) continue;
      fp.userData.t = (fp.userData.t + fp.userData.speed * 0.008 * (1 + stress)) % 1;
      const pos = c.getPoint(fp.userData.t);
      const wobble = stress * 0.35;
      fp.position.set(
        pos.x + wobble * Math.sin(t * 4 + i),
        pos.y + wobble * Math.cos(t * 3.5 + i),
        pos.z + wobble * Math.sin(t * 3.8 + i * 0.5),
      );
    }

    raycaster.setFromCamera(pointerNdc, camera);
    const hits = raycaster.intersectObjects(nodeMeshes, false);
    const over = hits.length ? hits[0].object : null;
    for (let i = 0; i < nodeMeshes.length; i += 1) {
      const mesh = nodeMeshes[i];
      const em = mesh.material && mesh.material.emissiveIntensity !== undefined;
      if (em && !mesh.userData.selected) {
        mesh.material.emissiveIntensity = mesh === over ? 1.05 : 0.18 + (mesh.userData.activity || 0.5) * 0.55 + stress * 0.2;
      }
    }

    updateCamera();
    renderer.render(scene, camera);
  }

  function resize() {
    if (!container || !renderer || !camera) return;
    const w = container.clientWidth;
    const h = container.clientHeight || 280;
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function init(containerSelector) {
    container =
      typeof containerSelector === "string"
        ? document.querySelector(containerSelector)
        : containerSelector;
    if (!container) return;

    clock = new THREE.Clock();
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    scene.fog = new THREE.Fog(0x070b14, 14, 48);

    camera = new THREE.PerspectiveCamera(48, 1, 0.1, 200);
    graphRoot = new THREE.Group();
    scene.add(graphRoot);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const amb = new THREE.AmbientLight(0x8899aa, 0.55);
    scene.add(amb);
    const d1 = new THREE.DirectionalLight(0xc7e2ff, 0.55);
    d1.position.set(6, 10, 8);
    scene.add(d1);
    const d2 = new THREE.DirectionalLight(0x2dd4bf, 0.35);
    d2.position.set(-8, -4, -6);
    scene.add(d2);

    raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.12 };

    resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(container);
    resize();

    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointermove", (ev) => {
      onPointerMove(ev);
      const m = pickNode();
      updateTooltip(ev, m);
    });
    canvas.addEventListener("pointerleave", () => {
      isDragging = false;
      const tip = document.getElementById("dashboardSystemMap3dTooltip");
      if (tip) tip.classList.add("hidden");
    });
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    buildIdle();
    animate();
  }

  function update(payload) {
    if (!graphRoot) return;
    const p = payload || {};
    distortion = stressFromScalars(p.drift, p.instability);
    riskLevel = String(p.risk || "UNKNOWN").toUpperCase();
    const labels = Array.isArray(p.labels) ? p.labels : [];
    const clean = labels.map((s) => String(s || "").trim()).filter(Boolean);
    if (clean.length < 2) {
      if (lastLabelsKey !== "__idle__") {
        lastLabelsKey = "__idle__";
        buildIdle();
      }
      return;
    }
    buildFromData({
      labels: clean,
      values: Array.isArray(p.values) ? p.values : [],
      drift: p.drift,
      instability: p.instability,
      risk: riskLevel,
    });
  }

  function dispose() {
    global.cancelAnimationFrame(animationId);
    if (resizeObserver && container) resizeObserver.unobserve(container);
    resizeObserver = null;
    clearGraph();
    if (renderer) {
      renderer.dispose();
      renderer = null;
    }
    scene = null;
    camera = null;
    graphRoot = null;
    clock = null;
    container = null;
  }

  global.NeraiumSensorGraph3D = {
    init,
    update,
    dispose,
  };
})(typeof window !== "undefined" ? window : globalThis);
