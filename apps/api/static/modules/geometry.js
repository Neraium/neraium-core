function geometryFlowPerfEnabled() {
  try {
    const v = String(window.localStorage.getItem(GEOMETRY_FLOW_PERF_KEY) ?? "1").toLowerCase();
    return v !== "0" && v !== "false" && v !== "off";
  } catch (_e) {
    return true;
  }
}

function geometryFlowReducedMotion() {
  try {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (_e) {
    return false;
  }
}

function geometryFlowDebugEnabled() {
  try {
    return new URLSearchParams(window.location.search).get("geomDebug") === "1";
  } catch (_e) {
    return false;
  }
}

function geometryFlow2dOnlyFromUrl() {
  try {
    return String(new URLSearchParams(window.location.search).get("flow2d") || "").trim() === "1";
  } catch (_e) {
    return false;
  }
}

function geometryFlowPrefer3dGraph() {
  /** Opt-in legacy Three.js node–link graph (?flow3d=1). Default is 2D bounded-fluid view. */
  try {
    return new URLSearchParams(window.location.search).get("flow3d") === "1";
  } catch (_e) {
    return false;
  }
}

function geometryFlowPrefer2dOnly() {
  if (geometryFlow2dOnlyFromUrl()) return true;
  return !geometryFlowPrefer3dGraph();
}

/**
 * Opt-in feature toggles for experiments (localStorage `neraium_feat_<name>=1` or `?feat_<name>=1`).
 */
function neraiumFeatureEnabled(name) {
  try {
    if (new URLSearchParams(window.location.search).get(`feat_${name}`) === "1") return true;
    return String(window.localStorage.getItem(`neraium_feat_${name}`) || "")
      .trim()
      .toLowerCase() === "1";
  } catch (_e) {
    return false;
  }
}

function neraiumClientErrorReportingDisabled() {
  try {
    if (new URLSearchParams(window.location.search).get("noClientLog") === "1") return true;
    return window.localStorage.getItem("neraium_disable_client_errors") === "1";
  } catch (_e) {
    return false;
  }
}

function scheduleHeavyWork(fn) {
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(fn, { timeout: 320 });
  } else {
    window.setTimeout(fn, 1);
  }
}

/** Prefer rAF for geometry scene build so the 3D view appears promptly (idle can delay hundreds of ms). */
function scheduleGeometrySceneBuild(fn) {
  if (typeof window.requestAnimationFrame !== "function") {
    window.setTimeout(fn, 0);
    return;
  }
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(fn);
  });
}

/**
 * Waits for `three-init.mjs` (loaded via `<script type="module" src="/web/three-init.mjs">`) to set
 * `window.__THREE_ESM`. We intentionally do not use a second `import("/web/...")` from this classic
 * script: duplicate dynamic imports can throw "Failed to fetch" under some CSP or timing setups.
 */
function ensureThreeModulesLoaded() {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Three.js is only available in the browser."));
  }
  if (window.__THREE_ESM) {
    return Promise.resolve();
  }
  if (!window.__THREE_ESM_LOADING) {
    window.__THREE_ESM_LOADING = true;
    const script = document.createElement("script");
    script.type = "module";
    script.src = "/web/three-init.mjs";
    script.async = true;
    document.head.appendChild(script);
  }
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 15000;
    const step = () => {
      if (window.__THREE_ESM) {
        resolve();
        return;
      }
      if (Date.now() > deadline) {
        reject(
          new Error(
            "Three.js did not load (on-demand module fetch failed). Check DevTools Console / Network for cdn.jsdelivr.net/npm/three and /web/three-init.mjs. Offline/air-gapped: run `python scripts/download_three_vendor.py` and switch index.html import map back to /web/vendor paths.",
          ),
        );
        return;
      }
      requestAnimationFrame(step);
    };
    step();
  });
}

function setGeometryViewportLoading(on) {
  const viewport = qs("#geometryViewport");
  const canvasWrap = qs("#geometryCanvasWrap");
  if (viewport) viewport.classList.toggle("geometry-viewport--loading", Boolean(on));
  if (canvasWrap) canvasWrap.classList.toggle("geometry-canvas-wrap--loading", Boolean(on));
}

let _lastClientErrorFingerprint = "";
let _lastClientErrorTime = 0;

function reportClientErrorToServer(payload) {
  if (neraiumClientErrorReportingDisabled()) return;
  const body = JSON.stringify(payload);
  const fp = body.slice(0, 280);
  const now = Date.now();
  if (fp === _lastClientErrorFingerprint && now - _lastClientErrorTime < 8000) return;
  _lastClientErrorFingerprint = fp;
  _lastClientErrorTime = now;
  const url = apiUrl("/client-errors");
  try {
    if (typeof navigator.sendBeacon === "function") {
      const blob = new Blob([body], { type: "application/json" });
      if (navigator.sendBeacon(url, blob)) return;
    }
  } catch (_e) {
    // fall through to fetch
  }
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {});
}

function initClientErrorReporting() {
  if (neraiumClientErrorReportingDisabled()) return;
  window.addEventListener(
    "error",
    (ev) => {
      reportClientErrorToServer({
        message: String(ev.message || "error"),
        stack: ev.error && ev.error.stack ? String(ev.error.stack) : "",
        url: String(window.location.href || ""),
        source: ev.filename != null ? String(ev.filename) : "",
        lineno: ev.lineno != null ? Number(ev.lineno) : null,
        colno: ev.colno != null ? Number(ev.colno) : null,
      });
    },
    true
  );
  window.addEventListener("unhandledrejection", (ev) => {
    const reason = ev.reason;
    const msg = reason instanceof Error ? reason.message : String(reason || "rejection");
    const stack = reason instanceof Error && reason.stack ? String(reason.stack) : "";
    reportClientErrorToServer({
      message: msg,
      stack,
      url: String(window.location.href || ""),
      reason: "unhandledrejection",
    });
  });
}

function customerIdValue(value) {
  const text = String(value || "").trim();
  return text || "default-customer";
}

function siteIdValue(value) {
  return String(value || "").trim();
}

function readTenantFromStorage() {
  try {
    const stored = window.localStorage.getItem(TENANT_STORAGE_KEY);
    state.tenant.customerId = customerIdValue(stored);
  } catch (_err) {
    state.tenant.customerId = "default-customer";
  }
}

function readDemoModeFromStorage() {
  try {
    const raw = String(window.localStorage.getItem(DEMO_MODE_STORAGE_KEY) || "").trim().toLowerCase();
    state.demo.enabled = raw === "1" || raw === "true" || raw === "on";
  } catch (_err) {
    state.demo.enabled = false;
  }
}

/** ?replay=1 (and legacy ?demo=1) enables replay mode for share links. Returns whether to auto-prepare runs. */
function applyDemoQueryParams() {
  try {
    const params = new URLSearchParams(window.location.search);
    const d = params.get("replay") || params.get("share_replay") || params.get("demo") || params.get("share_demo");
    if (d === "1" || d === "true" || d === "yes") {
      state.demo.enabled = true;
      persistDemoMode();
    }
    const prep = params.get("prepare") || params.get("autorun");
    if (prep === "1" || prep === "true" || prep === "yes") {
      state.demo.enabled = true;
      persistDemoMode();
    }
    const shouldAutoPrepare =
      prep === "1" || prep === "true" || prep === "yes";
    return { shouldAutoPrepare };
  } catch (_err) {
    return { shouldAutoPrepare: false };
  }
}

function applyDemoUiShell() {
  try {
    document.body.classList.toggle("demo-mode-active", !!state.demo.enabled);
  } catch (_e) {
    // no-op
  }
}

function persistDemoMode() {
  try {
    window.localStorage.setItem(DEMO_MODE_STORAGE_KEY, state.demo.enabled ? "1" : "0");
  } catch (_err) {
    // no-op
  }
}

function tenantScopeParams(extra = {}) {
  const out = {
    customer_id: customerIdValue(state.tenant.customerId),
  };
  const siteId = siteIdValue(state.tenant.siteId);
  if (siteId) {
    out.site_id = siteId;
  }
  return { ...out, ...extra };
}

function routeScopeFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const out = {};
  const customer = params.get("customer_id");
  const site = params.get("site_id");
  if (customer) out.customer_id = customerIdValue(customer);
  if (site) out.site_id = siteIdValue(site);
  return out;
}

function collectKnownSites(results) {
  const known = new Set(state.tenant.knownSites);
  (results || []).forEach((row) => {
    const site = siteIdValue(row?.site_id);
    if (site) known.add(site);
  });
  state.tenant.knownSites = Array.from(known).sort((a, b) => a.localeCompare(b));
}

function renderTenantControls() {
  const customerInput = qs("#customerFilterInput");
  const siteInput = qs("#siteFilterInput");
  const siteList = qs("#knownSitesList");
  if (customerInput) customerInput.value = customerIdValue(state.tenant.customerId);
  if (siteInput) siteInput.value = siteIdValue(state.tenant.siteId);
  const seedDemoBtn = qs("#seedDemoBtn");
  if (seedDemoBtn) {
    seedDemoBtn.disabled = state.demo.preparing;
    seedDemoBtn.textContent = state.demo.preparing ? "Preparing reference replay…" : "Run NASA CMAPSS FD004 Reference Replay";
  }
  if (siteList) {
    siteList.innerHTML = state.tenant.knownSites
      .map((site) => `<option value="${escapeHtml(site)}"></option>`)
      .join("");
  }
  applyDemoUiShell();
  updateUploadRunInfo();
  renderDashboardDemoHero();
}

function setDemoButtonsDisabled(disabled) {
  [
    "#seedDemoBtn",
    "#dashboardQuickDemoBtn",
    "#demoPlayPauseBtn",
    "#demoReplayBtn",
  ].forEach((selector) => {
    const el = qs(selector);
    if (el) el.disabled = !!disabled;
  });
}

async function applyTenantFromControls() {
  const customerInput = qs("#customerFilterInput");
  const siteInput = qs("#siteFilterInput");
  const customer = customerIdValue(customerInput?.value);
  const site = siteIdValue(siteInput?.value);
  state.tenant.customerId = customer;
  state.tenant.siteId = site;
  try {
    window.localStorage.setItem(TENANT_STORAGE_KEY, customer);
  } catch (_err) {
    // no-op
  }
  await refreshCurrentPage();
}

const chartTheme = {
  tickColor: "#9cb0ce",
  gridColor: "rgba(153, 178, 217, 0.12)",
  legendColor: "#d8e5ff",
  tooltipBg: "rgba(6, 12, 23, 0.92)",
  tooltipBorder: "#3d5786",
};

function buildTrendChartOptions() {
  return {
    animation: {
      duration: 260,
      easing: "easeOutCubic",
    },
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 6, right: 10, bottom: 32, left: 4 },
    },
    interaction: {
      mode: "index",
      intersect: false,
    },
    scales: {
      x: {
        ticks: {
          color: chartTheme.tickColor,
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 8,
          padding: 6,
        },
        grid: { color: chartTheme.gridColor },
      },
      y: {
        ticks: { color: chartTheme.tickColor, padding: 6 },
        grid: { color: chartTheme.gridColor },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: chartTheme.legendColor,
          usePointStyle: true,
          pointStyle: "circle",
          boxHeight: 7,
        },
      },
      tooltip: {
        backgroundColor: chartTheme.tooltipBg,
        borderColor: chartTheme.tooltipBorder,
        borderWidth: 1,
        titleColor: "#ecf3ff",
        bodyColor: "#cfddf8",
        displayColors: true,
      },
    },
    elements: {
      point: {
        hoverRadius: 5,
        radius: 1.8,
      },
      line: {
        borderWidth: 2.4,
      },
    },
  };
}

function disposeGeometryRenderer() {
  const g = state.geometry3d;
  if (g.sensorDatumShell) {
    try {
      g.sensorDatumShell.geometry?.dispose?.();
      if (g.sensorDatumShell.material) {
        if (Array.isArray(g.sensorDatumShell.material)) g.sensorDatumShell.material.forEach((m) => m.dispose?.());
        else g.sensorDatumShell.material.dispose?.();
      }
    } catch (_e) {
      // no-op
    }
    g.sensorDatumShell = null;
  }
  if (g.structuralFieldMesh) {
    try {
      g.structuralFieldMesh.geometry?.dispose?.();
      if (g.structuralFieldMesh.material) {
        if (g.structuralFieldMesh.material.map) g.structuralFieldMesh.material.map.dispose?.();
        if (Array.isArray(g.structuralFieldMesh.material)) {
          g.structuralFieldMesh.material.forEach((m) => m.dispose?.());
        } else g.structuralFieldMesh.material.dispose?.();
      }
    } catch (_e) {
      // no-op
    }
    g.structuralFieldMesh = null;
  }
  if (g.groundMesh) {
    try {
      g.groundMesh.geometry?.dispose?.();
      if (g.groundMesh.material) {
        if (Array.isArray(g.groundMesh.material)) g.groundMesh.material.forEach((m) => m.dispose?.());
        else g.groundMesh.material.dispose?.();
      }
    } catch (_e) {
      // no-op
    }
    g.groundMesh = null;
  }
  g.keyLight = null;
  g.keyLightBasePos = null;
  g.geometryRig = null;
  g.reducedGeometryMotion = false;
  g.structuralFlowPlaneLayout = false;
  g.structuralFieldMesh = null;
  g.flowUseStructuralTubes = false;
  g.flowUseLine2 = false;
  if (g.flowInstancedMesh) {
    try {
      g.flowInstancedMesh.geometry?.dispose?.();
      if (g.flowInstancedMesh.material) {
        if (Array.isArray(g.flowInstancedMesh.material)) g.flowInstancedMesh.material.forEach((m) => m.dispose?.());
        else g.flowInstancedMesh.material.dispose?.();
      }
      if (g.edgeGroup) g.edgeGroup.remove(g.flowInstancedMesh);
    } catch (_e) {
      // no-op
    }
    g.flowInstancedMesh = null;
  }
  if (g.edgeRefs && g.edgeRefs.length) {
    g.edgeRefs.forEach((ref) => {
      try {
        if (ref.lineGeomMain) ref.lineGeomMain.dispose?.();
        if (ref.lineGeomGlow) ref.lineGeomGlow.dispose?.();
        if (ref.lineMatMain) ref.lineMatMain.dispose?.();
        if (ref.lineMatGlow) ref.lineMatGlow.dispose?.();
        if (ref.line2Main && ref.line2Main.parent) ref.line2Main.parent.remove(ref.line2Main);
        if (ref.line2Glow && ref.line2Glow.parent) ref.line2Glow.parent.remove(ref.line2Glow);
        if (ref.tubeGeom) {
          ref.tubeGeom.dispose?.();
          ref.tubeGeom = null;
        }
        if (ref.tubeMaterial) {
          ref.tubeMaterial.dispose?.();
          ref.tubeMaterial = null;
        }
        if (ref.tubeMesh && ref.tubeMesh.parent) {
          ref.tubeMesh.parent.remove(ref.tubeMesh);
        }
      } catch (_e) {
        // no-op
      }
      ref.tubeMesh = null;
      ref.line2Main = null;
      ref.line2Glow = null;
    });
  }
  g.flowDummy = null;
  if (g.scene && g.scene.environment) {
    try {
      g.scene.environment.dispose();
    } catch (_e) {
      // no-op
    }
    g.scene.environment = null;
  }
  if (g.frameId) {
    window.cancelAnimationFrame(g.frameId);
    g.frameId = null;
  }
  if (g.hoverRaf) {
    try {
      window.cancelAnimationFrame(g.hoverRaf);
    } catch (_e) {
      // no-op
    }
    g.hoverRaf = null;
  }
  g.pendingHoverEvt = null;
  if (g.iblDeferredRaf != null) {
    try {
      window.cancelAnimationFrame(g.iblDeferredRaf);
    } catch (_e) {
      // no-op
    }
    g.iblDeferredRaf = null;
  }
  Object.values(g.nodeLabelById || {}).forEach((spr) => {
    try {
      if (spr?.material?.map) spr.material.map.dispose?.();
      spr?.material?.dispose?.();
    } catch (_e) {
      // no-op
    }
  });
  if (g.resizeRaf) {
    try {
      window.cancelAnimationFrame(g.resizeRaf);
    } catch (_e) {
      // no-op
    }
    g.resizeRaf = null;
  }
  if (g.canvas2d && g.canvas2d.parentElement) {
    try {
      g.canvas2d.parentElement.removeChild(g.canvas2d);
    } catch (_e) {
      // no-op
    }
  }
  g.canvas2d = null;
  g.useCanvas2d = false;
  g.structuralFlowCoherence01 = null;
  g.structuralFlowDerived = null;
  g.temporalFlow = {
    points: [],
    localTimeSec: 0,
    replayHoldSec: 0,
    latestResultId: null,
    historyCentroid: [],
    historyDriftTip: [],
    historyContact: [],
    contactPersistence: 0,
    lastTrendCursor: -1,
  };
  setTrendPlaybackCursorMarker(-1);
  g.flowEdgeScratch = null;
  if (g.resizeObserver) {
    try {
      g.resizeObserver.disconnect();
    } catch (_err) {
      // no-op
    }
    g.resizeObserver = null;
  }
  if (typeof g.cleanupResize === "function") {
    g.cleanupResize();
    g.cleanupResize = null;
  }
  if (typeof g.cleanupPointer === "function") {
    g.cleanupPointer();
    g.cleanupPointer = null;
  }
  if (g.controls) {
    g.controls.dispose();
    g.controls = null;
  }
  g.geometryIntroStart = null;
  g.geometryIntroMs = 0;
  const _gfw = qs("#geometryCanvasWrap");
  if (_gfw) {
    _gfw.style.opacity = "";
  }
  if (g.renderer) {
    g.renderer.dispose();
    if (g.renderer.domElement && g.renderer.domElement.parentElement) {
      g.renderer.domElement.parentElement.removeChild(g.renderer.domElement);
    }
    g.renderer = null;
  }
  g.scene = null;
  g.camera = null;
  g.edgeGroup = null;
  g.edgeRefs = [];
  g.raycaster = null;
  g.pointer = null;
  g.nodeGroup = null;
  g.geometryRig = null;
  g.nodeMeshById = {};
  g.nodeDataById = {};
  g.nodeLabelById = {};
  g.nodeGlowById = {};
  g.unstablePulseById = {};
  g.selectedId = null;
  g.hoveredId = null;
  g.interactionEnabled = false;
  g._2dT = 0;
}

function getThreeNamespace() {
  return window.__THREE_ESM || window.THREE;
}

function getOrbitControlsConstructor() {
  if (typeof window.__OrbitControls === "function") {
    return window.__OrbitControls;
  }
  if (window.THREE && window.THREE.OrbitControls) {
    return window.THREE.OrbitControls;
  }
  if (window.OrbitControls) {
    return window.OrbitControls;
  }
  return null;
}

/** RoomEnvironment + PMREM for believable metal/plastic reflections (skipped in perf mode). */
function setupGeometryIBL(three, renderer, scene, perf) {
  if (perf || !renderer || !scene) return false;
  const PMREM = window.__PMREMGenerator;
  const RoomEnv = window.__RoomEnvironment;
  if (!PMREM || !RoomEnv) return false;
  try {
    const pmrem = new PMREM(renderer);
    const env = new RoomEnv();
    const rt = pmrem.fromScene(env, 0.04);
    scene.environment = rt.texture;
    pmrem.dispose();
    env.dispose();
    return true;
  } catch (_e) {
    return false;
  }
}

function updateGeometryShadowFrustum(g, three) {
  if (!g?.keyLight || !g?.nodeGroup || !three) return;
  const box = new three.Box3().setFromObject(g.nodeGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new three.Vector3());
  const size = box.getSize(new three.Vector3());
  const ext = Math.max(size.x, size.y, size.z, 0.2) * 1.35;
  const cam = g.keyLight.shadow.camera;
  cam.left = -ext;
  cam.right = ext;
  cam.top = ext;
  cam.bottom = -ext;
  cam.near = 0.05;
  cam.far = Math.max(10, ext * 7);
  g.keyLight.position.copy(center.clone().add(new three.Vector3(2.4, 4.4, 2.2)));
  g.keyLight.target.position.copy(center);
  cam.updateProjectionMatrix();
}

function geometryViewportIsNarrow() {
  try {
    return window.matchMedia && window.matchMedia("(max-width: 740px)").matches;
  } catch (_e) {
    return false;
  }
}

function fitGeometryCamera(camera, controls, nodeGroup, three, options) {
  if (!camera || !controls || !nodeGroup || !three) return;
  const box = new three.Box3().setFromObject(nodeGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new three.Vector3());
  const size = box.getSize(new three.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.08);
  const planeLayout = Boolean(options?.structuralFlowPlaneLayout);
  const tight = planeLayout ? 1.34 : 1.42;
  const dist = Math.max(0.92, maxDim * tight);
  controls.target.copy(center);
  camera.near = Math.max(0.006, dist / 520);
  camera.far = Math.max(80, dist * 48);
  camera.updateProjectionMatrix();
  const narrow = geometryViewportIsNarrow();
  const distMul = narrow ? 0.82 : 1.0;
  const offset = new three.Vector3(1.05, 0.62, 1.22).normalize().multiplyScalar(dist * distMul);
  camera.position.copy(center.clone().add(offset));
  camera.lookAt(center);
}


function geometryViewportDimensions() {
  const host = qs("#geometryViewport");
  if (!host) return null;
  const rect = host.getBoundingClientRect();
  const w = Math.floor(rect.width || host.clientWidth || 0);
  const h = Math.floor(rect.height || host.clientHeight || 0);
  return { host, width: w, height: h };
}

function ensureGeometryViewportReady(timeoutMs = 1600) {
  return new Promise((resolve) => {
    const start = window.performance.now();
    function tick() {
      const dims = geometryViewportDimensions();
      if (dims && dims.width > 40 && dims.height > 40) {
        resolve(dims);
        return;
      }
      if (window.performance.now() - start > timeoutMs) {
        resolve(dims);
        return;
      }
      window.requestAnimationFrame(tick);
    }
    tick();
  });
}

function setGeometrySurfaceState(message, level = "info") {
  const fallback = qs("#geometryFallback");
  const canvasWrap = qs("#geometryCanvasWrap");
  const viewport = qs("#geometryViewport");
  if (fallback) {
    fallback.textContent = String(message || "");
    const tone = level === "error" ? "error" : level === "warn" ? "warn" : "info";
    fallback.className = `empty-state geometry-fallback geometry-fallback--${tone}`;
    fallback.classList.remove("hidden");
  }
  if (canvasWrap) {
    canvasWrap.classList.remove("hidden");
  }
  if (viewport) {
    viewport.classList.remove("ready");
  }
}

function showGeometryCanvas() {
  const fallback = qs("#geometryFallback");
  const canvasWrap = qs("#geometryCanvasWrap");
  const viewport = qs("#geometryViewport");
  if (fallback) {
    fallback.classList.add("hidden");
    fallback.textContent = "";
  }
  if (viewport) {
    viewport.classList.add("ready");
  }
}

function riskColorHex(riskLevel) {
  const risk = normalizeRiskLevel(riskLevel);
  if (risk === "HIGH") return 0xe46060;
  if (risk === "MEDIUM") return 0xffbf56;
  if (risk === "LOW") return 0x68d497;
  return 0x8aa6cf;
}

function nodeStateColorHex(nodeState) {
  const s = String(nodeState || "").toLowerCase();
  if (s === "critical") return 0xff7a7a;
  if (s === "watch") return 0xffca72;
  return 0x7fdaac;
}

function edgeColorHex(edge, metrics) {
  const risk = normalizeRiskLevel(metrics?.risk_level);
  if (risk === "HIGH" && Math.abs(Number(edge.delta || 0)) > 0.18) {
    return 0xff8a8a;
  }
  if (risk === "MEDIUM" && Math.abs(Number(edge.delta || 0)) > 0.12) {
    return 0xffca72;
  }
  return edge.type === "negative" ? 0x8aa4d0 : 0x9ebcf0;
}

function flowDriftInstabilityNorm(payload) {
  const d = Number(payload?.metrics?.structural_drift_score);
  const i = Number(payload?.metrics?.composite_instability);
  const driftN = Number.isFinite(d) ? Math.max(0, Math.min(1, d)) : 0;
  const instN = Number.isFinite(i) ? Math.max(0, Math.min(1, i)) : 0;
  return { driftN, instN };
}

/**
 * Legacy drift/instability blend (0–1). Previously used as "structural coherence" but it
 * collapses toward 0 whenever drift and instability are both moderate, disagreeing with
 * tolerance counts. Kept only for optional visuals that need drift magnitude as deformation.
 */
function driftInstabilityVisualStress01(payload) {
  const { driftN, instN } = flowDriftInstabilityNorm(payload);
  return Math.max(0, Math.min(1, 0.5 * driftN + 0.5 * instN));
}

/** Primary coherence 0–1: fraction of sensors within relational tolerance (matches panel counts). */
function coherence01FromTolerance(totalSensors, withinTolerance) {
  const n = Math.max(0, Math.floor(Number(totalSensors) || 0));
  const w = Math.max(0, Math.floor(Number(withinTolerance) || 0));
  if (n <= 0) return 0;
  return Math.max(0, Math.min(1, w / n));
}

function flowNodeStress01(node) {
  const s = Number(node?.stress);
  return Number.isFinite(s) ? Math.max(0, Math.min(1, s)) : 0;
}

/** API `in_range` / heuristic: sensor within relational tolerance band. */
function flowNodeInRange(node) {
  if (node && typeof node.in_range === "boolean") return node.in_range;
  return flowNodeStress01(node) < 0.33 && !node?.is_unstable;
}

const flowNodeWithinTolerance = flowNodeInRange;

/** Triangle+hub (n=4), ring, or dual ring: shared XZ plane + vertical lift story. */
function structuralFlowUsesPlaneLayout(payload) {
  const l = payload?.projection?.layout;
  return (
    l === "diamond_plane_four_sensors" ||
    l === "triangle_center_plane_four_sensors" ||
    l === "coherent_plane_ring"
  );
}

/** Complete-graph pairwise link count: C(n,2). */
function structuralPairwiseLinkCount(sensorCount) {
  const n = Math.max(0, Math.floor(Number(sensorCount) || 0));
  return (n * (n - 1)) / 2;
}

/**
 * Aggregates all inputs for the structural-flow panel from the geometry payload + engine metrics.
 * Base coherence 0–1 = withinTolerance / totalSensors; the 2D canvas view may override headline % via field blend.
 */
function computeStructuralFlowMetrics(payload) {
  const nodes = payload?.nodes || [];
  const edges = payload?.edges || [];
  const totalSensors = nodes.length;
  const totalLinks = edges.length;
  const maxPairwiseLinks = structuralPairwiseLinkCount(totalSensors);
  let withinTolerance = 0;
  let outsideTolerance = 0;
  const summaryIn = payload?.summary?.in_range_nodes;
  const summaryOut = payload?.summary?.out_of_range_nodes;
  if (Number.isFinite(Number(summaryIn)) && Number.isFinite(Number(summaryOut))) {
    withinTolerance = Math.max(0, Number(summaryIn));
    outsideTolerance = Math.max(0, Number(summaryOut));
  } else {
    nodes.forEach((n) => {
      if (flowNodeWithinTolerance(n)) withinTolerance += 1;
      else outsideTolerance += 1;
    });
  }
  const driftedLinks = Number.isFinite(Number(payload?.summary?.changed_edges_current))
    ? Math.max(0, Number(payload.summary.changed_edges_current))
    : 0;
  const coherence01 = coherence01FromTolerance(totalSensors, withinTolerance);
  const { driftN, instN } = flowDriftInstabilityNorm(payload);
  return {
    totalSensors,
    totalLinks,
    maxPairwiseLinks,
    withinTolerance,
    outsideTolerance,
    driftedLinks,
    coherence01,
    driftN,
    instN,
    riskLevel: payload?.metrics?.risk_level,
    engineState: payload?.metrics?.state,
    planeLayout: structuralFlowUsesPlaneLayout(payload),
  };
}

/**
 * Structural headline for this view — derived from tolerance counts, coherence, and link drift.
 * Never labels the relationship view "stable" when coherence is collapsed and multiple sensors are outside tolerance.
 *
 * Threshold intent (coherence01 = withinTolerance / totalSensors):
 * - "High-risk pattern": ≥2 sensors outside tolerance AND coherence very low (<0.22).
 * - "Unstable": ≥2 outside + coherence <0.4, OR ≥1 outside + coherence <0.35.
 * - "Degrading": material drift on links or partial tolerance breach.
 * - "Aligned": none outside tolerance, no drifted links.
 */
function deriveStructuralAssessment(m) {
  const {
    coherence01,
    outsideTolerance,
    totalSensors,
    driftedLinks,
    totalLinks,
  } = m;
  const risk = normalizeRiskLevel(m.riskLevel);
  const driftRatio = totalLinks > 0 ? driftedLinks / totalLinks : 0;

  if (totalSensors >= 2 && outsideTolerance >= 2 && coherence01 < 0.22) {
    return {
      label: "High-risk pattern",
      detail:
        "Multiple sensors exceed relational tolerance while coherence (share within tolerance) is critically low.",
      tone: "HIGH",
    };
  }
  if (totalSensors >= 2 && outsideTolerance >= 2 && coherence01 < 0.4) {
    return {
      label: "Unstable",
      detail: "Multiple sensors outside relational tolerance with substantially reduced coherence.",
      tone: "HIGH",
    };
  }
  if (outsideTolerance >= 1 && coherence01 < 0.35) {
    return {
      label: "Unstable",
      detail: "At least one sensor outside tolerance with low structural coherence.",
      tone: "HIGH",
    };
  }
  if (totalLinks >= 3 && driftRatio >= 0.5) {
    return {
      label: "Degrading",
      detail: "A majority of displayed pairwise links exceed the drift threshold.",
      tone: risk === "HIGH" ? "HIGH" : "MEDIUM",
    };
  }
  if (outsideTolerance >= 1 || coherence01 < 0.45 || driftRatio >= 0.34) {
    return {
      label: "Degrading",
      detail: "Relational tolerance or pairwise drift signals are elevated versus baseline.",
      tone: "MEDIUM",
    };
  }
  if (outsideTolerance === 0 && driftedLinks === 0) {
    return {
      label: "Aligned",
      detail: "All sensors within relational tolerance and no links above drift threshold at this snapshot.",
      tone: "LOW",
    };
  }
  return {
    label: "Monitoring",
    detail: "Insufficient deviation to classify; continue monitoring.",
    tone: "LOW",
  };
}

/**
 * One technical paragraph aligned to computeStructuralFlowMetrics + deriveStructuralAssessment.
 */
function buildStructuralFlowNarrative(m, assessment, derived) {
  const n = m.totalSensors;
  const L = m.totalLinks;
  const Lmax = m.maxPairwiseLinks;
  const w = m.withinTolerance;
  const o = m.outsideTolerance;
  const d = m.driftedLinks;
  const pch = Math.round(m.coherence01 * 100);
  const sentences = [];
  if (n > 0) {
    sentences.push(
      `The projection shows ${n} sensor${n === 1 ? "" : "s"} and ${L} displayed pairwise link${L === 1 ? "" : "s"} (a complete graph on ${n} nodes has ${Lmax}).`,
    );
    sentences.push(`${w} read as inside the valid manifold; ${o} outside (relational tolerance).`);
  }
  if (L > 0) {
    sentences.push(`${d} of ${L} displayed link${L === 1 ? "" : "s"} exceed the pairwise drift threshold.`);
  } else if (n > 1) {
    sentences.push("No pairwise links are included in this projection.");
  }
  if (derived && !derived.isFallback) {
    sentences.push(
      `Integrity ${pch}% with field containment ${Math.round(derived.containmentScore * 100)}% and flow state “${derived.statusLabel}”.`,
    );
  } else {
    sentences.push(
      `Integrity score is ${pch}% (tolerance and links blended with field containment when the flow grid is active).`,
    );
  }
  sentences.push(`Flow state: ${assessment.label}.`);
  const engine = String(m.engineState || "").trim();
  if (engine && /stable|nominal|ok/i.test(engine) && (assessment.tone === "HIGH" || assessment.tone === "MEDIUM")) {
    sentences.push(
      `Engine phase is ${engine}; the relational view above is the authoritative read for pairwise tolerance and link drift.`,
    );
  }
  return sentences.join(" ");
}

/** Consistent encoding copy for the projection (Y = deviation magnitude, not decorative). */
function structuralProjectionCaption(payload) {
  const plane = structuralFlowUsesPlaneLayout(payload);
  const layout = geometryFlowPrefer3dGraph()
    ? plane
      ? "3D graph: horizontal plane (X/Z) is baseline relational layout; vertical (Y) is deviation magnitude."
      : "3D graph: axes encode relational layout; vertical (Y) encodes deviation magnitude."
    : "2D manifold: sensors drive a force field V(u,v); energy E=|V|² is sampled on a grid. The closed curve is an E=T isocontour (marching squares). Dashed box = valid relational state space; red/orange segments = contour crossing or escaping the box. Heatmap = distributed E (no radial glow). Yellow dot = E-weighted centroid; cyan arrow = E-weighted mean flow direction.";
  const base = geometryFlowPrefer3dGraph()
    ? `${layout} Node color encodes tolerance band. Link color and opacity encode pairwise drift versus baseline.`
    : `${layout} Sensor dots sit at projected (x,z); headline coherence = within-tolerance ÷ total.`;
  const ve = payload?.projection?.visual_expansion;
  if (ve && ve.enabled) {
    return `${base} Extra run channels were added for coverage; dimmer nodes are outside the engine’s correlation window (see selection detail).`;
  }
  return base;
}

function structuralMetricDefinitionsFootnote() {
  return (
    "Integrity Score: blended field read used by the visualization and panel. Containment: energy held inside the manifold core. " +
    "Drift: net flow magnitude and direction (arrow). Instability: composite breakup signal. Link Agreement: links under drift threshold. " +
    "Boundary Pressure: energy at the manifold edge. Inside/Outside Manifold: sensor counts."
  );
}

/**
 * Premium SII palette: calm teal/cyan when healthy, amber/gold for warning, orange/red for critical.
 * `node` optional — uses `state` when present for tiering.
 */
function flowNodeColorTHREE(three, stress01, unstable, node) {
  const c = new three.Color();
  const deepTeal = new three.Color(0x0d9488);
  const aqua = new three.Color(0x2dd4bf);
  const ice = new three.Color(0x38bdf8);
  const amber = new three.Color(0xd97706);
  const gold = new three.Color(0xeab308);
  const orange = new three.Color(0xea580c);
  const red = new three.Color(0xdc2626);
  const stateStr = String(node?.state || "").toLowerCase();
  if (unstable || stateStr === "critical") {
    c.copy(orange).lerp(red, Math.min(1, stress01 * 0.75 + 0.22));
    if (node && node.in_correlation_window === false) c.lerp(new three.Color(0x64748b), 0.12);
    return c;
  }
  if (stateStr === "watch" || stress01 > 0.58) {
    c.copy(amber).lerp(gold, Math.min(1, (stress01 - 0.35) / 0.65));
    if (node && node.in_correlation_window === false) c.lerp(new three.Color(0x64748b), 0.16);
    return c;
  }
  if (stress01 < 0.48) {
    c.copy(deepTeal).lerp(aqua, stress01 / 0.48);
    if (node && node.in_correlation_window === false) c.lerp(new three.Color(0x64748b), 0.16);
    return c;
  }
  c.copy(aqua).lerp(ice, Math.min(1, (stress01 - 0.48) / 0.52));
  if (node && node.in_correlation_window === false) c.lerp(new three.Color(0x64748b), 0.16);
  return c;
}

function flowNodeColorCss(stress01, unstable, node) {
  const three = getThreeNamespace();
  if (!three?.Color) return "#38bdf8";
  const c = flowNodeColorTHREE(three, stress01, unstable, node);
  return `#${c.getHexString()}`;
}

function flowEdgeTension(edge) {
  const mag = Math.abs(Number(edge.magnitude || 0));
  const delta = Math.abs(Number(edge.delta || 0));
  return Math.max(0, Math.min(1, 0.4 * (1 - mag) + 0.6 * Math.min(1, delta * 3)));
}

/** 1 = healthy shared flow, 0 = broken / stressed link. */
function flowEdgeHealth01(verticalSpan, tension, coherence) {
  const span = Math.min(1, verticalSpan * 4);
  const t = Math.max(0, Math.min(1, tension));
  const c = typeof coherence === "number" ? coherence : 0.75;
  return Math.max(0, Math.min(1, c * (1 - span * 0.85) * (1 - t * 0.55)));
}

function createStructuralFlowParticles(three, particleCount, perf) {
  const n = Math.max(0, Math.floor(particleCount));
  if (n <= 0 || !three.InstancedMesh) return null;
  const r = perf ? 0.012 : 0.016;
  const geom = new three.SphereGeometry(r, perf ? 6 : 8, perf ? 6 : 8);
  const mat = new three.MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
    transparent: true,
    opacity: perf ? 0.82 : 0.9,
    blending: three.AdditiveBlending,
    depthWrite: false,
  });
  const mesh = new three.InstancedMesh(geom, mat, n);
  if (mesh.instanceMatrix && mesh.instanceMatrix.setUsage) {
    mesh.instanceMatrix.setUsage(three.DynamicDrawUsage);
  }
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  mesh.userData.pickMesh = false;
  return mesh;
}

function getFlowLine2Helpers() {
  return {
    Line2: typeof window !== "undefined" ? window.__Line2 : null,
    LineGeometry: typeof window !== "undefined" ? window.__LineGeometry : null,
    LineMaterial: typeof window !== "undefined" ? window.__LineMaterial : null,
  };
}

function applyFlowLineResolutionToEdges(g, width, height) {
  const w = Math.max(1, Math.floor(width || 1));
  const h = Math.max(1, Math.floor(height || 1));
  (g.edgeRefs || []).forEach((ref) => {
    if (ref.lineMatMain?.resolution) ref.lineMatMain.resolution.set(w, h);
    if (ref.lineMatGlow?.resolution) ref.lineMatGlow.resolution.set(w, h);
  });
}

/**
 * Slow-rotating faceted datum shell — reads as calibrated measurement volume, not decoration.
 */
function createSensorDatumShell(three, boxNg, perf) {
  if (!three?.IcosahedronGeometry || !boxNg || boxNg.isEmpty()) return null;
  const size = boxNg.getSize(new three.Vector3());
  const ext = Math.max(size.x, size.y, size.z, 0.12);
  const r = ext * 1.42;
  const detail = perf ? 1 : 2;
  const ico = new three.IcosahedronGeometry(r, detail);
  const edges = new three.EdgesGeometry(ico);
  ico.dispose?.();
  const mat = new three.LineBasicMaterial({
    color: 0x153a4a,
    transparent: true,
    opacity: perf ? 0.11 : 0.18,
    depthWrite: false,
  });
  const shell = new three.LineSegments(edges, mat);
  shell.userData.pickMesh = false;
  shell.renderOrder = -5;
  shell.name = "sensorDatumShell";
  return shell;
}

/**
 * Instrument-style sensor: faceted shell + emissive core + wire cage + equatorial ring (not a generic sphere).
 */
function createStructuralSensorAssembly(three, opts) {
  const {
    nodeId,
    col,
    radius,
    stress01,
    inRange,
    perf,
  } = opts;
  const root = new three.Group();
  root.name = "structuralSensor";
  root.userData.nodeId = String(nodeId);
  root.userData.radius = radius;

  const shellDetail = perf ? 0 : 1;
  const shellGeom = new three.IcosahedronGeometry(radius, shellDetail);
  let shellMat;
  if (perf) {
    shellMat = new three.MeshLambertMaterial({
      color: col.clone(),
      emissive: col.clone().multiplyScalar(0.07),
      emissiveIntensity: 0.26,
    });
  } else {
    shellMat = new three.MeshPhysicalMaterial({
      color: col.clone(),
      metalness: 0.48,
      roughness: 0.22,
      clearcoat: 0.82,
      clearcoatRoughness: 0.14,
      envMapIntensity: 1.05,
      emissive: col.clone().multiplyScalar(0.05),
      emissiveIntensity: 0.34,
    });
  }
  const shell = new three.Mesh(shellGeom, shellMat);
  shell.userData.nodeId = String(nodeId);
  shell.userData.pickMesh = true;
  shell.userData.flowColor = col.clone();
  shell.userData.emissiveMul = perf ? 0.07 : 0.05;
  shell.userData.emissiveIntensityBase = perf ? 0.26 : 0.34;
  root.add(shell);

  const innerR = radius * 0.36;
  const innerGeom = new three.SphereGeometry(innerR, perf ? 6 : 10, perf ? 6 : 10);
  let innerMat;
  if (perf) {
    innerMat = new three.MeshBasicMaterial({
      color: col.clone(),
      transparent: true,
      opacity: 0.88,
    });
  } else {
    innerMat = new three.MeshStandardMaterial({
      color: 0x040a10,
      metalness: 0.15,
      roughness: 0.35,
      emissive: col.clone(),
      emissiveIntensity: 0.42 + stress01 * 0.45 * (inRange ? 0.35 : 1),
      envMapIntensity: 0.55,
    });
  }
  const inner = new three.Mesh(innerGeom, innerMat);
  inner.userData.pickMesh = false;
  root.add(inner);

  const cageIco = new three.IcosahedronGeometry(radius * 1.06, shellDetail);
  const cageGeom = new three.EdgesGeometry(cageIco);
  cageIco.dispose?.();
  const cageMat = new three.LineBasicMaterial({
    color: col.clone(),
    transparent: true,
    opacity: perf ? 0.32 : 0.5,
    depthWrite: false,
  });
  const cage = new three.LineSegments(cageGeom, cageMat);
  cage.userData.pickMesh = false;
  root.add(cage);

  const torusR = radius * 1.1;
  const tubeT = radius * 0.062;
  const ringGeom = new three.TorusGeometry(torusR, tubeT, perf ? 6 : 10, perf ? 28 : 56);
  const ringMat = new three.MeshBasicMaterial({
    color: col.clone(),
    transparent: true,
    opacity: perf ? 0.2 : 0.34,
    blending: three.AdditiveBlending,
    depthWrite: false,
  });
  const ring = new three.Mesh(ringGeom, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.userData.pickMesh = false;
  root.add(ring);

  root.userData.shellMesh = shell;
  root.userData.innerCore = inner;
  root.userData.sensorCage = cage;
  root.userData.sensorRing = ring;
  root.userData.haloMesh = null;

  return root;
}

function ensureFlowEdgeScratch(g, three) {
  if (!g.flowEdgeScratch) {
    const v0 = new three.Vector3();
    const v1 = new three.Vector3();
    const v2 = new three.Vector3();
    g.flowEdgeScratch = {
      curve: new three.QuadraticBezierCurve3(v0, v1, v2),
      mid: new three.Vector3(),
      dir: new three.Vector3(),
      perp: new three.Vector3(),
      up: new three.Vector3(0, 1, 0),
      altUp: new three.Vector3(1, 0, 0),
      tmpPoint: new three.Vector3(),
      flowColorA: new three.Color(),
      flowColorB: new three.Color(),
      edgeColHealthy: new three.Color(0x2dd4bf),
      edgeColHealthy2: new three.Color(0x38bdf8),
      edgeColStress: new three.Color(0xd97706),
      edgeColStress2: new three.Color(0xf97316),
    };
  }
  return g.flowEdgeScratch;
}

function syncFlowEdgesFromMeshes(g, three, t) {
  if (!g?.edgeRefs?.length || !three?.QuadraticBezierCurve3) return;
  const coherence = typeof g.flowCoherence === "number" ? g.flowCoherence : 0.75;
  const driftN = typeof g.flowDriftN === "number" ? g.flowDriftN : 0;
  const instN = typeof g.flowInstN === "number" ? g.flowInstN : 0;
  const frayBase = 0.08 + driftN * 0.35 + instN * 0.35;
  const perf = Boolean(g.perfMode);
  const planeLayout = Boolean(g.structuralFlowPlaneLayout);
  const useTubes = Boolean(g.flowUseStructuralTubes && three.TubeGeometry && !g.flowUseLine2);
  const S = ensureFlowEdgeScratch(g, three);
  const framePhase = Math.floor(t * 30);
  const pep = Math.max(1, Math.floor(g.flowParticlesPerEdge || 3));
  const flowMesh = g.flowInstancedMesh;
  const dummy = g.flowDummy;
  const colA = S.flowColorA;
  const colB = S.flowColorB;
  const colHealthyA = S.edgeColHealthy;
  const colHealthyB = S.edgeColHealthy2;
  const colStressA = S.edgeColStress;
  const colStressB = S.edgeColStress2;

  g.edgeRefs.forEach((ref, index) => {
    const { line, lineGlow, sourceId, targetId, edge } = ref;
    const sm = g.nodeMeshById[sourceId];
    const tm = g.nodeMeshById[targetId];
    if (!sm || !tm) return;
    const useL2 = Boolean(g.flowUseLine2 && ref.lineGeomMain);
    if (!useL2 && !line?.geometry?.attributes?.position) return;
    const posA = sm.position;
    const posB = tm.position;
    const tension = flowEdgeTension(edge);
    const verticalSpan = Math.abs(posA.y - posB.y);
    const health = flowEdgeHealth01(verticalSpan, tension, coherence);
    const stress01 = 1 - health;
    const mid = S.mid;
    mid.copy(posA).add(posB).multiplyScalar(0.5);
    S.dir.subVectors(posB, posA);
    if (S.dir.lengthSq() < 1e-10) return;
    S.perp.crossVectors(S.dir, S.up);
    if (S.perp.lengthSq() < 1e-10) {
      S.perp.crossVectors(S.dir, S.altUp);
    }
    S.perp.normalize();
    const lift = planeLayout
      ? 0.07 * verticalSpan + 0.014 * coherence * (1 - tension * 0.5)
      : 0.055 * coherence * (1 - tension * 0.88);
    mid.y += lift;
    let fray = frayBase * (0.35 + tension * 0.65) * (1 - coherence * 0.75);
    if (planeLayout) {
      fray *= Math.max(0.12, 1 - Math.min(1, verticalSpan * 2.2));
    }
    const frayPulse = stress01 * (0.55 + 0.45 * Math.sin(t * 6.2 + index * 1.4));
    const arcSign = index % 2 === 0 ? 1 : -1;
    mid.addScaledVector(S.perp, arcSign * 0.018 * (0.35 + coherence * 0.65));
    mid.addScaledVector(S.perp, fray * Math.sin(t * 2.6 + index * 0.73) * (0.4 + 0.6 * health));
    mid.addScaledVector(S.perp, frayPulse * 0.045 * (planeLayout ? 1.25 : 1));
    mid.addScaledVector(S.perp, tension * (planeLayout ? 0.022 : 0.065) * Math.sin(t * 4.2 + index * 1.1));
    const curve = S.curve;
    curve.v0.copy(posA);
    curve.v1.copy(mid);
    curve.v2.copy(posB);
    const divisions = typeof ref.divisions === "number" ? ref.divisions : g.curveSegments || 9;
    const tmp = S.tmpPoint;
    if (useL2 && ref.lineGeomMain && ref.lineGeomGlow && ref.line2Main && ref.line2Glow) {
      const posFlat = [];
      for (let i = 0; i <= divisions; i += 1) {
        curve.getPoint(i / divisions, tmp);
        posFlat.push(tmp.x, tmp.y, tmp.z);
      }
      ref.lineGeomMain.setPositions(posFlat);
      ref.lineGeomGlow.setPositions(posFlat);
      ref.line2Main.computeLineDistances();
      ref.line2Glow.computeLineDistances();
    } else {
      const posAttr = line.geometry.attributes.position;
      const arr = posAttr.array;
      for (let i = 0; i <= divisions; i += 1) {
        curve.getPoint(i / divisions, tmp);
        const o = i * 3;
        arr[o] = tmp.x;
        arr[o + 1] = tmp.y;
        arr[o + 2] = tmp.z;
      }
      posAttr.needsUpdate = true;
    }
    const mag = Math.min(1, Math.abs(Number(edge.magnitude || 0)));
    colA.copy(colHealthyA).lerp(colStressA, stress01 * 0.88);
    colB.copy(colHealthyB).lerp(colStressB, stress01 * 0.82);

    const wave = 0.5 + 0.5 * Math.sin(t * (2.6 + coherence * 1.2) - index * 0.31);
    let pulse = 0.14 + 0.52 * mag * (0.45 + 0.55 * coherence) * (0.55 + 0.45 * health);
    if (planeLayout && verticalSpan > 0.06) {
      pulse +=
        0.09 +
        0.15 * Math.min(1, verticalSpan * 1.5) +
        tension * 0.12 * (0.5 + 0.5 * Math.sin(t * 4.1 + index));
    }
    pulse *= 0.84 + 0.16 * wave;

    if (useL2 && ref.lineMatMain && ref.lineMatGlow) {
      ref.lineMatMain.color.copy(colA);
      ref.lineMatGlow.color.copy(colB);
      const flicker = 0.55 + 0.45 * Math.sin(t * (3.1 + stress01 * 5) + index * 0.9);
      const erratic = stress01 * (0.35 + 0.65 * Math.abs(Math.sin(t * 11.3 + index * 2.1)));
      if (useTubes && ref.tubeMesh) {
        ref.lineMatMain.opacity = Math.min(0.55, 0.12 + 0.28 * health * wave);
        ref.lineMatGlow.opacity = Math.min(0.45, 0.08 + 0.22 * health * flicker * (1 - erratic * 0.75));
      } else {
        ref.lineMatMain.opacity = Math.min(
          0.94,
          0.28 + 0.52 * pulse + tension * (planeLayout ? 0.14 : 0.1) * Math.sin(t * 3 + index) * stress01,
        );
        ref.lineMatGlow.opacity = Math.min(
          0.72,
          (0.18 + 0.48 * health) * flicker * (1 - erratic * 0.85),
        );
      }
    } else if (line.material) {
      line.material.color.copy(colA);
      if (useTubes && ref.tubeMesh) {
        line.material.opacity = Math.min(0.22, 0.06 + 0.12 * health * wave);
      } else {
        line.material.opacity = Math.min(
          0.96,
          pulse + tension * (planeLayout ? 0.2 : 0.14) * Math.sin(t * 3 + index) * stress01,
        );
        line.material.opacity = Math.min(0.97, line.material.opacity + (0.08 + 0.14 * health) * wave);
      }
    }
    if (!useL2 && lineGlow && lineGlow.material) {
      lineGlow.material.color.copy(colB);
      const flicker = 0.55 + 0.45 * Math.sin(t * (3.1 + stress01 * 5) + index * 0.9);
      const erratic = stress01 * (0.35 + 0.65 * Math.abs(Math.sin(t * 11.3 + index * 2.1)));
      const perfGlow = perf ? 0.22 : 0.12;
      lineGlow.material.opacity = Math.min(
        0.82,
        (perfGlow + 0.52 * health) * flicker * (1 - erratic * 0.85) * (useTubes ? 0.55 : 1),
      );
    }
    if (!useL2 && line.material && perf && !useTubes && framePhase % 3 === index % 3) {
      line.material.opacity = Math.min(0.9, 0.22 + 0.52 * mag * (0.45 + 0.55 * coherence) * (0.5 + 0.5 * health));
    }

    if (useTubes && three.TubeGeometry) {
      const tubeRad =
        0.0034 +
        mag * 0.024 * (0.35 + 0.65 * health) +
        (1 - health) * 0.0038;
      const radialSeg = perf ? 4 : 5;
      if (!ref.tubeMesh) {
        ref.tubeMaterial = new three.MeshStandardMaterial({
          color: 0xffffff,
          metalness: 0.18,
          roughness: 0.4,
          envMapIntensity: 0.62,
          transparent: true,
          opacity: 0.88,
        });
        ref.tubeMesh = new three.Mesh(new three.BufferGeometry(), ref.tubeMaterial);
        ref.tubeMesh.userData.pickMesh = false;
        ref.tubeMesh.castShadow = !perf;
        ref.tubeMesh.receiveShadow = false;
        ref.tubeMesh.renderOrder = 0;
        line.parent.add(ref.tubeMesh);
      }
      ref.tubeGeom?.dispose?.();
      ref.tubeGeom = new three.TubeGeometry(curve, divisions, tubeRad, radialSeg, false);
      ref.tubeMesh.geometry = ref.tubeGeom;
      ref.tubeMaterial.color.copy(colA);
      ref.tubeMaterial.emissive.copy(colA);
      ref.tubeMaterial.emissiveIntensity = 0.08 + 0.22 * health + (1 - health) * 0.12 * Math.sin(t * 5.5 + index);
      ref.tubeMaterial.opacity = 0.42 + 0.48 * health * (0.65 + 0.35 * wave);
      ref.tubeMaterial.metalness = 0.1 + stress01 * 0.22;
      ref.tubeMaterial.roughness = 0.38 + stress01 * 0.28;
    } else if (ref.tubeMesh) {
      try {
        ref.tubeGeom?.dispose?.();
        ref.tubeGeom = null;
        ref.tubeMaterial?.dispose?.();
        ref.tubeMaterial = null;
        line.parent.remove(ref.tubeMesh);
      } catch (_e) {
        // no-op
      }
      ref.tubeMesh = null;
    }

    if (flowMesh && dummy && typeof flowMesh.setMatrixAt === "function") {
      const slot = typeof ref.particleSlotStart === "number" ? ref.particleSlotStart : index * pep;
      const baseSpeed = 0.12 + 0.24 * coherence + 0.18 * health;
      const chaos = (1 - health) * (0.016 + driftN * 0.026 + instN * 0.03);
      for (let p = 0; p < pep; p += 1) {
        const phase = index * 0.37 + p * 0.41;
        let u =
          (t * baseSpeed * (0.38 + health * 0.68) + phase + Math.sin(t * 0.32 + index) * (1 - coherence) * 0.035) %
          1;
        if (u < 0) u += 1;
        curve.getPoint(u, tmp);
        tmp.x += Math.sin(t * 6.8 + phase * 3) * chaos;
        tmp.y += Math.sin(t * 5.4 + index) * chaos * 0.55;
        tmp.z += Math.cos(t * 6.1 + phase * 2.2) * chaos;
        const ps = (0.68 + 0.42 * health) * (1 + stress01 * 0.38 * Math.sin(t * 8.2 + p));
        dummy.position.copy(tmp);
        dummy.scale.setScalar(ps * (perf ? 0.94 : 1));
        dummy.updateMatrix();
        flowMesh.setMatrixAt(slot + p, dummy.matrix);
        if (typeof flowMesh.setColorAt === "function") {
          colB.copy(colA).lerp(colHealthyB, (p / Math.max(1, pep - 1)) * 0.38);
          colB.lerp(colStressA, stress01 * 0.52);
          flowMesh.setColorAt(slot + p, colB);
        }
      }
    }
  });

  if (flowMesh?.instanceMatrix) {
    flowMesh.instanceMatrix.needsUpdate = true;
  }
  if (flowMesh?.instanceColor) {
    flowMesh.instanceColor.needsUpdate = true;
  }
}

function easeOutCubic(t) {
  const x = Math.max(0, Math.min(1, t));
  return 1 - (1 - x) ** 3;
}

function easeInOutCubic(t) {
  const x = Math.max(0, Math.min(1, t));
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2;
}

/** Snapshot current node world positions for continuous transitions between geometry updates. */
function captureGeometryNodeDisplayPositions() {
  const g = state.geometry3d;
  const out = {};
  Object.keys(g.nodeMeshById || {}).forEach((id) => {
    const m = g.nodeMeshById[id];
    if (m?.position) {
      out[id] = { x: m.position.x, y: m.position.y, z: m.position.z };
    }
  });
  return out;
}

/** Subtle structural field under the graph — depth anchor without heavy shading. */
function createStructuralFieldPlane(three, center, boxNg, perf) {
  if (!three?.PlaneGeometry || !boxNg || boxNg.isEmpty()) return null;
  const size = boxNg.getSize(new three.Vector3());
  const planeW = Math.max(6.5, Math.max(size.x, size.z) * 2.35);
  const cnv = document.createElement("canvas");
  cnv.width = 64;
  cnv.height = 64;
  const ctx = cnv.getContext("2d");
  if (!ctx) return null;
  const grd = ctx.createRadialGradient(32, 28, 4, 32, 32, 44);
  grd.addColorStop(0, "rgba(45, 212, 191, 0.12)");
  grd.addColorStop(0.45, "rgba(14, 40, 64, 0.08)");
  grd.addColorStop(1, "rgba(2, 6, 14, 0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 64, 64);
  ctx.strokeStyle = "rgba(56, 120, 200, 0.06)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 64; i += 8) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 64);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(64, i);
    ctx.stroke();
  }
  const tex = new three.CanvasTexture(cnv);
  tex.wrapS = three.RepeatWrapping;
  tex.wrapT = three.RepeatWrapping;
  tex.repeat.set(perf ? 2.4 : 3, perf ? 2.4 : 3);
  const mat = new three.MeshBasicMaterial({
    map: tex,
    transparent: true,
    opacity: perf ? 0.38 : 0.5,
    depthWrite: false,
  });
  const mesh = new three.Mesh(new three.PlaneGeometry(planeW, planeW, 1, 1), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(center.x, boxNg.min.y - 0.018, center.z);
  mesh.userData.pickMesh = false;
  mesh.renderOrder = -2;
  return mesh;
}

/** Subtitle under panel title: sensor/link counts and projection role. */
function updateStructuralFlowCopy(payload) {
  const el = qs("#geometryFlowSubtitle");
  if (!el) return;
  if (!payload?.available) {
    el.textContent = "Loads when pairwise structural geometry is available for this run.";
    return;
  }
  const m = computeStructuralFlowMetrics(payload);
  const n = m.totalSensors;
  const lmax = m.maxPairwiseLinks;
  if (n <= 0) {
    el.textContent = "No sensors in this structural view.";
    return;
  }
  el.textContent = `${n} sensor${n === 1 ? "" : "s"} · ${lmax} possible pairwise links · plane = relational layout; field view = energy on the manifold.`;
}

function setStructuralFlowMicrobar(el, frac01) {
  if (!el) return;
  const w = Math.max(0, Math.min(100, Math.round(frac01 * 100)));
  el.style.width = `${w}%`;
}

function updateGeometryFlowPanel(payload, opts) {
  const cohEl = qs("#geometryCoherence");
  const assessEl = qs("#geometryStructuralAssessment");
  const engineEl = qs("#geometryEnginePhase");
  const defEl = qs("#geometryMetricDefinitions");
  const details = qs("#geometryDetails");
  const g = state.geometry3d;

  const elContain = qs("#geometryFlowContainment");
  const elContainHint = qs("#geometryFlowContainmentHint");
  const elContainBar = qs("#geometryFlowContainmentBar");
  const elDrift = qs("#geometryFlowDrift");
  const elDriftHint = qs("#geometryFlowDriftHint");
  const elDriftBar = qs("#geometryFlowDriftBar");
  const elDriftDir = qs("#geometryFlowDriftDir");
  const elInst = qs("#geometryFlowInstability");
  const elInstHint = qs("#geometryFlowInstabilityHint");
  const elInstBar = qs("#geometryFlowInstabilityBar");
  const elLink = qs("#geometryFlowLinkAgree");
  const elLinkHint = qs("#geometryFlowLinkHint");
  const elLinkBar = qs("#geometryFlowLinkBar");
  const elInManifold = qs("#geometryInsideManifold");
  const elOutManifold = qs("#geometryOutsideManifold");
  const elPressure = qs("#geometryBoundaryPressure");
  const elStatusText = qs("#geometryFlowStatusText");
  const elLinkInline = qs("#geometryFlowLinkAgreeInline");

  if (!payload || !payload.available) {
    if (cohEl) {
      cohEl.textContent = "—";
      cohEl.removeAttribute("title");
    }
    if (assessEl) {
      assessEl.textContent = "—";
      assessEl.removeAttribute("title");
    }
    if (engineEl) {
      engineEl.textContent = "—";
      engineEl.removeAttribute("title");
    }
    if (defEl) defEl.textContent = "";
    if (details) details.setAttribute("data-geometry-risk", "UNKNOWN");
    [
      elContain,
      elContainHint,
      elDrift,
      elDriftHint,
      elInst,
      elInstHint,
      elLink,
      elLinkHint,
      elInManifold,
      elOutManifold,
      elPressure,
      elStatusText,
      elLinkInline,
    ].forEach((el) => {
      if (el) el.textContent = "—";
    });
    if (elDriftDir) {
      elDriftDir.textContent = "—";
      elDriftDir.removeAttribute("title");
    }
    setStructuralFlowMicrobar(elContainBar, 0);
    setStructuralFlowMicrobar(elDriftBar, 0);
    setStructuralFlowMicrobar(elInstBar, 0);
    setStructuralFlowMicrobar(elLinkBar, 0);
    updateStructuralFlowCopy(payload || { available: false, nodes: [] });
    return;
  }

  const mBase = computeStructuralFlowMetrics(payload);
  if (!g?.useCanvas2d || !g.structuralFlowDerived) {
    g.structuralFlowDerived = deriveStructuralFlowDerivedFallback(payload);
  }
  const derived = g.structuralFlowDerived;
  let displayIntegrity = derived.integrityScore;
  if (opts?.structuralFlowCoherence01 != null && Number.isFinite(opts.structuralFlowCoherence01)) {
    displayIntegrity = opts.structuralFlowCoherence01;
  }

  const integ = displayIntegrity;
  if (cohEl) {
    cohEl.textContent = `${Math.round(integ * 100)}%`;
    cohEl.title =
      "Integrity score (0–100%): blended field read — sensor tolerance, link agreement, energy containment in the core band, and centroid offset. Same blend as the heatmap threshold when the flow field is sampled.";
  }

  const c01 = derived.containmentScore;
  if (elContain) elContain.textContent = `${Math.round(c01 * 100)}%`;
  if (elContainHint) elContainHint.textContent = structuralFlowContainmentHint(derived);
  setStructuralFlowMicrobar(elContainBar, c01);

  const d01 = derived.driftMag01;
  if (elDrift) {
    elDrift.textContent = d01.toFixed(2);
    elDrift.title = `Normalized net drift (0–1); raw mean |V| ≈ ${derived.driftMagnitude.toFixed(3)}.`;
  }
  if (elDriftHint)
    elDriftHint.textContent = structuralFlowBandHint(d01, 0.28, 0.55, "Calm", "Rising", "High");
  setStructuralFlowMicrobar(elDriftBar, d01);
  if (elDriftDir) {
    elDriftDir.textContent =
      derived.driftCompass && derived.driftCompass !== "—"
        ? `Net ${derived.driftCompass}`
        : "Net —";
    elDriftDir.title = "Direction of E-weighted mean flow (matches the cyan arrow).";
  }

  const i01 = derived.instabilityScore;
  if (elInst) elInst.textContent = `${Math.round(i01 * 100)}%`;
  if (elInstHint)
    elInstHint.textContent = structuralFlowBandHint(i01, 0.25, 0.55, "Calm", "Active", "Severe");
  setStructuralFlowMicrobar(elInstBar, i01);

  const l01 = derived.linkAgreementScore;
  if (elLink) elLink.textContent = `${Math.round(l01 * 100)}%`;
  if (elLinkHint)
    elLinkHint.textContent = structuralFlowBandHint(l01, 0.45, 0.75, "Low", "Moderate", "Link Agreement");
  setStructuralFlowMicrobar(elLinkBar, l01);
  if (elLinkInline) elLinkInline.textContent = `${Math.round(l01 * 100)}%`;

  if (elInManifold) elInManifold.textContent = String(derived.insideCount);
  if (elOutManifold) elOutManifold.textContent = String(derived.outsideCount);
  if (elPressure) {
    elPressure.textContent = `${Math.round(derived.boundaryPressureScore * 100)}%`;
    elPressure.title = "Share of field energy in the border band next to the dashed manifold edge.";
  }

  if (assessEl) {
    assessEl.textContent = derived.statusLabel;
    assessEl.title = derived.statusDetail;
  }
  if (elStatusText) {
    elStatusText.textContent = derived.statusLabel;
    elStatusText.title = derived.statusDetail;
  }
  if (engineEl) {
    engineEl.textContent = toPretty(mBase.engineState);
    engineEl.title = "Engine phase label from this run (orthogonal to the field interpretation).";
  }
  if (defEl) defEl.textContent = structuralMetricDefinitionsFootnote();
  if (details) details.setAttribute("data-geometry-risk", derived.tone || "UNKNOWN");
  updateStructuralFlowCopy(payload);
}

function geometryFlowSummaryString(payload) {
  if (!payload?.available) return "—";
  const base = computeStructuralFlowMetrics(payload);
  const g = state.geometry3d;
  const derived = g?.useCanvas2d && g.structuralFlowDerived ? g.structuralFlowDerived : null;
  const m =
    derived && typeof derived.integrityScore === "number"
      ? { ...base, coherence01: derived.integrityScore }
      : g?.useCanvas2d && typeof g.structuralFlowCoherence01 === "number"
        ? { ...base, coherence01: g.structuralFlowCoherence01 }
        : base;
  const assessment = derived
    ? { label: derived.statusLabel, detail: derived.statusDetail, tone: derived.tone }
    : deriveStructuralAssessment(m);
  return buildStructuralFlowNarrative(m, assessment, derived);
}

function scheduleGeometryResize(g, fn) {
  if (!g || typeof fn !== "function") return;
  if (g.resizeRaf) {
    window.cancelAnimationFrame(g.resizeRaf);
  }
  g.resizeRaf = window.requestAnimationFrame(() => {
    g.resizeRaf = null;
    fn();
  });
}

/** Deterministic scalar in [0,1) from a string key (stable across frames; no Math.random). */
function structuralFlowHash01(key) {
  const str = String(key ?? "");
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 10001) / 10000;
}

/** Risk → isotropic expansion pressure on emitters (0–1). */
function structuralFlowRiskExpansion01(payload) {
  const r = normalizeRiskLevel(payload?.metrics?.risk_level);
  if (r === "HIGH") return 0.88;
  if (r === "MEDIUM") return 0.48;
  if (r === "LOW") return 0.14;
  return 0.28;
}

/**
 * Unit direction for drift bias inside V (not the arrow — arrow comes from E-weighted mean V).
 */
function structuralFlowDriftBiasDirectionUV(nodes, normToPlane) {
  let wcx = 0;
  let wcz = 0;
  let sw = 0;
  nodes.forEach((n) => {
    const px = Number(n.position?.x) || 0;
    const pz = Number(n.position?.z) || 0;
    const { nx, nz } = normToPlane(px, pz);
    const stress = flowNodeStress01(n);
    const wt = 0.15 + 0.85 * stress;
    wcx += nx * wt;
    wcz += nz * wt;
    sw += wt;
  });
  if (sw < 1e-9) return { x: 1, y: 0 };
  wcx /= sw;
  wcz /= sw;
  let dx = wcx - 0.5;
  let dz = wcz - 0.5;
  const len = Math.hypot(dx, dz);
  if (len < 1e-6) return { x: 0, y: 1 };
  dx /= len;
  dz /= len;
  return { x: dx, y: dz };
}

function structuralFlowForceAt(pu, pv, nodes, edges, normToPlane, driftDir, driftN, instN, riskExp) {
  let fx = 0;
  let fy = 0;
  /* Wider, softer influence than inverse-square: Gaussian-like falloff (single connected field). */
  const rBase = 0.15 + 0.05 * driftN;

  nodes.forEach((n) => {
    const px = Number(n.position?.x) || 0;
    const pz = Number(n.position?.z) || 0;
    const { nx, nz } = normToPlane(px, pz);
    const du = nx - pu;
    const dv = nz - pv;
    const d2 = du * du + dv * dv;
    const stress = flowNodeStress01(n);
    const inTol = flowNodeWithinTolerance(n);
    const unstable = Boolean(n.is_unstable);
    const sigma = rBase * (1 + 0.5 * riskExp) * (1 + 0.28 * stress) * 1.45;
    const g = Math.exp(-d2 / (2 * sigma * sigma + 1e-12));

    if (inTol && !unstable) {
      const amp = (1 + 0.16 * (1 - stress)) * 1.2;
      fx += du * g * amp;
      fy += dv * g * amp;
    } else {
      const rep = (unstable ? 1.2 : 0.8) * (0.68 + 0.32 * stress) * 1.02;
      fx -= du * g * rep;
      fy -= dv * g * rep;
    }
  });

  (edges || []).forEach((edge) => {
    const a = nodes.find((nn) => String(nn.id) === String(edge.source));
    const b = nodes.find((nn) => String(nn.id) === String(edge.target));
    if (!a || !b) return;
    const ax = Number(a.position?.x) || 0;
    const az = Number(a.position?.z) || 0;
    const bx = Number(b.position?.x) || 0;
    const bz = Number(b.position?.z) || 0;
    const pa = normToPlane(ax, az);
    const pb = normToPlane(bx, bz);
    const mx = (pa.nx + pb.nx) / 2;
    const mz = (pa.nz + pb.nz) / 2;
    const delta = Math.abs(Number(edge.delta || 0));
    const mag = Math.abs(Number(edge.magnitude || 0));
    const disagree = Math.max(0, Math.min(1, 0.55 * delta + 0.45 * mag));
    if (disagree < 1e-4) return;
    const ddx = pu - mx;
    const ddz = pv - mz;
    const d2 = ddx * ddx + ddz * ddz + 1e-9;
    const sig = 0.14 + 0.1 * disagree;
    const w = disagree * Math.exp(-d2 / (2 * sig * sig)) * 1.05;
    const len = Math.sqrt(d2);
    if (len < 1e-7) return;
    fx += w * (ddx / len) * 0.42;
    fy += w * (ddz / len) * 0.42;
  });

  const dmag = 0.34 + 0.66 * driftN;
  fx += driftDir.x * dmag * 0.42;
  fy += driftDir.y * dmag * 0.42;

  const om = 2.15 * Math.PI;
  fx += instN * 0.072 * Math.sin(om * pu + 0.77 * pv);
  fy += instN * 0.072 * Math.sin(om * pv - 0.48 * pu);

  return { fx, fy };
}

function structuralFlowSmoothstep01(edge0, edge1, x) {
  if (x <= edge0) return 0;
  if (x >= edge1) return 1;
  const t = (x - edge0) / (edge1 - edge0);
  return t * t * (3 - 2 * t);
}

/** Separable box blur on corner grid (smooths hotspots into one continuous field). */
function structuralFlowBoxBlurE2D(E, NX, NY, r) {
  const tmp = [];
  for (let i = 0; i <= NX; i += 1) {
    tmp[i] = [];
    for (let j = 0; j <= NY; j += 1) {
      let s = 0;
      let c = 0;
      for (let di = -r; di <= r; di += 1) {
        const ii = Math.max(0, Math.min(NX, i + di));
        s += E[ii][j];
        c += 1;
      }
      tmp[i][j] = s / c;
    }
  }
  const out = [];
  for (let i = 0; i <= NX; i += 1) {
    out[i] = [];
    for (let j = 0; j <= NY; j += 1) {
      let s = 0;
      let c = 0;
      for (let dj = -r; dj <= r; dj += 1) {
        const jj = Math.max(0, Math.min(NY, j + dj));
        s += tmp[i][jj];
        c += 1;
      }
      out[i][j] = s / c;
    }
  }
  return out;
}

/** Energy threshold whose super-level set keeps the top targetMassRatio of field energy. */
function structuralFlowMassThreshold(E, NX, NY, targetMassRatio, fallbackMin, fallbackMax) {
  const values = [];
  let total = 0;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const e = Number(E[i]?.[j]) || 0;
      values.push(e);
      total += e;
    }
  }
  if (!values.length || total <= 1e-12) return fallbackMin + 0.5 * Math.max(0, fallbackMax - fallbackMin);
  values.sort((a, b) => b - a);
  const target = Math.max(0.35, Math.min(0.96, Number(targetMassRatio) || 0.78)) * total;
  let acc = 0;
  for (let k = 0; k < values.length; k += 1) {
    acc += values[k];
    if (acc >= target) return values[k];
  }
  return values[values.length - 1];
}

/** Skew scalar energy along drift so the surface stretches toward net drift (aligns with arrow). */
function structuralFlowApplyDriftSkewScalar(E, NX, NY, driftDir, driftN) {
  const out = [];
  const du = driftDir.x;
  const dv = driftDir.y;
  const k = 1.95 * (0.35 + 0.65 * driftN);
  const gain = 1 + 0.68 * driftN;
  for (let i = 0; i <= NX; i += 1) {
    out[i] = [];
    const u = i / NX;
    for (let j = 0; j <= NY; j += 1) {
      const v = j / NY;
      const tilt = du * (u - 0.5) + dv * (v - 0.5);
      const cross = -dv * (u - 0.5) + du * (v - 0.5);
      const mult = Math.exp(k * tilt - 0.35 * Math.abs(cross) * driftN) * gain;
      out[i][j] = E[i][j] * mult;
    }
  }
  return out;
}

/** Global compression field that increases as the dominant contour approaches manifold boundaries. */
function structuralFlowApplyBoundaryCompression(
  E,
  NX,
  NY,
  contourEdgeDistance01,
  contourCrossing01,
  driftDir,
  driftN,
  instN,
) {
  const out = [];
  const pressure = Math.max(
    0,
    Math.min(1, 0.72 * (1 - contourEdgeDistance01) + 0.28 * contourCrossing01),
  );
  const strength = pressure * (0.16 + 0.58 * Math.max(driftN, instN));
  if (strength <= 1e-5) return E;
  const driftLen = Math.hypot(driftDir.x, driftDir.y);
  const dx = driftLen > 1e-9 ? driftDir.x / driftLen : 1;
  const dy = driftLen > 1e-9 ? driftDir.y / driftLen : 0;
  const edgeBand = 0.22;
  for (let i = 0; i <= NX; i += 1) {
    out[i] = [];
    const u = i / NX;
    for (let j = 0; j <= NY; j += 1) {
      const v = j / NY;
      const edgeDist = Math.min(u, 1 - u, v, 1 - v);
      const edgeProx = 1 - structuralFlowSmoothstep01(0, edgeBand, edgeDist);
      const alongDrift = dx * (u - 0.5) + dy * (v - 0.5);
      const driftPush = Math.max(0, alongDrift) * (0.75 + 0.25 * driftN);
      const squeeze = 1 - Math.min(0.58, strength * edgeProx * (0.85 + 0.5 * driftPush));
      out[i][j] = E[i][j] * Math.max(0.48, squeeze);
    }
  }
  return out;
}

function structuralFlowCornerTouchesLargestCC(largestMask, NX, NY, i, j) {
  const cells = [
    [i - 1, j - 1],
    [i, j - 1],
    [i - 1, j],
    [i, j],
  ];
  for (let k = 0; k < cells.length; k += 1) {
    const ci = cells[k][0];
    const cj = cells[k][1];
    if (ci >= 0 && ci < NX && cj >= 0 && cj < NY && largestMask[ci][cj]) return true;
  }
  return false;
}

/** Largest 4-connected high cell region kills stray islands for one dominant iso-boundary. */
function structuralFlowLargestCCMask(cellHigh, NX, NY) {
  const labels = [];
  const sizes = [];
  let cur = 0;
  const minKeep = Math.max(6, Math.floor(NX * NY * 0.012));
  for (let i = 0; i < NX; i += 1) {
    labels[i] = [];
    for (let j = 0; j < NY; j += 1) {
      labels[i][j] = 0;
    }
  }
  for (let i = 0; i < NX; i += 1) {
    for (let j = 0; j < NY; j += 1) {
      if (!cellHigh[i][j] || labels[i][j]) continue;
      cur += 1;
      let sz = 0;
      const stack = [[i, j]];
      while (stack.length) {
        const p = stack.pop();
        const x = p[0];
        const y = p[1];
        if (x < 0 || y < 0 || x >= NX || y >= NY) continue;
        if (!cellHigh[x][y] || labels[x][y]) continue;
        labels[x][y] = cur;
        sz += 1;
        stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
      }
      sizes[cur] = sz;
    }
  }
  let best = 0;
  let bestId = 0;
  for (let k = 1; k <= cur; k += 1) {
    if (sizes[k] > best) {
      best = sizes[k];
      bestId = k;
    }
  }
  const out = [];
  for (let i = 0; i < NX; i += 1) {
    out[i] = [];
    for (let j = 0; j < NY; j += 1) {
      const id = labels[i][j];
      const isTinyIsland = id > 0 && sizes[id] < minKeep;
      out[i][j] = bestId > 0 && labels[i][j] === bestId && !isTinyIsland;
    }
  }
  return out;
}

/** Corners only on the dominant super-level set → one main contour, no stray rings. */
function structuralFlowIsoFieldForMarching(E, NX, NY, minE, largestMask) {
  const below = minE - 0.02;
  const Eiso = [];
  for (let i = 0; i <= NX; i += 1) {
    Eiso[i] = [];
    for (let j = 0; j <= NY; j += 1) {
      Eiso[i][j] = structuralFlowCornerTouchesLargestCC(largestMask, NX, NY, i, j) ? E[i][j] : below;
    }
  }
  return Eiso;
}

function structuralFlowPrimarySegmentComponent(segs) {
  const PREC = 1e5;
  const R = (x) => Math.round(x * PREC) / PREC;
  const key = (x, y) => `${R(x)},${R(y)}`;
  const n = segs.length;
  const visited = new Array(n).fill(false);
  let bestSet = new Set();
  let bestLen = -1;
  for (let s0 = 0; s0 < n; s0 += 1) {
    if (visited[s0]) continue;
    const q = [s0];
    visited[s0] = true;
    const comp = [];
    let len = 0;
    while (q.length) {
      const si = q.shift();
      comp.push(si);
      const s = segs[si];
      len += Math.hypot(s[2] - s[0], s[3] - s[1]);
      const k1 = key(s[0], s[1]);
      const k2 = key(s[2], s[3]);
      for (let sj = 0; sj < n; sj += 1) {
        if (visited[sj]) continue;
        const t = segs[sj];
        const t1 = key(t[0], t[1]);
        const t2 = key(t[2], t[3]);
        if (t1 === k1 || t1 === k2 || t2 === k1 || t2 === k2) {
          visited[sj] = true;
          q.push(sj);
        }
      }
    }
    if (len > bestLen) {
      bestLen = len;
      bestSet = new Set(comp);
    }
  }
  return bestSet;
}

/** Share of sampled field energy in the ~10% border band (pressure at the dashed manifold edge). */
function computeStructuralFlowBoundaryPressure01(E, NX, NY) {
  const band = 0.1;
  let margin = 0;
  let tot = 0;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const u = i / NX;
      const v = j / NY;
      const e = E[i][j];
      tot += e;
      const ed = Math.min(u, 1 - u, v, 1 - v);
      if (ed < band) margin += e;
    }
  }
  return tot > 1e-12 ? margin / tot : 0;
}

/** 8-point compass for net drift in normalized u,v (v down on screen). */
function formatStructuralFlowCompass8(dx, dy) {
  if (Math.hypot(dx, dy) < 1e-7) return "—";
  const ang = Math.atan2(-dy, dx);
  const sec = (Math.round(ang / (Math.PI / 4)) + 8) % 8;
  const rosa = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
  return rosa[sec];
}

function structuralFlowBandHint(v, t0, t1, a, b, c) {
  if (v <= t0) return a;
  if (v >= t1) return c;
  return b;
}

function structuralFlowContainmentHint(derived) {
  const outside = Number(derived?.outsideCount) || 0;
  const pressure = Number(derived?.boundaryPressureScore) || 0;
  if (outside === 0 && pressure < 0.24) return "Inside Manifold";
  if (outside > 0 && pressure >= 0.38) return "Outside Manifold";
  if (pressure >= 0.3) return "Boundary Pressure";
  return "Inside Manifold";
}

/**
 * Flow state label for the side panel — driven by the same containment / drift / instability / escape signals as the canvas.
 */
function deriveStructuralFlowFieldStatus(d) {
  const { integrity01, containment01, driftMag01, instN, escapeRatio, boundaryPressure01 } = d;
  if (escapeRatio > 0.4 || containment01 < 0.36) {
    return {
      label: "Escaping Boundary",
      detail: "Iso-contour or energy is leaving the valid manifold (orange/red segments).",
      tone: "HIGH",
    };
  }
  if (integrity01 < 0.3 && driftMag01 > 0.52 && instN > 0.48) {
    return {
      label: "Unstable Drift",
      detail: "Strong net flow and high fragmentation relative to baseline.",
      tone: "HIGH",
    };
  }
  if (boundaryPressure01 > 0.44 && containment01 < 0.55) {
    return {
      label: "Under Pressure",
      detail: "Edge-weighted energy is high; contour is compressing the border.",
      tone: "MEDIUM",
    };
  }
  if (boundaryPressure01 > 0.26 && driftMag01 > 0.36) {
    return {
      label: "Under Pressure",
      detail: "Net drift and edge band energy are elevated; watch the dashed boundary.",
      tone: "MEDIUM",
    };
  }
  if (containment01 > 0.68 && driftMag01 < 0.34 && instN < 0.34 && escapeRatio < 0.2) {
    return {
      label: "Contained",
      detail: "Energy centroid and iso-contour favor the interior of the manifold.",
      tone: "LOW",
    };
  }
  return {
    label: "Monitoring",
    detail: "Mixed field signals; no single failure mode dominates this snapshot.",
    tone: "LOW",
  };
}

/**
 * Single object used by the 2D canvas + side panel: same field model, no duplicate math.
 * Fallback path uses payload metrics only when the grid has not been sampled (3D-only view).
 */
function finalizeStructuralFlowDerivedState(model, segs, driftN, instN) {
  let escaped = 0;
  const total = segs.length;
  for (let i = 0; i < segs.length; i += 1) {
    const s = segs[i];
    if (structuralFlowSegmentEscapesManifold(s[0], s[1], s[2], s[3])) escaped += 1;
  }
  const escapeRatio = total > 0 ? escaped / total : 0;
  const boundaryPressure01 = computeStructuralFlowBoundaryPressure01(model.E, model.NX, model.NY);
  const driftMag = Math.hypot(model.netVx, model.netVy);
  const driftMag01 = Math.max(0, Math.min(1, driftMag * 3.2 + driftN * 0.38));
  const containment01 = model.containmentMassRatio;
  const integrity01 = model.coherence01;
  const st = deriveStructuralFlowFieldStatus({
    integrity01,
    containment01,
    driftMag01,
    instN,
    escapeRatio,
    boundaryPressure01,
  });
  const len = driftMag;
  const nx = len > 1e-9 ? model.netVx / len : 0;
  const ny = len > 1e-9 ? model.netVy / len : 0;
  return {
    integrityScore: integrity01,
    containmentScore: containment01,
    driftMagnitude: driftMag,
    driftMag01,
    driftDirection: { x: nx, y: ny },
    driftCompass: formatStructuralFlowCompass8(nx, ny),
    instabilityScore: instN,
    linkAgreementScore: model.pairAgreement01,
    insideCount: model.metrics.withinTolerance,
    outsideCount: model.metrics.outsideTolerance,
    boundaryPressureScore: boundaryPressure01,
    escapedSegments: escaped,
    totalSegments: total,
    escapeRatio,
    statusLabel: st.label,
    statusDetail: st.detail,
    tone: st.tone,
    isFallback: false,
  };
}

function structuralFlowBuildIsoFromScalar(E, NX, NY, T, minE) {
  const cellHigh = Array.from({ length: NX }, () => Array(NY).fill(false));
  for (let i = 0; i < NX; i += 1) {
    for (let j = 0; j < NY; j += 1) {
      const av = (E[i][j] + E[i + 1][j] + E[i + 1][j + 1] + E[i][j + 1]) / 4;
      cellHigh[i][j] = av >= T;
    }
  }
  const largestMask = structuralFlowLargestCCMask(cellHigh, NX, NY);
  const E_iso = structuralFlowIsoFieldForMarching(E, NX, NY, minE, largestMask);
  return { largestMask, E_iso };
}

function deriveStructuralFlowDerivedFallback(payload) {
  const m = computeStructuralFlowMetrics(payload);
  const { driftN, instN } = flowDriftInstabilityNorm(payload);
  const linkAgreement = m.totalLinks > 0 ? 1 - Math.min(1, m.driftedLinks / m.totalLinks) : 1;
  const containment01 = m.totalSensors > 0 ? m.withinTolerance / m.totalSensors : 0;
  const driftMag01 = Math.max(0, Math.min(1, driftN));
  const boundaryPressure01 = Math.max(0, Math.min(1, 1 - containment01));
  const st = deriveStructuralFlowFieldStatus({
    integrity01: m.coherence01,
    containment01,
    driftMag01,
    instN,
    escapeRatio: 0,
    boundaryPressure01,
  });
  return {
    integrityScore: m.coherence01,
    containmentScore: containment01,
    driftMagnitude: driftN,
    driftMag01,
    driftDirection: { x: 1, y: 0 },
    driftCompass: "—",
    instabilityScore: instN,
    linkAgreementScore: linkAgreement,
    insideCount: m.withinTolerance,
    outsideCount: m.outsideTolerance,
    boundaryPressureScore: boundaryPressure01,
    escapedSegments: 0,
    totalSegments: 0,
    escapeRatio: 0,
    statusLabel: st.label,
    statusDetail: st.detail,
    tone: st.tone,
    isFallback: true,
  };
}

/**
 * Single shared model: sample V on a 60×30 grid, E = |V|², coherence from tolerance + pairwise + containment + centroid.
 */
function buildStructuralFlowFieldModel(payload, normToPlane, driftDir, driftN, instN, riskExp) {
  const NX = 60;
  const NY = 30;
  const nodes = payload.nodes || [];
  const edges = payload.edges || [];
  const mCount = computeStructuralFlowMetrics(payload);
  const Vx = [];
  const Vy = [];
  const E0 = [];
  for (let i = 0; i <= NX; i += 1) {
    Vx[i] = [];
    Vy[i] = [];
    E0[i] = [];
    for (let j = 0; j <= NY; j += 1) {
      const u = i / NX;
      const v = j / NY;
      const { fx, fy } = structuralFlowForceAt(u, v, nodes, edges, normToPlane, driftDir, driftN, instN, riskExp);
      Vx[i][j] = fx;
      Vy[i][j] = fy;
      E0[i][j] = fx * fx + fy * fy;
    }
  }

  const smoothRadius = driftN > 0.56 || instN > 0.56 ? 2 : 3;
  const smoothPasses = driftN > 0.56 || instN > 0.56 ? 3 : 4;
  let E = E0;
  for (let k = 0; k < smoothPasses; k += 1) E = structuralFlowBoxBlurE2D(E, NX, NY, smoothRadius);
  E = structuralFlowApplyDriftSkewScalar(E, NX, NY, driftDir, driftN);

  let minE = Infinity;
  let maxE = -Infinity;
  let massTot = 0;
  let massIn = 0;
  let sw = 0;
  let swu = 0;
  let swv = 0;
  const u0 = 0.2;
  const u1 = 0.8;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const e = E[i][j];
      minE = Math.min(minE, e);
      maxE = Math.max(maxE, e);
      massTot += e;
      const u = i / NX;
      const vv = j / NY;
      if (u >= u0 && u <= u1 && vv >= u0 && vv <= u1) massIn += e;
      sw += e;
      swu += u * e;
      swv += vv * e;
    }
  }

  const tolRatio =
    mCount.totalSensors > 0 ? Math.max(0, Math.min(1, mCount.withinTolerance / mCount.totalSensors)) : 0;
  const pairScore =
    mCount.totalLinks > 0
      ? Math.max(0, Math.min(1, 1 - Math.min(1, mCount.driftedLinks / mCount.totalLinks)))
      : 1;
  let containment = massTot > 1e-12 ? massIn / massTot : 0.5;
  const cu = sw > 1e-12 ? swu / sw : 0.5;
  const cv = sw > 1e-12 ? swv / sw : 0.5;
  const continuityFloor = minE + (maxE - minE) * 0.08;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      E[i][j] = continuityFloor + (E[i][j] - continuityFloor) * 0.92;
    }
  }
  let boundaryPressure01 = computeStructuralFlowBoundaryPressure01(E, NX, NY);
  minE = Infinity;
  maxE = -Infinity;
  massTot = 0;
  massIn = 0;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const e = E[i][j];
      minE = Math.min(minE, e);
      maxE = Math.max(maxE, e);
      massTot += e;
      const u = i / NX;
      const vv = j / NY;
      if (u >= u0 && u <= u1 && vv >= u0 && vv <= u1) massIn += e;
    }
  }
  let span = Math.max(maxE - minE, 1e-12);
  containment = massTot > 1e-12 ? massIn / massTot : containment;
  const targetMass = 0.76 + 0.11 * (1 - containment) - 0.06 * Math.max(driftN, instN);
  const massQuantileCut = structuralFlowMassThreshold(E, NX, NY, targetMass, minE, maxE);
  const centroidScore = Math.max(0, Math.min(1, 1 - Math.min(1, 2.2 * Math.hypot(cu - 0.5, cv - 0.5))));
  let coherence01 = Math.max(
    0,
    Math.min(
      1,
      0.28 * tolRatio + 0.22 * pairScore + 0.26 * containment + 0.24 * centroidScore,
    ),
  );
  if (mCount.totalSensors <= 0) coherence01 = 0;

  const coherenceThreshold = minE + span * (0.22 + 0.66 * coherence01);
  const T = Math.max(massQuantileCut, coherenceThreshold);

  const cellHigh = [];
  for (let i = 0; i < NX; i += 1) {
    cellHigh[i] = [];
    for (let j = 0; j < NY; j += 1) {
      const av = (E[i][j] + E[i + 1][j] + E[i + 1][j + 1] + E[i][j + 1]) / 4;
      cellHigh[i][j] = av >= T;
    }
  }
  let largestMask = structuralFlowLargestCCMask(cellHigh, NX, NY);
  let E_iso = structuralFlowIsoFieldForMarching(E, NX, NY, minE, largestMask);

  const segsPreview = structuralFlowMarchingSquaresSegments(E_iso, NX, NY, T);
  const primaryPreviewIdx = structuralFlowPrimarySegmentComponent(segsPreview);
  const previewPrimary = segsPreview.filter((_, iSeg) => primaryPreviewIdx.has(iSeg));
  const boundaryMetrics = structuralFlowContourBoundaryMetrics(previewPrimary);
  E = structuralFlowApplyBoundaryCompression(
    E,
    NX,
    NY,
    boundaryMetrics.edgeDistance01,
    boundaryMetrics.crossingRatio,
    driftDir,
    driftN,
    instN,
  );

  minE = Infinity;
  maxE = -Infinity;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      minE = Math.min(minE, E[i][j]);
      maxE = Math.max(maxE, E[i][j]);
    }
  }
  span = Math.max(maxE - minE, 1e-12);
  boundaryPressure01 = computeStructuralFlowBoundaryPressure01(E, NX, NY);

  const T2 = Math.max(minE + span * (0.24 + 0.1 * (1 - containment)), minE + span * (0.22 + 0.64 * coherence01));
  for (let i = 0; i < NX; i += 1) {
    for (let j = 0; j < NY; j += 1) {
      const av = (E[i][j] + E[i + 1][j] + E[i + 1][j + 1] + E[i][j + 1]) / 4;
      cellHigh[i][j] = av >= T2;
    }
  }
  largestMask = structuralFlowLargestCCMask(cellHigh, NX, NY);
  E_iso = structuralFlowIsoFieldForMarching(E, NX, NY, minE, largestMask);

  let netVx = 0;
  let netVy = 0;
  let wv = 0;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const w = E[i][j];
      netVx += Vx[i][j] * w;
      netVy += Vy[i][j] * w;
      wv += w;
    }
  }
  if (wv > 1e-12) {
    netVx /= wv;
    netVy /= wv;
  }

  return {
    NX,
    NY,
    Vx,
    Vy,
    E,
    E_iso,
    largestMask,
    minE,
    maxE,
    T: T2,
    coherence01,
    containmentMassRatio: containment,
    pairAgreement01: pairScore,
    tolRatio,
    cu,
    cv,
    boundaryPressure01,
    netVx,
    netVy,
    metrics: mCount,
  };
}

function structuralFlowInterpEdge(a, b, pa, pb, T) {
  if (Math.abs(a - b) < 1e-12) return null;
  if ((a >= T) === (b >= T)) return null;
  const t = (T - a) / (b - a);
  return [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
}

/** Marching squares on E ≥ T; corners v0..v3 = BL, BR, TR, TL per cell (u right, v down on screen). */
function structuralFlowMarchingSquaresSegments(E, nx, ny, T) {
  const out = [];
  const pushSeg = (p, q) => {
    if (p && q) out.push([p[0], p[1], q[0], q[1]]);
  };
  for (let i = 0; i < nx; i += 1) {
    for (let j = 0; j < ny; j += 1) {
      const v0 = E[i][j + 1];
      const v1 = E[i + 1][j + 1];
      const v2 = E[i + 1][j];
      const v3 = E[i][j];
      const x0 = i / nx;
      const x1 = (i + 1) / nx;
      const y0 = (j + 1) / ny;
      const y1 = j / ny;
      const pBL = [x0, y0];
      const pBR = [x1, y0];
      const pTR = [x1, y1];
      const pTL = [x0, y1];
      const e0 = structuralFlowInterpEdge(v0, v1, pBL, pBR, T);
      const e1 = structuralFlowInterpEdge(v1, v2, pBR, pTR, T);
      const e2 = structuralFlowInterpEdge(v2, v3, pTR, pTL, T);
      const e3 = structuralFlowInterpEdge(v3, v0, pTL, pBL, T);
      const idx = (v0 >= T ? 1 : 0) | (v1 >= T ? 2 : 0) | (v2 >= T ? 4 : 0) | (v3 >= T ? 8 : 0);
      const avg = (v0 + v1 + v2 + v3) / 4;
      switch (idx) {
        case 0:
        case 15:
          break;
        case 1:
          pushSeg(e3, e0);
          break;
        case 2:
          pushSeg(e0, e1);
          break;
        case 3:
          pushSeg(e3, e1);
          break;
        case 4:
          pushSeg(e1, e2);
          break;
        case 5:
          if (avg >= T) {
            pushSeg(e0, e1);
            pushSeg(e2, e3);
          } else {
            pushSeg(e3, e0);
            pushSeg(e1, e2);
          }
          break;
        case 6:
          pushSeg(e0, e2);
          break;
        case 7:
          pushSeg(e3, e2);
          break;
        case 8:
          pushSeg(e2, e3);
          break;
        case 9:
          pushSeg(e0, e2);
          break;
        case 10:
          if (avg >= T) {
            pushSeg(e0, e3);
            pushSeg(e1, e2);
          } else {
            pushSeg(e0, e1);
            pushSeg(e2, e3);
          }
          break;
        case 11:
          pushSeg(e2, e1);
          break;
        case 12:
          pushSeg(e0, e3);
          break;
        case 13:
          pushSeg(e0, e1);
          break;
        case 14:
          pushSeg(e1, e3);
          break;
        default:
          break;
      }
    }
  }
  return out;
}

function structuralFlowUvInBounds(u, v) {
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
}

function structuralFlowSegmentEscapesManifold(u1, v1, u2, v2) {
  const i1 = structuralFlowUvInBounds(u1, v1);
  const i2 = structuralFlowUvInBounds(u2, v2);
  return !i1 || !i2;
}

function structuralFlowSegmentBoundaryContact(u1, v1, u2, v2, eps = 0.018) {
  const d1 = Math.min(u1, 1 - u1, v1, 1 - v1);
  const d2 = Math.min(u2, 1 - u2, v2, 1 - v2);
  return d1 <= eps || d2 <= eps;
}

/** Dominant contour distance to manifold boundary + crossing/contact ratio for continuous pressure cues. */
function structuralFlowContourBoundaryMetrics(segs) {
  if (!segs.length) return { edgeDistance01: 1, crossingRatio: 0 };
  let minDist = 1;
  let touchOrCross = 0;
  segs.forEach((s) => {
    const [u1, v1, u2, v2] = s;
    const d1 = Math.min(u1, 1 - u1, v1, 1 - v1);
    const d2 = Math.min(u2, 1 - u2, v2, 1 - v2);
    minDist = Math.min(minDist, d1, d2);
    if (structuralFlowSegmentEscapesManifold(u1, v1, u2, v2) || structuralFlowSegmentBoundaryContact(u1, v1, u2, v2)) {
      touchOrCross += 1;
    }
  });
  return {
    edgeDistance01: Math.max(0, Math.min(1, minDist / 0.5)),
    crossingRatio: touchOrCross / segs.length,
  };
}

/** enNorm = relative energy; systemicPressure01 = smooth buildup toward dashed boundary (cyan→amber→red). */
function structuralFlowHeatmapRgba(enNorm, systemicPressure01) {
  const t = Math.max(0, Math.min(1, enNorm));
  const p = Math.max(0, Math.min(1, systemicPressure01));
  const r = Math.floor(8 + t * 22 + p * 22);
  const g = Math.floor(18 + t * 56 + p * 18);
  const b = Math.floor(46 + (1 - t) * 42 + p * 8);
  const a = 0.035 + t * 0.08 + p * 0.04;
  return `rgba(${r},${g},${b},${a})`;
}

function hexToRgba(hex, alpha) {
  const a = Math.max(0, Math.min(1, Number(alpha) || 0));
  const m = String(hex || "").trim().match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
  if (!m) return `rgba(56,189,248,${a})`;
  const r = parseInt(m[1], 16);
  const g = parseInt(m[2], 16);
  const b = parseInt(m[3], 16);
  return `rgba(${r},${g},${b},${a})`;
}

function buildInterpolatedFieldModel(payload, normToPlane, driftDir, riskExp, driftA, instA, driftB, instB, t) {
  const ma = buildStructuralFlowFieldModel(payload, normToPlane, driftDir, driftA, instA, riskExp);
  const mb = buildStructuralFlowFieldModel(payload, normToPlane, driftDir, driftB, instB, riskExp);
  const lerp = (a, b) => a + (b - a) * t;
  const { NX, NY } = ma;
  const E = Array.from({ length: NX + 1 }, () => Array(NY + 1).fill(0));
  let minE = Infinity;
  let maxE = -Infinity;
  for (let i = 0; i <= NX; i += 1) {
    for (let j = 0; j <= NY; j += 1) {
      const e = lerp(ma.E[i][j], mb.E[i][j]);
      E[i][j] = e;
      minE = Math.min(minE, e);
      maxE = Math.max(maxE, e);
    }
  }
  const T = lerp(ma.T, mb.T);
  const { largestMask, E_iso } = structuralFlowBuildIsoFromScalar(E, NX, NY, T, minE);
  return {
    ...ma,
    E,
    E_iso,
    minE,
    maxE,
    T,
    coherence01: lerp(ma.coherence01, mb.coherence01),
    cu: lerp(ma.cu, mb.cu),
    cv: lerp(ma.cv, mb.cv),
    netVx: lerp(ma.netVx, mb.netVx),
    netVy: lerp(ma.netVy, mb.netVy),
    metrics: ma.metrics,
    largestMask,
  };
}

/**
 * Temporal wrapper around the existing structural field builder.
 * Keeps one source of truth for field/contour generation while allowing snapshot interpolation.
 */
function resolveStructuralFlowModelForFrame(payload, normToPlane, driftDir, riskExp, frame) {
  const driftN = Math.max(0, Math.min(1, Number(frame?.driftN ?? 0)));
  const instN = Math.max(0, Math.min(1, Number(frame?.instN ?? 0)));
  const aDrift = Number(frame?.a?.driftN);
  const aInst = Number(frame?.a?.instN);
  const bDrift = Number(frame?.b?.driftN);
  const bInst = Number(frame?.b?.instN);
  const t = Math.max(0, Math.min(1, Number(frame?.t ?? 0)));
  const canInterpolate =
    Number.isFinite(aDrift) &&
    Number.isFinite(aInst) &&
    Number.isFinite(bDrift) &&
    Number.isFinite(bInst) &&
    t > 0 &&
    (Math.abs(aDrift - bDrift) > 1e-6 || Math.abs(aInst - bInst) > 1e-6);
  if (!canInterpolate) {
    return buildStructuralFlowFieldModel(payload, normToPlane, driftDir, driftN, instN, riskExp);
  }
  return buildInterpolatedFieldModel(payload, normToPlane, driftDir, riskExp, aDrift, aInst, bDrift, bInst, t);
}

function getStructuralFlowPlaybackFrame() {
  const g = state.geometry3d;
  const tf = g.temporalFlow || {};
  const points = Array.isArray(tf.points) ? tf.points : [];
  if (!points.length) {
    return { driftN: g.flowDriftN, instN: g.flowInstN, a: null, b: null, t: 0, cursor: -1 };
  }
  const maxPos = Math.max(0, points.length - 1);
  if (state.runDetailView.flowPlaybackMode === "live") {
    tf.localTimeSec = maxPos;
  }
  const pos = Math.max(0, Math.min(maxPos, Number(tf.localTimeSec || 0)));
  const i0 = Math.floor(pos);
  const i1 = Math.min(maxPos, i0 + 1);
  const t = Math.max(0, Math.min(1, pos - i0));
  const a = points[i0];
  const b = points[i1] || a;
  return {
    a,
    b,
    t,
    cursor: i0,
    driftN: (a?.driftN ?? 0) + ((b?.driftN ?? a?.driftN ?? 0) - (a?.driftN ?? 0)) * t,
    instN: (a?.instN ?? 0) + ((b?.instN ?? a?.instN ?? 0) - (a?.instN ?? 0)) * t,
  };
}

/**
 * 2D structural-flow: one force field V → E = |V|² → heatmap + marching-squares contour + E-weighted arrow.
 */
function drawStructuralFlow2dCanvas(ctx, width, height, payload, frame) {
  ctx.fillStyle = "#060d18";
  ctx.fillRect(0, 0, width, height);
  const nodes = payload.nodes || [];
  if (!nodes.length) return;

  const riskExp = structuralFlowRiskExpansion01(payload);

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  nodes.forEach((n) => {
    const x = Number(n.position?.x) || 0;
    const z = Number(n.position?.z) || 0;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  });
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanZ = Math.max(maxZ - minZ, 1e-6);
  const normToPlane = (x, z) => ({
    nx: (x - minX) / spanX,
    nz: (z - minZ) / spanZ,
  });
  const driftDir = structuralFlowDriftBiasDirectionUV(nodes, normToPlane);

  const pad = 28;
  const innerX = pad;
  const innerY = pad;
  const innerW = Math.max(40, width - 2 * pad);
  const innerH = Math.max(40, height - 2 * pad);

  const driftN = Math.max(0, Math.min(1, Number(frame?.driftN ?? 0)));
  const instN = Math.max(0, Math.min(1, Number(frame?.instN ?? 0)));
  const model = resolveStructuralFlowModelForFrame(payload, normToPlane, driftDir, riskExp, frame);
  const { NX, NY, E, E_iso, largestMask, minE, maxE, T, coherence01, cu, cv, netVx, netVy, metrics: m } =
    model;
  const spanE = Math.max(maxE - minE, 1e-12);

  state.geometry3d.structuralFlowCoherence01 = coherence01;

  ctx.strokeStyle = "rgba(51, 65, 85, 0.14)";
  ctx.lineWidth = 1;
  const gs = 10;
  for (let g = 0; g <= gs; g += 1) {
    const gx = innerX + (g / gs) * innerW;
    ctx.beginPath();
    ctx.moveTo(gx, innerY);
    ctx.lineTo(gx, innerY + innerH);
    ctx.stroke();
  }
  for (let g = 0; g <= gs; g += 1) {
    const gy = innerY + (g / gs) * innerH;
    ctx.beginPath();
    ctx.moveTo(innerX, gy);
    ctx.lineTo(innerX + innerW, gy);
    ctx.stroke();
  }

  const segs = structuralFlowMarchingSquaresSegments(E_iso, NX, NY, T);
  const primarySegIdx = structuralFlowPrimarySegmentComponent(segs);
  const primarySegs = [];
  const secondarySegs = [];
  segs.forEach((s, si) => {
    if (primarySegIdx.has(si)) primarySegs.push(s);
    else secondarySegs.push(s);
  });
  const clamp01 = (value) => Math.max(0, Math.min(1, value));
  for (let i = 0; i < NX; i += 1) {
    for (let j = 0; j < NY; j += 1) {
      const inPrimaryMask = largestMask[i][j];
      const cellEnergy = (E[i][j] + E[i + 1][j] + E[i + 1][j + 1] + E[i][j + 1]) / 4;
      const tintedEnergy = inPrimaryMask ? cellEnergy : cellEnergy * 0.08;
      const energyNorm = (tintedEnergy - minE) / spanE;
      const thresholdNorm = clamp01((tintedEnergy - T) / Math.max(spanE * 0.9, 1e-9));
      const edgePressureCell = inPrimaryMask ? 0.14 + 0.26 * thresholdNorm : 0.03;
      const px0 = innerX + (i / NX) * innerW;
      const py0 = innerY + (j / NY) * innerH;
      const pw = innerW / NX;
      const ph = innerH / NY;

      // One final per-cell render path: assign tint once, then paint once.
      ctx.fillStyle = structuralFlowHeatmapRgba(energyNorm * 0.48, edgePressureCell);
      ctx.fillRect(px0, py0, pw + 0.5, ph + 0.5);
    }
  }

  ctx.strokeStyle = "rgba(148, 163, 184, 0.62)";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  ctx.strokeRect(innerX, innerY, innerW, innerH);
  ctx.setLineDash([]);

  const cxPlane = innerX + 0.5 * innerW;
  const cyPlane = innerY + 0.5 * innerH;
  ctx.fillStyle = "rgba(148, 163, 184, 0.55)";
  ctx.beginPath();
  ctx.arc(cxPlane, cyPlane, 3, 0, Math.PI * 2);
  ctx.fill();

  state.geometry3d.structuralFlowDerived = finalizeStructuralFlowDerivedState(model, segs, driftN, instN);
  const derived = state.geometry3d.structuralFlowDerived || {};
  const tf = state.geometry3d.temporalFlow || {};
  const boundaryContacts = [];
  for (let i = 0; i < NX; i += 1) {
    for (let j = 0; j < NY; j += 1) {
      if (!largestMask[i][j]) continue;
      const eC = (E[i][j] + E[i + 1][j] + E[i + 1][j + 1] + E[i][j + 1]) / 4;
      const en = Math.max(0, Math.min(1, (eC - T) / Math.max(spanE * 0.9, 1e-9)));
      const fillA = 0.14 + 0.26 * en;
      const px0 = innerX + (i / NX) * innerW;
      const py0 = innerY + (j / NY) * innerH;
      const pw = innerW / NX;
      const ph = innerH / NY;
      ctx.fillStyle = `rgba(45, 212, 191, ${fillA})`;
      ctx.fillRect(px0, py0, pw + 0.5, ph + 0.5);
    }
  }
  secondarySegs.forEach((s) => {
    const u1 = s[0];
    const v1 = s[1];
    const u2 = s[2];
    const v2 = s[3];
    ctx.beginPath();
    ctx.moveTo(innerX + u1 * innerW, innerY + v1 * innerH);
    ctx.lineTo(innerX + u2 * innerW, innerY + v2 * innerH);
    ctx.strokeStyle = "rgba(94, 234, 212, 0.14)";
    ctx.lineWidth = 1.1;
    ctx.stroke();
  });
  primarySegs.forEach((s) => {
    const u1 = s[0];
    const v1 = s[1];
    const u2 = s[2];
    const v2 = s[3];
    ctx.beginPath();
    ctx.moveTo(innerX + u1 * innerW, innerY + v1 * innerH);
    ctx.lineTo(innerX + u2 * innerW, innerY + v2 * innerH);
    ctx.strokeStyle = "rgba(45, 212, 191, 0.18)";
    ctx.lineWidth = 8;
    ctx.stroke();
  });
  primarySegs.forEach((s) => {
    const u1 = s[0];
    const v1 = s[1];
    const u2 = s[2];
    const v2 = s[3];
    const contact = structuralFlowSegmentBoundaryContact(u1, v1, u2, v2);
    if (contact) boundaryContacts.push([(u1 + u2) * 0.5, (v1 + v2) * 0.5]);
  });
  tf.contactPersistence = Math.max(
    0,
    Math.min(1, (tf.contactPersistence || 0) * 0.93 + (boundaryContacts.length ? 0.08 : -0.05)),
  );
  const breachStage = Math.max(
    0,
    Math.min(1, derived.boundaryPressureScore * 0.5 + tf.contactPersistence * 0.5),
  );
  primarySegs.forEach((s) => {
    const u1 = s[0];
    const v1 = s[1];
    const u2 = s[2];
    const v2 = s[3];
    const esc = structuralFlowSegmentEscapesManifold(u1, v1, u2, v2) && breachStage > 0.62;
    const contact = structuralFlowSegmentBoundaryContact(u1, v1, u2, v2);
    const localFail = esc || (contact && breachStage > 0.28);
    const intensity = Math.max(driftN, instN);
    const badStroke = esc ? "rgba(239, 68, 68, 0.98)" : "rgba(249, 115, 22, 0.95)";
    ctx.beginPath();
    ctx.moveTo(innerX + u1 * innerW, innerY + v1 * innerH);
    ctx.lineTo(innerX + u2 * innerW, innerY + v2 * innerH);
    ctx.strokeStyle = localFail ? badStroke : "rgba(94, 234, 212, 0.94)";
    ctx.lineWidth = localFail ? 2.5 + intensity * 0.9 : 2.25;
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  const cmPx = innerX + cu * innerW;
  const cmPy = innerY + cv * innerH;

  if (state.runDetailView.flowHistoryEnabled) {
    tf.historyCentroid = Array.isArray(tf.historyCentroid) ? tf.historyCentroid : [];
    tf.historyDriftTip = Array.isArray(tf.historyDriftTip) ? tf.historyDriftTip : [];
    tf.historyContact = Array.isArray(tf.historyContact) ? tf.historyContact : [];
    tf.historyCentroid.push({ x: cmPx, y: cmPy });
    if (tf.historyCentroid.length > 26) tf.historyCentroid.shift();
  }
  ctx.fillStyle = "rgba(250, 204, 21, 0.88)";
  ctx.beginPath();
  ctx.arc(cmPx, cmPy, 4, 0, Math.PI * 2);
  ctx.fill();

  const nlen = Math.hypot(netVx, netVy);
  const ndx = nlen > 1e-8 ? netVx / nlen : 0;
  const ndy = nlen > 1e-8 ? netVy / nlen : 1;
  const arrLen = Math.min(innerW, innerH) * (0.08 + 0.36 * Math.min(1, nlen * 4 + driftN * 0.45));
  const ax1 = cxPlane + ndx * arrLen;
  const ay1 = cyPlane + ndy * arrLen;
  if (state.runDetailView.flowHistoryEnabled) {
    tf.historyDriftTip.push({ x: ax1, y: ay1 });
    if (tf.historyDriftTip.length > 20) tf.historyDriftTip.shift();
    boundaryContacts.forEach((c) => {
      tf.historyContact.push({ x: innerX + c[0] * innerW, y: innerY + c[1] * innerH });
    });
    if (tf.historyContact.length > 28) {
      tf.historyContact = tf.historyContact.slice(-28);
    }
  } else {
    tf.historyCentroid = [];
    tf.historyDriftTip = [];
    tf.historyContact = [];
  }

  if (state.runDetailView.flowHistoryEnabled) {
    tf.historyCentroid.forEach((p, idx) => {
      const alpha = ((idx + 1) / tf.historyCentroid.length) * 0.35;
      ctx.fillStyle = `rgba(250, 204, 21, ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.strokeStyle = "rgba(56, 189, 248, 0.24)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    tf.historyDriftTip.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    tf.historyContact.forEach((p, idx) => {
      const alpha = ((idx + 1) / tf.historyContact.length) * 0.48;
      ctx.fillStyle = `rgba(251, 146, 60, ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  ctx.strokeStyle = "rgba(56, 189, 248, 0.72)";
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  ctx.moveTo(cxPlane, cyPlane);
  ctx.lineTo(ax1, ay1);
  ctx.stroke();
  ctx.fillStyle = "rgba(56, 189, 248, 0.92)";
  ctx.beginPath();
  ctx.moveTo(ax1, ay1);
  ctx.lineTo(ax1 - ndx * 8 - ndy * 4, ay1 - ndy * 8 + ndx * 4);
  ctx.lineTo(ax1 - ndx * 8 + ndy * 4, ay1 - ndy * 8 - ndx * 4);
  ctx.closePath();
  ctx.fill();

  nodes.forEach((n) => {
    const px = Number(n.position?.x) || 0;
    const pz = Number(n.position?.z) || 0;
    const { nx, nz } = normToPlane(px, pz);
    const sx = innerX + nx * innerW;
    const sy = innerY + nz * innerH;
    const stress = flowNodeStress01(n);
    const sensorBase = flowNodeColorCss(stress, Boolean(n.is_unstable), n);
    const sensorAlpha = 0.22 + stress * 0.18 + (n.is_unstable ? 0.1 : 0);
    ctx.fillStyle = hexToRgba(sensorBase, Math.max(0.12, Math.min(0.5, sensorAlpha)));
    ctx.beginPath();
    ctx.arc(sx, sy, 1.45 + stress * 1.35 + (n.is_unstable ? 0.35 : 0), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(15, 23, 42, 0.24)";
    ctx.lineWidth = 0.75;
    ctx.stroke();
  });

  ctx.fillStyle = "rgba(226, 232, 240, 0.55)";
  ctx.font = "11px system-ui, Segoe UI, sans-serif";
  const _st = state.geometry3d.structuralFlowDerived;
  const _lbl = _st?.statusLabel || "—";
  const ts = frame?.a?.timestamp ? ` · ${frame.a.timestamp}` : "";
  ctx.fillText(
    `Integrity ${Math.round(coherence01 * 100)}% · ${_lbl} · inside manifold ${m.withinTolerance}/${m.totalSensors}${ts}`,
    innerX + 6,
    innerY + 14,
  );
}

function renderStructuralFlow2dOnly(payload, viewportDims) {
  const canvasHost = qs("#geometryViewport");
  const viewport = qs("#geometryViewport");
  if (!canvasHost || !viewport || !viewportDims) return;
  disposeGeometryRenderer();
  state.runGeometry = payload;
  buildGeometryLegend(payload);
  if (!payload || !payload.available) {
    disposeGeometryRenderer();
    setGeometrySurfaceState(payload?.reason || "Geometry unavailable.", "warn");
    updateGeometryDetails(null);
    return;
  }
  if (!Array.isArray(payload.nodes) || payload.nodes.length === 0) {
    disposeGeometryRenderer();
    setGeometrySurfaceState("No geometry nodes available in this result.", "warn");
    updateGeometryDetails(null);
    return;
  }

  showGeometryCanvas();
  const g = state.geometry3d;
  g.useCanvas2d = true;
  g.perfMode = true;
  g.structuralFlowPlaneLayout = structuralFlowUsesPlaneLayout(payload);
  const { driftN, instN } = flowDriftInstabilityNorm(payload);
  const _m = computeStructuralFlowMetrics(payload);
  g.flowCoherence = _m.coherence01;
  g.flowDriftN = driftN;
  g.flowInstN = instN;
  const width = Math.max(240, viewportDims.width || viewport.clientWidth || 240);
  const height = Math.max(240, viewportDims.height || viewport.clientHeight || 360);
  const cv = document.createElement("canvas");
  cv.className = "geometry-renderer-canvas geometry-flow-2d";
  cv.width = width;
  cv.height = height;
  canvasHost.appendChild(cv);
  g.canvas2d = cv;
  const ctx = cv.getContext("2d", { alpha: false });

  function drawFrame(ts = performance.now()) {
    if (!g.canvas2d || !g.useCanvas2d) return;
    const tf = g.temporalFlow || (g.temporalFlow = {});
    const lastTs = Number(tf.lastTickMs || ts);
    const dt = Math.max(0, Math.min(120, ts - lastTs));
    tf.lastTickMs = ts;
    const points = Array.isArray(tf.points) ? tf.points : [];
    const speed = Math.max(0.1, Number(state.runDetailView.flowPlaybackSpeed || 1));
    if (state.runDetailView.flowPlaybackMode === "replay" && points.length > 1) {
      const stepPerSec = 0.4 * speed;
      tf.localTimeSec = Number(tf.localTimeSec || 0) + (dt / 1000) * stepPerSec;
      if (tf.localTimeSec >= points.length - 1) {
        tf.replayHoldSec = Number(tf.replayHoldSec || 0) + dt / 1000;
        if (tf.replayHoldSec > 0.9) {
          tf.localTimeSec = 0;
          tf.replayHoldSec = 0;
          tf.historyCentroid = [];
          tf.historyDriftTip = [];
          tf.historyContact = [];
          tf.contactPersistence = 0;
        }
      }
    } else if (points.length) {
      tf.localTimeSec = Math.max(0, points.length - 1);
      tf.replayHoldSec = 0;
    }
    const frame = getStructuralFlowPlaybackFrame();
    const dims = geometryViewportDimensions();
    const w = Math.max(240, dims?.width || viewport.clientWidth || 240);
    const h = Math.max(240, dims?.height || viewport.clientHeight || 360);
    drawStructuralFlow2dCanvas(ctx, w, h, payload, frame);
    g.flowCoherence = typeof g.structuralFlowCoherence01 === "number" ? g.structuralFlowCoherence01 : g.flowCoherence;
    updateGeometryFlowPanel(payload, { structuralFlowCoherence01: g.structuralFlowCoherence01 });
    if (frame.cursor !== tf.lastTrendCursor) {
      tf.lastTrendCursor = frame.cursor;
      setTrendPlaybackCursorMarker(frame.cursor, points.length);
    }
    g.frameId = window.requestAnimationFrame(drawFrame);
  }

  function resize2d() {
    const dims = geometryViewportDimensions();
    const w = Math.max(240, dims?.width || viewport.clientWidth || 240);
    const h = Math.max(240, dims?.height || viewport.clientHeight || 360);
    cv.width = w;
    cv.height = h;
    const frame = getStructuralFlowPlaybackFrame();
    drawStructuralFlow2dCanvas(ctx, w, h, payload, frame);
  }

  resize2d();
  g.frameId = window.requestAnimationFrame(drawFrame);
  if (window.ResizeObserver) {
    const ro = new window.ResizeObserver(() => {
      scheduleGeometryResize(g, resize2d);
    });
    ro.observe(canvasHost);
    g.resizeObserver = ro;
  }

  updateGeometryDetails(null);
}

function ensureThreeLibs() {
  const three = getThreeNamespace();
  const controlsCtor = getOrbitControlsConstructor();
  if (!three || !controlsCtor) {
    throw new Error("Three.js did not load (check network or import map).");
  }
  return { three, controlsCtor };
}

function renderGeometryLegend(payload) {
  const legend = qs("#geometryLegend");
  if (!legend) return;
  legend.innerHTML = "";
}

function geometryPositionForNode(node) {
  const source = node?.position_current || node?.position || {};
  return {
    x: Number(source?.x || 0),
    y: Number(source?.y || 0),
    z: Number(source?.z || 0),
  };
}

function geometryStructureSummary(payload) {
  return geometryFlowSummaryString(payload);
}

function clearGeometryModelsPanel() {
  const panel = qs("#geometryModelsPanel");
  const graphDl = qs("#geometryGraphMetricsDl");
  const graphEmpty = qs("#geometryGraphMetricsEmpty");
  const sysDl = qs("#geometrySystemStateDl");
  const sysEmpty = qs("#geometrySystemStateEmpty");
  if (panel) panel.classList.add("hidden");
  if (graphDl) graphDl.innerHTML = "";
  if (sysDl) sysDl.innerHTML = "";
  if (graphEmpty) {
    graphEmpty.classList.add("hidden");
    graphEmpty.textContent = "No graph metrics in this result.";
  }
  if (sysEmpty) {
    sysEmpty.classList.add("hidden");
    sysEmpty.textContent = "No system state snapshot.";
  }
}

function renderGeometryModelsPanel(payload) {
  const panel = qs("#geometryModelsPanel");
  const graphDl = qs("#geometryGraphMetricsDl");
  const graphEmpty = qs("#geometryGraphMetricsEmpty");
  const sysDl = qs("#geometrySystemStateDl");
  const sysEmpty = qs("#geometrySystemStateEmpty");
  if (!panel || !graphDl || !graphEmpty || !sysDl || !sysEmpty) return;

  const ga = payload && payload.graph_analytics;
  const ss = payload && payload.system_state;
  const hasGraph =
    ga &&
    typeof ga === "object" &&
    (Boolean(ga.correlation_graph) || Boolean(ga.causal_graph));
  const hasSys = ss && typeof ss === "object" && Object.keys(ss).length > 0;

  if (!hasGraph && !hasSys) {
    const hasStructuralPayload =
      payload &&
      (payload.available === true ||
        (Array.isArray(payload.nodes) && payload.nodes.length > 0) ||
        (payload.graph && typeof payload.graph === "object"));
    if (state.demo.enabled && hasStructuralPayload) {
      panel.classList.remove("hidden");
      graphEmpty.classList.remove("hidden");
      graphEmpty.textContent =
        "Viz loading — full graph analytics are computed server-side. Use structural drift trend above for progression.";
      graphDl.innerHTML = "";
      sysEmpty.classList.remove("hidden");
      sysEmpty.textContent =
        "System state snapshot not returned in this build — the 3D structural flow still reflects live telemetry geometry.";
      sysDl.innerHTML = "";
      return;
    }
    clearGeometryModelsPanel();
    return;
  }

  panel.classList.remove("hidden");

  if (!hasGraph) {
    graphEmpty.classList.remove("hidden");
    graphDl.innerHTML = "";
  } else {
    graphEmpty.classList.add("hidden");
    const rows = [];
    const cg = ga.correlation_graph;
    if (cg && typeof cg === "object") {
      rows.push(
        ["Correlation graph · mean degree", toPretty(cg.mean_degree)],
        ["Correlation graph · density", toPretty(cg.density)],
        ["Correlation graph · clustering", toPretty(cg.clustering)],
        ["Correlation graph · connectivity", toPretty(cg.connectivity)],
        ["Correlation graph · mean |corr| off-diag", toPretty(cg.mean_absolute_connectivity)]
      );
    }
    const cag = ga.causal_graph;
    if (cag && typeof cag === "object") {
      const labels = Array.isArray(cag.dominant_source_labels) ? cag.dominant_source_labels.filter(Boolean) : [];
      const labelText = labels.length ? labels.join(", ") : "-";
      rows.push(
        ["Causal graph · density", toPretty(cag.density)],
        ["Causal graph · asymmetry", toPretty(cag.asymmetry)],
        ["Causal graph · dominant sources", escapeHtml(labelText)]
      );
    }
    graphDl.innerHTML = rows
      .map(([dt, dd]) => `<div><dt>${escapeHtml(String(dt))}</dt><dd>${escapeHtml(String(dd))}</dd></div>`)
      .join("");
  }

  if (!hasSys) {
    sysEmpty.classList.remove("hidden");
    sysDl.innerHTML = "";
  } else {
    sysEmpty.classList.add("hidden");
    const rm = ss.regime_memory && typeof ss.regime_memory === "object" ? ss.regime_memory : {};
    const rs = ss.regime_signature && typeof ss.regime_signature === "object" ? ss.regime_signature : {};
    const nr = ss.nearest_regime && typeof ss.nearest_regime === "object" ? ss.nearest_regime : null;
    const sysRows = [
      ["Phase", toPretty(ss.phase)],
      ["Interpreted state", toPretty(ss.interpreted_state)],
      ["Regime name", toPretty(ss.regime_name)],
      ["Confidence", toPretty(ss.confidence)],
      ["Structural analysis", ss.structural_analysis_available ? "available" : "limited"],
      ["Regime library size", toPretty(rs.library_size != null ? rs.library_size : rm.library_size)],
      ["Regime memory entries", toPretty(rm.baseline_count)],
      ["Assigned regime (signature)", toPretty(rs.assigned_name)],
      ["Regime distance", ss.regime_distance != null && ss.regime_distance !== undefined ? toPretty(ss.regime_distance) : "-"],
    ];
    if (nr && nr.name != null) {
      sysRows.push(["Nearest regime (library)", `${String(nr.name)} · ${toPretty(nr.distance)}`]);
    }
    sysDl.innerHTML = sysRows
      .map(([dt, dd]) => `<div><dt>${escapeHtml(String(dt))}</dt><dd>${escapeHtml(String(dd))}</dd></div>`)
      .join("");
  }
}

function applyGeometryDisplayMode() {
  const g = state.geometry3d;
  Object.entries(g.nodeMeshById || {}).forEach(([nodeId, mesh]) => {
    const node = g.nodeDataById[nodeId];
    if (!node || !mesh) return;
    const pos = geometryPositionForNode(node);
    if (!mesh.userData) mesh.userData = {};
    const three = getThreeNamespace();
    if (!three?.Vector3) return;
    mesh.userData.basePos = new three.Vector3(pos.x, pos.y, pos.z);
    mesh.position.set(pos.x, pos.y, pos.z);
    const label = g.nodeLabelById[nodeId];
    if (label) {
      const radius = Number(mesh.userData.radius || 0.05);
      if (label.parent === mesh) {
        label.position.set(0, radius + 0.092, 0);
      } else {
        label.position.set(pos.x, pos.y + radius + 0.08, pos.z);
      }
    }
    const halo = g.nodeGlowById[nodeId];
    if (halo) halo.position.set(pos.x, pos.y, pos.z);
  });
  const threeNs = getThreeNamespace();
  syncFlowEdgesFromMeshes(g, threeNs, 0);
  const note = qs("#geometryProjectionNote");
  const payload = state.runGeometry;
  if (note && payload) {
    note.textContent = structuralProjectionCaption(payload);
  }
  const summary = qs("#geometryStructureSummary");
  if (summary) summary.textContent = geometryStructureSummary(state.runGeometry);
  updateGeometryFlowPanel(payload);
  if (g.camera && g.controls && g.nodeGroup && threeNs) {
    fitGeometryCamera(g.camera, g.controls, g.nodeGroup, threeNs, {
      structuralFlowPlaneLayout: g.structuralFlowPlaneLayout,
    });
    updateGeometryShadowFrustum(g, threeNs);
    if (g.structuralFlowPlaneLayout && g.camera && g.controls) {
      g.camera.position.y += 0.38;
      g.camera.position.x += 0.26;
      g.camera.position.z += 0.14;
      g.camera.lookAt(g.controls.target);
    }
  }
}

function updateGeometryDetails(nodeId = null) {
  const payload = state.runGeometry;
  const g = state.geometry3d;
  const nodeLabel = qs("#geometryNodeLabel");
  const nodeStress = qs("#geometryNodeStress");
  const nodeMagnitude = qs("#geometryNodeMagnitude");
  const nodeState = qs("#geometryNodeState");
  const metricEnginePhase = qs("#geometryEnginePhase");
  const metricRisk = qs("#geometryRisk");
  const metricDrift = qs("#geometryDrift");
  const metricComposite = qs("#geometryComposite");

  if (metricEnginePhase) metricEnginePhase.textContent = toPretty(payload?.metrics?.state);
  if (metricRisk) metricRisk.textContent = toPretty(payload?.metrics?.risk_level);
  if (metricDrift) metricDrift.textContent = toPretty(payload?.metrics?.structural_drift_score);
  if (metricComposite) metricComposite.textContent = toPretty(payload?.metrics?.composite_instability);
  updateGeometryFlowPanel(payload);

  const setNodeFields = (label, stress, magnitude, stateText) => {
    if (nodeLabel) nodeLabel.textContent = label;
    if (nodeStress) nodeStress.textContent = stress;
    if (nodeMagnitude) nodeMagnitude.textContent = magnitude;
    if (nodeState) nodeState.textContent = stateText;
  };

  if (!payload || !payload.available) {
    setNodeFields("—", "—", "—", "—");
    return;
  }
  if (!nodeId) {
    setNodeFields("—", "—", "—", "—");
    return;
  }
  const node = g.nodeDataById[nodeId];
  if (!node) {
    setNodeFields("None selected", "-", "-", "-");
    return;
  }
  const stressVal = node.stress;
  const magVal = node.magnitude;
  const stressText =
    stressVal != null && stressVal !== "" && Number.isFinite(Number(stressVal)) ? toPretty(Number(stressVal)) : "—";
  const magText =
    magVal != null && magVal !== "" && Number.isFinite(Number(magVal)) ? toPretty(Number(magVal)) : "—";
  if (nodeLabel) {
    nodeLabel.title =
      "Telemetry channel name for this run (same id as ingest / sensor_relationships when present).";
  }
  if (nodeStress) {
    nodeStress.title =
      "Normalized structural stress (0–1) from correlation geometry: row-wise deviation versus baseline (or importance when baseline is unavailable).";
  }
  if (nodeMagnitude) {
    nodeMagnitude.title =
      "Coupling (0–1): mean absolute off-diagonal correlation — how strongly this channel moves with others in the projection.";
  }
  if (nodeState) {
    nodeState.title = "Watch / stable / critical band from normalized stress; UNSTABLE marks critical.";
  }
  const catalogNote =
    node.in_correlation_window === false
      ? " · outside engine correlation window (listed for asset context)"
      : "";
  setNodeFields(
    String(node.label || node.id || "-"),
    stressText,
    magText,
    `${String(node.state || "-")}${node.is_unstable ? " (UNSTABLE)" : ""}${catalogNote}`,
  );
}

function resetNodeMeshVisual(mesh) {
  const shell = mesh?.userData?.shellMesh || mesh;
  const root = mesh?.userData?.shellMesh ? mesh : mesh.parent;
  if (!shell?.material) return;
  if (root && root.scale) root.scale.setScalar(1);
  const em = shell.material.emissive;
  if (shell.userData && shell.userData.flowColor && shell.material.color && em) {
    const k = shell.userData.emissiveMul != null ? shell.userData.emissiveMul : 0.12;
    const inten = shell.userData.emissiveIntensityBase != null ? shell.userData.emissiveIntensityBase : 0.35;
    shell.material.color.copy(shell.userData.flowColor);
    em.copy(shell.userData.flowColor).multiplyScalar(k);
    shell.material.emissiveIntensity = inten;
  } else if (em) {
    em.setHex(0x0d1424);
    shell.material.emissiveIntensity = 0.35;
  }
}

function setGeometrySelection(nextId = null) {
  const g = state.geometry3d;
  if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
    resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
  }
  g.hoveredId = null;
  const previous = g.selectedId;
  if (previous && g.nodeMeshById[previous]) {
    resetNodeMeshVisual(g.nodeMeshById[previous]);
  }
  g.selectedId = nextId;
  if (nextId && g.nodeMeshById[nextId]) {
    const root = g.nodeMeshById[nextId];
    const shell = root.userData?.shellMesh || root;
    root.scale.setScalar(1.14);
    if (shell.material && shell.material.emissive) {
      shell.material.emissive.setHex(0x1e4a68);
      shell.material.emissiveIntensity = 0.78;
    }
  }
  updateGeometryDetails(g.selectedId);
}

function buildGeometryLegend(payload) {
  const details = qs("#geometryDetails");
  if (!details) return;
  if (!payload || !payload.available) {
    details.setAttribute("data-geometry-risk", "UNKNOWN");
    renderGeometryLegend(payload);
    return;
  }
  const assessment = deriveStructuralAssessment(computeStructuralFlowMetrics(payload));
  details.setAttribute(
    "data-geometry-risk",
    assessment.tone || normalizeRiskLevel(payload.metrics?.risk_level),
  );
  renderGeometryLegend(payload);
}

function geometryNodeLabelShort(node) {
  const raw = String(node?.label || node?.id || "?");
  const max = 18;
  if (raw.length <= max) return raw;
  return `${raw.slice(0, max - 1)}…`;
}

function createNodeLabelSprite(three, text, colorHex = 0xd9e6ff) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  canvas.width = 256;
  canvas.height = 64;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(8, 16, 30, 0.72)";
  ctx.strokeStyle = "rgba(86, 119, 176, 0.86)";
  ctx.lineWidth = 2;
  const x = 4;
  const y = 8;
  const w = 248;
  const h = 48;
  const r = 8;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, "0")}`;
  ctx.font = "600 22px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(text || ""), canvas.width / 2, canvas.height / 2);
  const texture = new three.CanvasTexture(canvas);
  texture.needsUpdate = true;
  if (three.SRGBColorSpace) {
    texture.colorSpace = three.SRGBColorSpace;
  }
  const material = new three.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new three.Sprite(material);
  sprite.scale.set(0.58, 0.145, 1);
  sprite.userData.pickMesh = false;
  return sprite;
}

function beginGeometryIntroAnimation() {
  const g = state.geometry3d;
  const three = getThreeNamespace();
  if (!three?.Vector3) return;
  const reduceGeomMotion = geometryFlowReducedMotion();
  Object.values(g.nodeMeshById || {}).forEach((mesh) => {
    if (!mesh.userData?.basePos) return;
    const bp = mesh.userData.basePos;
    if (!mesh.userData.geomIntroFrom) mesh.userData.geomIntroFrom = new three.Vector3();
    if (!mesh.userData.geomIntroFromPreset) {
      mesh.userData.geomIntroFrom.set(bp.x * 0.06, bp.y * 0.075 + 0.012, bp.z * 0.06);
    }
    delete mesh.userData.geomIntroFromPreset;
    mesh.position.copy(mesh.userData.geomIntroFrom);
    const nid = mesh.userData.nodeId != null ? String(mesh.userData.nodeId) : "";
    const halo = nid ? g.nodeGlowById[nid] : null;
    if (halo) halo.position.copy(mesh.position);
  });
  if (three) syncFlowEdgesFromMeshes(g, three, 0);
  g.geometryIntroStart = performance.now();
  g.geometryIntroMs = reduceGeomMotion ? 160 : 920;
}

function renderGeometryScene(payload, viewportDims) {
  const canvasHost = qs("#geometryViewport");
  const viewport = qs("#geometryViewport");
  if (!canvasHost || !viewport || !viewportDims) return;
  if (geometryFlowPrefer2dOnly()) {
    renderStructuralFlow2dOnly(payload, viewportDims);
    return;
  }
  let threeLib;
  try {
    threeLib = ensureThreeLibs();
  } catch (_err) {
    renderStructuralFlow2dOnly(payload, viewportDims);
    return;
  }
  const { three, controlsCtor } = threeLib;
  const prevDisplayPos = captureGeometryNodeDisplayPositions();
  disposeGeometryRenderer();
  state.runGeometry = payload;
  buildGeometryLegend(payload);
  if (!payload || !payload.available) {
    disposeGeometryRenderer();
    setGeometrySurfaceState(payload?.reason || "Geometry unavailable.", "warn");
    updateGeometryDetails(null);
    return;
  }
  if (!Array.isArray(payload.nodes) || payload.nodes.length === 0) {
    disposeGeometryRenderer();
    setGeometrySurfaceState("No geometry nodes available in this result.", "warn");
    updateGeometryDetails(null);
    return;
  }

  if (typeof console !== "undefined" && console.debug) {
    console.debug("[geometry3d] build scene", {
      nodes: payload.nodes.length,
      edges: Array.isArray(payload.edges) ? payload.edges.length : 0,
      sampleNode: payload.nodes[0],
    });
  }

  showGeometryCanvas();
  const perf = geometryFlowPerfEnabled();
  const width = Math.max(240, viewportDims.width || viewport.clientWidth || 240);
  const height = Math.max(240, viewportDims.height || viewport.clientHeight || 360);
  const renderer = new three.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
    stencil: false,
    depth: true,
  });
  const dprCap = perf ? 1.5 : 2;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, dprCap));
  renderer.setSize(width, height);
  if (three.SRGBColorSpace) {
    renderer.outputColorSpace = three.SRGBColorSpace;
  } else if (typeof renderer.outputEncoding !== "undefined" && typeof three.sRGBEncoding !== "undefined") {
    renderer.outputEncoding = three.sRGBEncoding;
  }
  if (three.ACESFilmicToneMapping !== undefined) {
    renderer.toneMapping = three.ACESFilmicToneMapping;
    renderer.toneMappingExposure = perf ? 0.94 : 1.06;
  }
  renderer.shadowMap.enabled = !perf;
  if (three.PCFSoftShadowMap) {
    renderer.shadowMap.type = three.PCFSoftShadowMap;
  }
  renderer.domElement.className = "geometry-renderer-canvas";
  canvasHost.appendChild(renderer.domElement);
  const canvasFlowWrap = qs("#geometryCanvasWrap");
  if (canvasFlowWrap) {
    canvasFlowWrap.style.opacity = "0.22";
  }

  const scene = new three.Scene();
  scene.background = null;
  scene.fog = new three.FogExp2(perf ? 0x060a14 : 0x0a1424, perf ? 0.05 : 0.034);

  const camera = new three.PerspectiveCamera(44, width / height, 0.01, 60);
  camera.position.set(0, 0.45, 3.05);
  scene.add(camera);

  const hemi = new three.HemisphereLight(0x9ec5f0, 0x0a1528, perf ? 0.64 : 0.52);
  scene.add(hemi);
  const key = new three.DirectionalLight(0xd8ecff, perf ? 0.68 : 1.02);
  key.position.set(2.4, 3.6, 2.2);
  if (!perf) {
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.bias = -0.00025;
    key.shadow.normalBias = 0.025;
  }
  scene.add(key);
  scene.add(key.target);
  const fill = new three.DirectionalLight(0x2dd4bf, perf ? 0.26 : 0.38);
  fill.position.set(-2.2, 1.2, -1.8);
  scene.add(fill);
  const rim = new three.PointLight(0x38bdf8, perf ? 0.34 : 0.52, 9, 2);
  rim.position.set(0, 0.85, 0);
  scene.add(rim);

  const geomDebug = DEBUG_GEOMETRY || geometryFlowDebugEnabled();
  if (geomDebug) {
    const testSphere = new three.Mesh(
      new three.SphereGeometry(0.14, 16, 16),
      new three.MeshBasicMaterial({ color: 0xff2222 })
    );
    testSphere.position.set(0, 0.1, 0);
    testSphere.name = "geomDebugMarker";
    scene.add(testSphere);
  }

  const controls = new controlsCtor(camera, renderer.domElement);
  controls.enableDamping = !perf;
  controls.dampingFactor = perf ? 0.05 : 0.08;
  controls.rotateSpeed = perf ? 0.38 : 0.55;
  controls.zoomSpeed = perf ? 0.55 : 0.75;
  controls.panSpeed = perf ? 0.45 : 0.55;
  controls.minDistance = 1.15;
  controls.maxDistance = 6.5;
  controls.target.set(0, 0, 0);
  const reduceGeomMotion = geometryFlowReducedMotion();
  const narrowVp = geometryViewportIsNarrow();
  controls.autoRotate = !reduceGeomMotion;
  controls.autoRotateSpeed = perf ? 0.32 : narrowVp ? 0.5 : 0.62;

  const g = state.geometry3d;
  g.renderer = renderer;
  g.scene = scene;
  g.camera = camera;
  g.controls = controls;
  g.raycaster = new three.Raycaster();
  g.pointer = new three.Vector2();
  g.nodeMeshById = {};
  g.nodeDataById = {};
  g.nodeLabelById = {};
  g.nodeGlowById = {};
  g.unstablePulseById = {};
  g.edgeRefs = [];
  g.perfMode = perf;
  {
    const nc = Array.isArray(payload.nodes) ? payload.nodes.length : 0;
    const ec = Array.isArray(payload.edges) ? payload.edges.length : 0;
    g.curveSegments = perf ? 7 : ec > 22 ? 9 : nc > 16 ? 10 : 12;
  }
  let motionScale = perf ? 0.72 : 1;
  if (geometryFlowReducedMotion()) motionScale *= 0.42;
  g.motionScale = motionScale;
  g.useCanvas2d = false;
  g.pendingHoverEvt = null;
  g.hoverRaf = null;
  const { driftN, instN } = flowDriftInstabilityNorm(payload);
  const _flowM = computeStructuralFlowMetrics(payload);
  g.flowCoherence = _flowM.coherence01;
  g.flowDriftN = driftN;
  g.flowInstN = instN;
  g.selectedId = null;
  g.hoveredId = null;
  g.interactionEnabled = true;
  g.keyLight = key;
  g.keyLightBasePos = key.position.clone();
  g.groundMesh = null;
  g.structuralFlowPlaneLayout = structuralFlowUsesPlaneLayout(payload);
  g.reducedGeometryMotion = reduceGeomMotion;

  const L2 = getFlowLine2Helpers();
  const flowUseLine2 = Boolean(L2.Line2 && L2.LineGeometry && L2.LineMaterial);
  g.flowUseLine2 = flowUseLine2;

  const edgeGroup = new three.Group();
  const coherence0 = g.flowCoherence;
  const divisions0 = g.curveSegments;
  const pep = perf ? 2 : 3;
  g.flowParticlesPerEdge = pep;

  (payload.edges || []).forEach((edge, edgeIdx) => {
    const source = payload.nodes.find((n) => n.id === edge.source);
    const target = payload.nodes.find((n) => n.id === edge.target);
    if (!source || !target) return;
    const p1 = new three.Vector3(Number(source.position.x), Number(source.position.y), Number(source.position.z));
    const p2 = new three.Vector3(Number(target.position.x), Number(target.position.y), Number(target.position.z));
    const mid = p1.clone().add(p2).multiplyScalar(0.5);
    mid.y += 0.05 * coherence0 * (1 - flowEdgeTension(edge) * 0.85);
    const curve = new three.QuadraticBezierCurve3(p1.clone(), mid, p2.clone());
    const pts = curve.getPoints(divisions0);
    const flat = [];
    pts.forEach((p) => flat.push(p.x, p.y, p.z));
    const tension = flowEdgeTension(edge);
    const ec = new three.Color(0x5eead4);
    ec.lerp(new three.Color(0xfb7185), tension * (1.2 - coherence0 * 0.5));
    const mag = Math.min(1, Math.abs(Number(edge.magnitude || 0)));
    const edgeGeom = new three.BufferGeometry().setFromPoints(pts);
    const edgeMat = new three.LineBasicMaterial({
      color: ec,
      transparent: true,
      opacity: perf ? 0.82 : 0.26 + 0.5 * mag,
      depthWrite: false,
    });
    const line = new three.Line(edgeGeom, edgeMat);
    const glowMat = new three.LineBasicMaterial({
      color: perf ? 0x5ecfff : 0x7ee8ff,
      transparent: true,
      opacity: perf ? 0.24 : 0.36,
      blending: three.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    });
    const lineGlow = new three.Line(edgeGeom, glowMat);
    lineGlow.renderOrder = 2;

    let line2Main = null;
    let line2Glow = null;
    let lineGeomMain = null;
    let lineGeomGlow = null;
    let lineMatMain = null;
    let lineMatGlow = null;

    if (flowUseLine2) {
      line.visible = false;
      lineGlow.visible = false;
      lineGeomMain = new L2.LineGeometry();
      lineGeomMain.setPositions(flat);
      lineGeomGlow = new L2.LineGeometry();
      lineGeomGlow.setPositions(flat);
      lineMatMain = new L2.LineMaterial({
        color: ec.getHex(),
        linewidth: perf ? 0.0055 : 0.011,
        worldUnits: true,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
      });
      lineMatMain.resolution.set(width, height);
      lineMatGlow = new L2.LineMaterial({
        color: new three.Color(0x6ee0ff).getHex(),
        linewidth: perf ? 0.014 : 0.028,
        worldUnits: true,
        transparent: true,
        opacity: 0.48,
        blending: three.AdditiveBlending,
        depthWrite: false,
      });
      lineMatGlow.resolution.set(width, height);
      line2Main = new L2.Line2(lineGeomMain, lineMatMain);
      line2Main.computeLineDistances();
      line2Main.renderOrder = 1;
      line2Glow = new L2.Line2(lineGeomGlow, lineMatGlow);
      line2Glow.computeLineDistances();
      line2Glow.renderOrder = 2;
      edgeGroup.add(line2Main);
      edgeGroup.add(line2Glow);
    } else {
      edgeGroup.add(line);
      edgeGroup.add(lineGlow);
    }

    const slotStart = g.edgeRefs.length * pep;
    g.edgeRefs.push({
      line,
      lineGlow,
      line2Main,
      line2Glow,
      lineGeomMain,
      lineGeomGlow,
      lineMatMain,
      lineMatGlow,
      sourceId: String(edge.source),
      targetId: String(edge.target),
      edge,
      edgeIdx,
      divisions: divisions0,
      particleSlotStart: slotStart,
    });
  });

  g.flowUseStructuralTubes =
    !flowUseLine2 &&
    !perf &&
    g.edgeRefs.length > 0 &&
    g.edgeRefs.length <= 28 &&
    Boolean(three.TubeGeometry);

  const flowCount = g.edgeRefs.length * pep;
  g.flowDummy = null;
  g.flowInstancedMesh = null;
  if (flowCount > 0) {
    const flowMesh = createStructuralFlowParticles(three, flowCount, perf);
    if (flowMesh) {
      g.flowInstancedMesh = flowMesh;
      g.flowDummy = new three.Object3D();
      edgeGroup.add(flowMesh);
    }
  }

  g.edgeGroup = edgeGroup;

  const nodeGroup = new three.Group();
  payload.nodes.forEach((node) => {
    const magnitude = Math.max(0, Number(node.magnitude || 0));
    const stress01 = flowNodeStress01(node);
    const inRange = flowNodeInRange(node);
    const importance = Math.min(
      1,
      0.32 + magnitude * 0.42 + stress01 * 0.28 + (inRange ? 0 : 0.14),
    );
    const radius = 0.034 + importance * (g.structuralFlowPlaneLayout ? 0.068 : 0.074);
    const col = flowNodeColorTHREE(
      three,
      g.structuralFlowPlaneLayout && inRange ? Math.min(stress01, 0.38) : stress01,
      Boolean(node.is_unstable),
      node,
    );
    const assembly = createStructuralSensorAssembly(three, {
      nodeId: node.id,
      col,
      radius,
      stress01,
      inRange,
      perf,
    });
    const px = Number(node.position?.x || 0);
    const py = Number(node.position?.y || 0);
    const pz = Number(node.position?.z || 0);
    const shell = assembly.userData.shellMesh;
    assembly.userData.radius = radius;
    assembly.userData.basePos = new three.Vector3(px, py, pz);
    assembly.userData.geomIntroFrom = new three.Vector3();
    const prevSnap = prevDisplayPos[String(node.id)];
    if (prevSnap) {
      assembly.userData.geomIntroFrom.set(prevSnap.x, prevSnap.y, prevSnap.z);
      assembly.userData.geomIntroFromPreset = true;
    } else {
      assembly.userData.geomIntroFromPreset = false;
    }
    assembly.position.set(px, py, pz);
    assembly.userData.jitterSeed = (String(node.id).length * 0.37 + px * 0.12 + pz * 0.09) % (Math.PI * 2);
    assembly.userData.perfMode = perf;
    if (!perf) {
      shell.castShadow = true;
      shell.receiveShadow = true;
    }
    nodeGroup.add(assembly);
    g.nodeMeshById[String(node.id)] = assembly;
    g.nodeDataById[String(node.id)] = node;
    g.unstablePulseById[String(node.id)] =
      0.65 + structuralFlowHash01(`pulse:${node.id}:${px}:${py}:${pz}`) * 0.7;
    const lab = createNodeLabelSprite(three, geometryNodeLabelShort(node), 0xcfe4ff);
    if (lab) {
      lab.position.set(0, radius + 0.092, 0);
      lab.scale.set(perf ? 0.48 : 0.58, perf ? 0.12 : 0.145, 1);
      lab.userData.pickMesh = false;
      assembly.add(lab);
      g.nodeLabelById[String(node.id)] = lab;
    }
  });
  nodeGroup.updateMatrixWorld(true);
  const boxNg = new three.Box3().setFromObject(nodeGroup);
  if (!boxNg.isEmpty()) {
    const center = boxNg.getCenter(new three.Vector3());
    const field = createStructuralFieldPlane(three, center, boxNg, perf);
    if (field) {
      scene.add(field);
      g.structuralFieldMesh = field;
    }
    if (!perf) {
      const size = boxNg.getSize(new three.Vector3());
      const planeW = Math.max(10, Math.max(size.x, size.z) * 2.05);
      const planeGeom = new three.PlaneGeometry(planeW, planeW, 1, 1);
      const planeMat = new three.MeshStandardMaterial({
        color: 0x080f18,
        metalness: 0.06,
        roughness: 0.96,
        envMapIntensity: 0.42,
        emissive: 0x020508,
        emissiveIntensity: 0.06,
        transparent: true,
        opacity: 0.94,
      });
      const ground = new three.Mesh(planeGeom, planeMat);
      ground.rotation.x = -Math.PI / 2;
      ground.position.set(center.x, boxNg.min.y - 0.04, center.z);
      ground.receiveShadow = true;
      ground.userData.pickMesh = false;
      ground.renderOrder = -3;
      scene.add(ground);
      g.groundMesh = ground;
    }
  }
  const geometryRig = new three.Group();
  geometryRig.name = "geometryRig";
  if (!boxNg.isEmpty()) {
    const datumShell = createSensorDatumShell(three, boxNg, perf);
    if (datumShell) {
      geometryRig.add(datumShell);
      g.sensorDatumShell = datumShell;
    }
  }
  geometryRig.add(edgeGroup);
  geometryRig.add(nodeGroup);
  scene.add(geometryRig);
  g.nodeGroup = nodeGroup;
  g.geometryRig = geometryRig;
  applyFlowLineResolutionToEdges(g, width, height);
  applyGeometryDisplayMode();
  beginGeometryIntroAnimation();
  updateGeometryDetails(null);

  function onResize() {
    if (!g.renderer || !g.camera) return;
    const dims = geometryViewportDimensions();
    const w = Math.max(240, dims?.width || viewport.clientWidth || 240);
    const h = Math.max(240, dims?.height || viewport.clientHeight || 360);
    const cap = g.perfMode ? 1.5 : 2;
    g.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, cap));
    g.renderer.setSize(w, h);
    g.camera.aspect = w / h;
    g.camera.updateProjectionMatrix();
    applyFlowLineResolutionToEdges(g, w, h);
  }

  if (window.ResizeObserver) {
    const ro = new window.ResizeObserver(() => {
      scheduleGeometryResize(g, onResize);
    });
    ro.observe(canvasHost);
    g.resizeObserver = ro;
  } else {
    const handler = () => scheduleGeometryResize(g, onResize);
    window.addEventListener("resize", handler);
    g.cleanupResize = () => window.removeEventListener("resize", handler);
  }

  let pointerDown = null;

  function updateHoverFromEventImpl(evt) {
    if (!g.interactionEnabled || !g.raycaster || !g.camera || !g.renderer || !g.scene || !g.nodeGroup) return;
    if (g.selectedId) return;
    if (pointerDown) return;
    g.scene.updateMatrixWorld(true);
    const rect = g.renderer.domElement.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    g.pointer.x = ((evt.clientX - rect.left) / w) * 2 - 1;
    g.pointer.y = -((evt.clientY - rect.top) / h) * 2 + 1;
    g.raycaster.setFromCamera(g.pointer, g.camera);
    const hits = g.raycaster.intersectObjects([g.nodeGroup], true);
    const hit = hits.find((h) => h.object && h.object.userData && h.object.userData.pickMesh === true);
    const nextHover = hit && hit.object.userData.nodeId ? String(hit.object.userData.nodeId) : null;
    if (nextHover === g.hoveredId) return;
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
    }
    g.hoveredId = nextHover;
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      const hm = g.nodeMeshById[g.hoveredId];
      const shell = hm?.userData?.shellMesh || hm;
      if (shell.material && shell.material.emissive) {
        shell.material.emissive.setHex(0x2a4a66);
        shell.material.emissiveIntensity = 0.48;
      }
    }
    const tip = g.hoveredId && g.nodeDataById[g.hoveredId];
    g.renderer.domElement.title = tip ? String(tip.label || tip.id || g.hoveredId || "") : "";
  }

  function scheduleHoverFromEvent(evt) {
    g.pendingHoverEvt = evt;
    if (g.hoverRaf) return;
    g.hoverRaf = window.requestAnimationFrame(() => {
      g.hoverRaf = null;
      const ev = g.pendingHoverEvt;
      g.pendingHoverEvt = null;
      if (ev) updateHoverFromEventImpl(ev);
    });
  }

  function clearHover() {
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
    }
    g.hoveredId = null;
    if (g.renderer && g.renderer.domElement) {
      g.renderer.domElement.title = "";
    }
  }

  function pickNodeFromEvent(evt) {
    if (!g.interactionEnabled || !g.raycaster || !g.camera || !g.renderer || !g.scene) return;
    if (!g.nodeGroup) return;
    g.scene.updateMatrixWorld(true);
    const rect = g.renderer.domElement.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    g.pointer.x = ((evt.clientX - rect.left) / w) * 2 - 1;
    g.pointer.y = -((evt.clientY - rect.top) / h) * 2 + 1;
    g.raycaster.setFromCamera(g.pointer, g.camera);
    const hits = g.raycaster.intersectObjects([g.nodeGroup], true);
    if (typeof console !== "undefined" && console.debug) {
      console.debug("[geometry3d] pick ray", {
        pointer: [g.pointer.x, g.pointer.y],
        hitCount: hits.length,
        firstPick: hits[0]?.object?.userData,
      });
    }
    const hit = hits.find((h) => h.object && h.object.userData && h.object.userData.pickMesh === true);
    const id = hit && hit.object.userData.nodeId ? String(hit.object.userData.nodeId) : null;
    if (typeof console !== "undefined" && console.debug) {
      console.debug("[geometry3d] pick result", { selectedId: id });
    }
    setGeometrySelection(id);
  }

  const onPointerDown = (evt) => {
    if (evt.button !== 0) return;
    pointerDown = { x: evt.clientX, y: evt.clientY };
  };
  const onPointerUp = (evt) => {
    if (evt.button !== 0 || !pointerDown) return;
    const dx = evt.clientX - pointerDown.x;
    const dy = evt.clientY - pointerDown.y;
    pointerDown = null;
    if (Math.hypot(dx, dy) > 10) return;
    pickNodeFromEvent(evt);
  };
  const onPointerMove = (evt) => {
    scheduleHoverFromEvent(evt);
  };
  const onPointerLeave = () => {
    clearHover();
    pointerDown = null;
  };

  const el = renderer.domElement;
  el.addEventListener("pointerdown", onPointerDown);
  el.addEventListener("pointerup", onPointerUp);
  el.addEventListener("pointermove", onPointerMove);
  el.addEventListener("pointerleave", onPointerLeave);
  g.cleanupPointer = () => {
    el.removeEventListener("pointerdown", onPointerDown);
    el.removeEventListener("pointerup", onPointerUp);
    el.removeEventListener("pointermove", onPointerMove);
    el.removeEventListener("pointerleave", onPointerLeave);
  };

  let t = 0;
  let geometryFrameIndex = 0;
  function animate() {
    if (g.useCanvas2d) return;
    g.frameId = window.requestAnimationFrame(animate);
    try {
      t += 0.014;
      const introMs = g.geometryIntroMs || 0;
      let introProg = 1;
      if (introMs > 0 && g.geometryIntroStart != null) {
        introProg = easeInOutCubic((performance.now() - g.geometryIntroStart) / introMs);
        if (introProg >= 1) {
          g.geometryIntroMs = 0;
          g.geometryIntroStart = null;
        }
      }
      const introActive = introProg < 1 && introMs > 0;

      const canvasWrapEl = qs("#geometryCanvasWrap");
      if (canvasWrapEl) {
        if (introActive) {
          canvasWrapEl.style.opacity = String(0.18 + 0.82 * introProg);
        } else if (canvasWrapEl.style.opacity) {
          canvasWrapEl.style.opacity = "";
        }
      }

      if (introActive) {
        Object.keys(g.nodeMeshById || {}).forEach((nodeId) => {
          const mesh = g.nodeMeshById[nodeId];
          if (!mesh?.userData?.basePos || !mesh.userData.geomIntroFrom) return;
          mesh.position.lerpVectors(mesh.userData.geomIntroFrom, mesh.userData.basePos, introProg);
          const halo = g.nodeGlowById[nodeId];
          if (halo) halo.position.copy(mesh.position);
        });
        syncFlowEdgesFromMeshes(g, three, t);
        controls.update();
        renderer.render(scene, camera);
        geometryFrameIndex += 1;
        return;
      }

      const coherence = typeof g.flowCoherence === "number" ? g.flowCoherence : 0.75;
      const driftN = typeof g.flowDriftN === "number" ? g.flowDriftN : 0;
      const instN = typeof g.flowInstN === "number" ? g.flowInstN : 0;
      const unifiedPhase = t * 0.88;
      const ms = typeof g.motionScale === "number" ? g.motionScale : 1;
      const planeLayout = Boolean(g.structuralFlowPlaneLayout);
      const narrowFlow = geometryViewportIsNarrow();
      const planeCalm = 0.32;
      const rigMul = narrowFlow ? 0.42 : 1;

      Object.keys(g.nodeMeshById || {}).forEach((nodeId) => {
        const mesh = g.nodeMeshById[nodeId];
        const nodeData = g.nodeDataById[nodeId];
        if (!mesh?.userData?.basePos || !nodeData) return;
        const base = mesh.userData.basePos;
        const stress = flowNodeStress01(nodeData);
        const unstable = Boolean(nodeData.is_unstable);
        const phase = mesh.userData.jitterSeed || 0;
        if (planeLayout) {
          const inRange = flowNodeInRange(nodeData);
          const liftHint = Math.min(1, base.y * 2.4);
          const cohWobble = (1 - coherence) * 0.06;
          const wobble =
            (0.012 + cohWobble) *
            (0.55 + 0.45 * coherence) *
            ms *
            planeCalm *
            (inRange ? 1 : 0.55 + liftHint * 0.45);
          const breatheX = Math.sin(unifiedPhase * 0.85 + phase * 0.08) * wobble;
          const breatheZ = Math.cos(unifiedPhase * 0.8 + phase * 0.08) * wobble;
          const breatheY = Math.sin(unifiedPhase * 0.68 + phase * 0.06) * 0.003 * ms * (0.35 + 0.65 * coherence);
          if (inRange) {
            mesh.position.x = base.x + breatheX;
            mesh.position.z = base.z + breatheZ;
            mesh.position.y = base.y + breatheY + Math.sin(unifiedPhase * 0.82 + phase) * 0.002 * ms;
          } else {
            const decouple = 0.5 + 0.5 * (1 - coherence);
            const asyncAmp = 0.008 * ms * decouple * (0.45 + liftHint * 0.55);
            mesh.position.x =
              base.x + breatheX * (1 - decouple * 0.82) + Math.sin(t * 1.05 + phase * 2.1) * asyncAmp;
            mesh.position.z =
              base.z + breatheZ * (1 - decouple * 0.82) + Math.cos(t * 1.02 + phase * 1.8) * asyncAmp;
            mesh.position.y =
              base.y +
              breatheY * 0.45 +
              Math.sin(t * 0.92 + phase * 1.3) * 0.008 * ms * decouple * (0.4 + liftHint * 0.6);
          }
          return;
        }
        const inRange = flowNodeInRange(nodeData);
        const cohMul = inRange ? 1 : 0.55 + 0.45 * stress;
        const breathe =
          Math.sin(unifiedPhase) * 0.038 * coherence * (1 - stress * 0.85) * (1 - driftN * 0.4) * cohMul;
        const breatheY =
          Math.cos(unifiedPhase * 0.97) * 0.032 * coherence * (1 - stress * 0.85) * cohMul;
        const chaos =
          (0.04 + stress * 0.42 + (unstable ? 0.12 : 0) + driftN * 0.14 + instN * 0.1) *
          (1 - coherence * 0.72) *
          (inRange ? 1 : 1.15 + (1 - coherence) * 0.35);
        const sep = (1 - coherence) * 0.08 * stress;
        mesh.position.x =
          base.x + (breathe + Math.sin(t * 1.65 + phase) * chaos + sep * Math.sin(phase)) * ms;
        mesh.position.y =
          base.y + (breatheY + Math.cos(t * 1.25 + phase) * chaos * 0.92) * ms;
        mesh.position.z =
          base.z + (Math.sin(t * 1.05 + phase * 1.1) * chaos + sep * Math.cos(phase)) * ms;
      });

      syncFlowEdgesFromMeshes(g, three, t);

      Object.keys(g.nodeMeshById || {}).forEach((nodeId) => {
        const root = g.nodeMeshById[nodeId];
        const ring = root?.userData?.sensorRing;
        const cage = root?.userData?.sensorCage;
        const inner = root?.userData?.innerCore;
        const nodeData = g.nodeDataById[nodeId];
        if (!nodeData || !root) return;
        const k = Number(g.unstablePulseById[nodeId] || 1);
        const stress = flowNodeStress01(nodeData);
        const inRange = flowNodeInRange(nodeData);
        const perfN = Boolean(g.perfMode);
        if (ring) {
          ring.rotation.z += 0.0065 * (0.85 + stress * 0.45);
          const breathe = 0.045 * (inRange ? 0.72 : 1.12) * (0.45 + 0.55 * coherence);
          ring.scale.setScalar(1 + breathe * Math.sin(t * 1.88 + k) + (inRange ? 0 : 0.05) * Math.sin(t * 3.1 + stress * 4));
        }
        if (cage?.material) {
          cage.material.opacity = Math.min(
            0.62,
            (perfN ? 0.3 : 0.48) + stress * 0.14 + (inRange ? 0 : 0.16) + 0.06 * Math.sin(t * 2.2 + k * 0.2),
          );
        }
        if (inner?.material) {
          if (inner.material.emissiveIntensity != null) {
            const pulse = 0.36 + stress * 0.42 * (inRange ? 0.38 : 1);
            inner.material.emissiveIntensity = pulse + 0.1 * Math.sin(t * 2.9 + k);
          } else if (inner.material.opacity != null) {
            inner.material.opacity = Math.min(0.98, 0.82 + 0.12 * Math.sin(t * 2.4 + k));
          }
        }
      });

      if (g.sensorDatumShell) {
        g.sensorDatumShell.rotation.y += 0.0028 * ms;
        g.sensorDatumShell.rotation.x += 0.001 * ms * Math.sin(t * 0.22);
      }

      const rig = g.geometryRig;
      if (rig) {
        if (!g.reducedGeometryMotion) {
          const sway = 0.034 * ms * rigMul;
          rig.rotation.y = Math.sin(t * 0.38) * sway;
          rig.rotation.x = Math.sin(t * 0.29 + 0.85) * sway * 0.35;
          rig.rotation.z = Math.sin(t * 0.24 + 1.1) * sway * 0.18;
        } else {
          rig.rotation.set(0, 0, 0);
        }
      }

      if (g.keyLight && g.keyLightBasePos && !g.perfMode && !g.reducedGeometryMotion) {
        const bp = g.keyLightBasePos;
        const kl = g.keyLight;
        kl.position.set(
          bp.x + Math.sin(t * 0.33) * 0.45,
          bp.y + Math.sin(t * 0.37 + 0.6) * 0.24,
          bp.z + Math.cos(t * 0.35) * 0.4,
        );
      }

      controls.update();
      if (geometryFrameIndex === 0 && typeof console !== "undefined" && console.debug) {
        console.debug("[geometry3d] first frame", {
          sceneChildren: scene.children.length,
          nodes: Object.keys(g.nodeMeshById || {}).length,
          edges: g.edgeRefs?.length || 0,
          cameraPos: camera.position.toArray(),
        });
      }
      geometryFrameIndex += 1;
      renderer.render(scene, camera);
    } catch (err) {
      if (typeof console !== "undefined" && console.error) {
        console.error("[geometry3d] frame/render error:", err);
      }
    }
  }
  animate();

  if (!perf) {
    const sceneRef = scene;
    const rendererRef = renderer;
    const cameraRef = camera;
    g.iblDeferredRaf = window.requestAnimationFrame(() => {
      g.iblDeferredRaf = null;
      const live = state.geometry3d;
      if (!live || live.scene !== sceneRef || !live.renderer) return;
      setupGeometryIBL(three, rendererRef, sceneRef, false);
      live.renderer.render(sceneRef, cameraRef);
    });
  }
}

async function loadRunGeometry(runId, resultId = null) {
  const statusEl = qs("#geometry3dStatus");
  setGeometryViewportLoading(true);
  if (statusEl) {
    statusEl.textContent = "Loading geometry…";
    statusEl.className = "geometry-status info";
    statusEl.classList.remove("hidden");
  }
  const params = tenantScopeParams(resultId != null ? { result_id: resultId } : {});
  let payload;
  try {
    const fetchPromise = fetchJson(
      apiUrl(`/runs/${encodeURIComponent(runId)}/geometry`, params)
    );
    await Promise.all([fetchPromise, ensureThreeModulesLoaded()]);
    payload = await fetchPromise;
  } catch (_err) {
    clearGeometryModelsPanel();
    if (statusEl) statusEl.classList.add("hidden");
    setGeometrySurfaceState("Geometry unavailable for this snapshot yet. Structural charts remain active.", "info");
    const fallback = qs("#geometryFallback");
    if (fallback) {
      fallback.textContent = "Structural geometry is unavailable for this snapshot. Continue with structural drift, instability trend, and operational recommendation.";
      fallback.classList.remove("hidden");
    }
    setGeometryViewportLoading(false);
    return;
  }
  state.runGeometry = payload;
  renderGeometryModelsPanel(payload);
  const projectionNote =
    payload?.projection?.note ||
    "Structural projection metadata was not returned for this result.";
  const fallback = qs("#geometryFallback");
  const summary = qs("#geometryStructureSummary");
  if (fallback) {
    fallback.setAttribute("title", projectionNote);
    if (!payload?.available) {
      fallback.textContent = payload?.reason || "No structural relationship graph is available for this snapshot.";
    }
  }
  if (summary) summary.textContent = geometryStructureSummary(payload);
  try {
    const dims = await ensureGeometryViewportReady();
    if (!dims || dims.width <= 40 || dims.height <= 40) {
      setGeometrySurfaceState("Viewport has no size yet. Resize the window or refresh.", "info");
      updateGeometryDetails(null);
      if (statusEl) statusEl.classList.add("hidden");
      setGeometryViewportLoading(false);
      return;
    }
    await new Promise((resolve, reject) => {
      scheduleGeometrySceneBuild(() => {
        try {
          renderGeometryScene(payload, dims);
          resolve();
        } catch (err) {
          reject(err);
        }
      });
    });
    if (statusEl) statusEl.classList.add("hidden");
    setGeometryViewportLoading(false);
  } catch (err) {
    setGeometrySurfaceState(`3D view failed: ${String(err.message || err)}`, "error");
    updateGeometryDetails(null);
    if (statusEl) statusEl.classList.add("hidden");
    setGeometryViewportLoading(false);
  }
}


