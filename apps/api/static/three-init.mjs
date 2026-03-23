/**
 * Loads Three.js and addons as ES modules; exposes globals for app.js (classic script).
 *
 * Core uses an absolute URL so it does not depend on import maps (older Safari / edge cases).
 * Addon files still import `three` internally; the import map in index.html must load in <head>.
 */
import * as THREE from "/web/vendor/three/build/three.module.js";

window.__THREE_ESM = THREE;

try {
  const { OrbitControls } = await import(
    "/web/vendor/three/examples/jsm/controls/OrbitControls.js",
  );
  const { RoomEnvironment } = await import(
    "/web/vendor/three/examples/jsm/environments/RoomEnvironment.js",
  );
  const { Line2 } = await import("/web/vendor/three/examples/jsm/lines/Line2.js");
  const { LineGeometry } = await import("/web/vendor/three/examples/jsm/lines/LineGeometry.js");
  const { LineMaterial } = await import("/web/vendor/three/examples/jsm/lines/LineMaterial.js");
  const { PMREMGenerator } = await import("/web/vendor/three/examples/jsm/utils/PMREMGenerator.js");

  window.__OrbitControls = OrbitControls;
  window.__RoomEnvironment = RoomEnvironment;
  window.__PMREMGenerator = PMREMGenerator;
  window.__Line2 = Line2;
  window.__LineGeometry = LineGeometry;
  window.__LineMaterial = LineMaterial;
} catch (err) {
  console.error(
    "three-init: addon modules failed (jsm files import bare 'three'; ensure import map in <head>):",
    err,
  );
}
