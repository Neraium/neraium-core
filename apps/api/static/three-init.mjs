/**
 * Loads Three.js + addons from jsDelivr (ESM). Same version as download_three_vendor.py.
 * Does not rely on the API serving /web/vendor/three — only /web/three-init.mjs must load.
 */
import * as THREE from "three";

window.__THREE_ESM = THREE;

try {
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const { RoomEnvironment } = await import("three/addons/environments/RoomEnvironment.js");
  const { Line2 } = await import("three/addons/lines/Line2.js");
  const { LineGeometry } = await import("three/addons/lines/LineGeometry.js");
  const { LineMaterial } = await import("three/addons/lines/LineMaterial.js");
  const { PMREMGenerator } = await import("three/addons/utils/PMREMGenerator.js");

  window.__OrbitControls = OrbitControls;
  window.__RoomEnvironment = RoomEnvironment;
  window.__PMREMGenerator = PMREMGenerator;
  window.__Line2 = Line2;
  window.__LineGeometry = LineGeometry;
  window.__LineMaterial = LineMaterial;
} catch (err) {
  console.error("three-init: addon modules failed (CDN / network / import map):", err);
}
