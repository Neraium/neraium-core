/**
 * Loads Three.js and OrbitControls as ES modules and exposes them for the
 * non-module app bundle (app.js, sensor_graph_3d.js) via window.
 *
 * The core `three` build is assigned to `window.__THREE_ESM` immediately after it
 * loads. Addon imports are loaded separately so a failing addon (404/MIME) does
 * not prevent the core library from being available (avoids a misleading
 * "download_three_vendor" error when only three.module.js was OK).
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
  /** Wide structural links (triangle strips, not GL line width). */
  window.__Line2 = Line2;
  window.__LineGeometry = LineGeometry;
  window.__LineMaterial = LineMaterial;
} catch (err) {
  console.error(
    "three-init: addon modules failed to load (check Network for /web/vendor/three/examples/jsm/…):",
    err,
  );
}
