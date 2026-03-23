/**
 * Loads Three.js and OrbitControls as ES modules and exposes them for the
 * non-module app bundle (app.js, sensor_graph_3d.js) via window.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { PMREMGenerator } from "three/addons/utils/PMREMGenerator.js";

window.__THREE_ESM = THREE;
window.__OrbitControls = OrbitControls;
window.__RoomEnvironment = RoomEnvironment;
window.__PMREMGenerator = PMREMGenerator;
/** Wide structural links (triangle strips, not GL line width). */
window.__Line2 = Line2;
window.__LineGeometry = LineGeometry;
window.__LineMaterial = LineMaterial;
