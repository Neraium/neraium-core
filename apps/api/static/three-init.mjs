/**
 * Loads Three.js and OrbitControls as ES modules and exposes them for the
 * non-module app bundle (app.js, sensor_graph_3d.js) via window.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

window.__THREE_ESM = THREE;
window.__OrbitControls = OrbitControls;
