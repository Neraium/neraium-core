import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function renderThreeScene() {
  const shell = document.createElement("div");
  shell.style.position = "fixed";
  shell.style.inset = "auto 12px 12px auto";
  shell.style.width = "140px";
  shell.style.height = "140px";
  shell.style.opacity = "0.35";
  document.body.appendChild(shell);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(140, 140);
  shell.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = false;
  const geometry = new THREE.IcosahedronGeometry(1.4, 0);
  const material = new THREE.MeshBasicMaterial({ color: 0x2e7d4b, wireframe: true });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
  camera.position.z = 3.5;
  function animate() {
    mesh.rotation.x += 0.003;
    mesh.rotation.y += 0.006;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
}
