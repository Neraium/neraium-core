// Main app module
import { createToast, riskBadgeHtml, phaseBadgeHtml } from './modules/dashboard.js';
import { initGeometry } from './modules/geometry.js';
import { initValidation } from './modules/validation.js';
import { initThreeScene } from './three-init.mjs';

// Initialize the application
export function createToast(message) {
  console.log('Toast:', message);
}

export function riskBadgeHtml(risk) {
  return `<span class="badge badge-risk">${risk}</span>`;
}

export function phaseBadgeHtml(phase) {
  return `<span class="badge badge-phase">${phase}</span>`;
}

// Initialize modules
document.addEventListener('DOMContentLoaded', () => {
  initGeometry();
  initValidation();
  initThreeScene();
});
