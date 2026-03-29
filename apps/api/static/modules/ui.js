(function attachUiModule(globalObj) {
  function qs(sel) {
    return document.querySelector(sel);
  }

  function qsa(sel) {
    return Array.from(document.querySelectorAll(sel));
  }

  function debounce(fn, wait = 120) {
    let timer = null;
    return (...args) => {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => fn(...args), wait);
    };
  }

  function animateNumberText(el, target, opts = {}) {
    if (!el) return;
    const decimals = Number.isFinite(opts.decimals) ? opts.decimals : 0;
    const suffix = String(opts.suffix || "");
    const safeTarget = Number.isFinite(target) ? target : 0;
    const from = Number.parseFloat(el.dataset.currentNumber || "0") || 0;
    const duration = Number.isFinite(opts.durationMs) ? opts.durationMs : 420;
    const start = performance.now();
    const step = (ts) => {
      const t = Math.min(1, (ts - start) / duration);
      const eased = 1 - (1 - t) ** 3;
      const val = from + (safeTarget - from) * eased;
      el.textContent = `${val.toFixed(decimals)}${suffix}`;
      el.dataset.currentNumber = String(val);
      if (t < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
  }

  function friendlyErrorMessage(err) {
    const msg = String(err?.message || err || "");
    if (msg.toLowerCase().includes("network")) return "Connection interrupted — attempting telemetry recovery.";
    if (msg.toLowerCase().includes("timeout")) return "Telemetry request timed out — re-establishing stream.";
    if (msg.toLowerCase().includes("demo")) return "Reference replay interrupted — retry available.";
    return "Operational request interrupted — retry when ready.";
  }

  function toPretty(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(2) : "-";
    return String(v);
  }

  function formatBytes(bytes) {
    const n = Number(bytes || 0);
    if (!Number.isFinite(n) || n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let value = n;
    let i = 0;
    while (value >= 1024 && i < units.length - 1) {
      value /= 1024;
      i += 1;
    }
    const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
    return `${value.toFixed(digits)} ${units[i]}`;
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  globalObj.NeraiumUI = {
    qs,
    qsa,
    debounce,
    animateNumberText,
    friendlyErrorMessage,
    toPretty,
    formatBytes,
    escapeHtml,
  };
})(window);
