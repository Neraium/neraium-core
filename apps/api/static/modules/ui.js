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

  function friendlyErrorMessage(err, context = "analysis") {
    const rawMessage = String(err?.message || err || "");
    const msg = rawMessage.toLowerCase();
    const normalizedContext = String(context || "analysis").toLowerCase();
    const networkText = msg.includes("network")
      ? " Connection to the service was interrupted."
      : msg.includes("timeout")
        ? " The request timed out before completion."
        : "";
    if (normalizedContext === "replay" || msg.includes("replay") || msg.includes("demo") || msg.includes("validation")) {
      if (msg.includes("core_runtime_unavailable") || msg.includes("fallback mode is active") || msg.includes("analysis engine is unavailable")) {
        return `Analysis engine unavailable: runtime is degraded/fallback. Restore full core runtime and retry replay.${networkText}`;
      }
      if (msg.includes("no analysis") || msg.includes("materialized")) {
        return `Replay completed but no analysis was generated. Increase ingest window or verify runtime health.${networkText}`;
      }
      if (msg.includes("temporarily unavailable") || msg.includes("retrying status check")) {
        return `Replay is still initializing; status visibility is temporarily delayed.${networkText}`;
      }
      return `Validation replay could not complete yet. Check replay status and retry only if this persists.${networkText}`;
    }
    if (normalizedContext === "ingest" || msg.includes("upload") || msg.includes("ingest") || msg.includes("csv")) {
      return `Telemetry ingest did not complete. Upload fresh data and retry.${networkText}`;
    }
    if (normalizedContext === "geometry" || msg.includes("geometry") || msg.includes("structural view")) {
      return `Structural view is unavailable for the current snapshot.${networkText}`;
    }
    return `Analysis could not complete. Refresh the run or try again.${networkText}`;
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
